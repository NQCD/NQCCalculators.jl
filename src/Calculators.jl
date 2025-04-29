
mutable struct DependentField{V,R}
    value::V
    position::R
end
needsupdate(field::DependentField, r) = field.position != r

function Base.getproperty(cache::Abstract_Cache, name::Symbol)
    field = getfield(cache, name)
    if field isa DependentField
        return field.value
    else
        return field
    end
end

function Base.setproperty!(cache::Abstract_Cache, name::Symbol, x)
    field = getfield(cache, name)
    if field isa DependentField
        setfield!(field, :value, x)
    else
        setfield!(cache, name, x)
    end
end

"""
Each of the quantities specified here has functions:
`get_quantity(calculator, r)`
`evaluate_quantity!(calculator, r)!`

The user should access only the former.
This will ensure quantities are correctly evaluated and cached accordingly.

The latter is called by the former and is where the details required to calculate the quantity are found.
"""
const quantities = [
    :potential,
    :derivative,
    :eigen,
    :adiabatic_derivative,
    :nonadiabatic_coupling,

    :traceless_potential,
    :V̄,
    :traceless_derivative,
    :D̄,
    :traceless_adiabatic_derivative,

    :centroid,
    :centroid_potential,
    :centroid_derivative,
    :centroid_eigen,
    :centroid_adiabatic_derivative,
    :centroid_nonadiabatic_coupling,

    :friction,
]

for quantity in quantities
    get_quantity = Symbol(:get_, quantity)
    evaluate_quantity! = Symbol(:evaluate_, quantity, :!)
    field = Expr(:call, :getfield, :calculator, QuoteNode(quantity))

    @eval function $(get_quantity)(calculator, r)

        if needsupdate($field, r)
            copyto!($field.position, r)
            $(evaluate_quantity!)(calculator, r)
        end
        return $field.value
    end
end

#evaluate functions

function evaluate_potential!(cache::Abstract_Cache, R)
    cache.stats[:potential] += 1
    cache.potential = NQCModels.potential(cache.model, R)
    return nothing
end

function evaluate_potential!(cache::Abstract_Cache, R::AbstractArray{T,3}) where {T}
    cache.stats[:potential] += 1
    @views for i in axes(R, 3)
        cache.potential[i] = NQCModels.potential(cache.model, R[:,:,i])
    end
    return nothing
end

function evaluate_V̄!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:V̄] += 1
    potential = get_potential(cache, r)
    for i in 1:length(cache.V̄)
        cache.V̄[i] = tr(potential[i]) / nstates(cache.model)
    end
end

function evaluate_traceless_potential!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:traceless_potential] += 1
    n = nstates(cache.model)
    potential = get_potential(cache, r)
    V̄ = get_V̄(cache, r)
    for I in eachindex(potential)
        cache.traceless_potential[I] = Hermitian(SMatrix{n,n}(
            i != j ? potential[I][j,i] : potential[I][j,i] - V̄[I] for j=1:n, i=1:n
        ))
    end
end

function evaluate_centroid!(cache::Abstract_Cache, r::AbstractArray{T,3}) where {T}
    cache.stats[:centroid] += 1
    get_centroid!(cache.centroid, r)
end

function evaluate_centroid_potential!(cache::Abstract_Cache, r::AbstractArray{T,3}) where {T}
    cache.stats[:centroid_potential] += 1
    centroid = get_centroid(cache, r)
    cache.centroid_potential = NQCModels.potential(cache.model, centroid)
end

function evaluate_derivative!(cache::Abstract_Cache, R)
    cache.stats[:derivative] += 1
    NQCModels.derivative!(cache.model, cache.derivative, R)
end

function evaluate_derivative!(cache::Abstract_Cache, R::AbstractArray{T,3}) where {T}
    cache.stats[:derivative] += 1
    @views for i in axes(R, 3)
        NQCModels.derivative!(cache.model, cache.derivative[:,:,i], R[:,:,i])
    end
end

function evaluate_D̄!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:D̄] += 1
    derivative = get_derivative(cache, r)
    for I in eachindex(derivative)
        cache.D̄[I] = tr(derivative[I]) / nstates(cache.model)
    end
end

function evaluate_traceless_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:traceless_derivative] += 1
    n = nstates(cache.model)
    derivative = get_derivative(cache, r)
    D̄ = get_D̄(cache, r)
    for I in eachindex(derivative)
        cache.traceless_derivative[I] = Hermitian(SMatrix{n,n}(
            i != j ? derivative[I][j,i] : derivative[I][j,i] - D̄[I] for j=1:n, i=1:n
        ))
    end
end

function evaluate_traceless_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:traceless_adiabatic_derivative] += 1
    n = nstates(cache.model)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    D̄ = get_D̄(cache, r)
    for I in eachindex(D̄)
        cache.traceless_adiabatic_derivative[I] = SMatrix{n,n}(
            i != j ? adiabatic_derivative[I][j,i] : adiabatic_derivative[I][j,i] - D̄[I] for j=1:n, i=1:n
        )
    end
end

function evaluate_centroid_derivative!(cache::Abstract_Cache, r::AbstractArray{T,3}) where {T}
    cache.stats[:centroid_derivative] += 1
    centroid = get_centroid(cache, r)
    NQCModels.derivative!(cache.model, cache.centroid_derivative, centroid)
end

function evaluate_eigen!(cache::Abstract_QuantumModel_Cache, r)
    cache.stats[:eigen] += 1
    potential = get_potential(cache, r)
    eig = LinearAlgebra.eigen(potential)
    corrected_vectors = correct_phase(eig.vectors, cache.eigen.vectors)
    cache.eigen = Eigen(eig.values, corrected_vectors)
    return nothing
end

function evaluate_centroid_eigen!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:centroid_eigen] += 1
    potential = get_centroid_potential(cache, r)
    eig = LinearAlgebra.eigen(potential)
    corrected_vectors = correct_phase(eig.vectors, cache.centroid_eigen.vectors)
    cache.centroid_eigen = Eigen(eig.values, corrected_vectors)
    return nothing
end

function evaluate_eigen!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:eigen] += 1
    potential = get_potential(cache, r)
    for i=1:length(potential)
        eig = LinearAlgebra.eigen(potential[i])
        corrected_vectors = correct_phase(eig.vectors, cache.eigen[i].vectors)
        cache.eigen[i] = Eigen(eig.values, corrected_vectors)
    end
    return nothing
end

function correct_phase(new_vectors::SMatrix, old_vectors::SMatrix)
    n = size(new_vectors, 1)
    vect = SVector{n}(sign(LinearAlgebra.dot(new_vectors[:,i], old_vectors[:,i])) for i=1:n)
    return new_vectors .* vect'
end

function evaluate_adiabatic_derivative!(cache::Abstract_QuantumModel_Cache, r)
    cache.stats[:adiabatic_derivative] += 1
    U = get_eigen(cache, r).vectors
    diabatic_derivative = get_derivative(cache, r)
    for I in eachindex(diabatic_derivative)
        cache.adiabatic_derivative[I] = U' * diabatic_derivative[I] * U
    end
end

function evaluate_centroid_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:centroid_adiabatic_derivative] += 1
    centroid_derivative = get_centroid_derivative(cache, r)
    centroid_eigen = get_centroid_eigen(cache, r)
    for I in eachindex(centroid_derivative)
        cache.centroid_adiabatic_derivative[I] = centroid_eigen.vectors' * centroid_derivative[I] * centroid_eigen.vectors
    end
end

function evaluate_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:adiabatic_derivative] += 1
    derivative = get_derivative(cache, r)
    eigen = get_eigen(cache, r)
    for i in axes(derivative, 3) # Beads
        for j in axes(derivative, 2) # Atoms
            for k in axes(derivative, 1) # DoFs
                cache.adiabatic_derivative[k,j,i] = eigen[i].vectors' * derivative[k,j,i] * eigen[i].vectors
            end
        end
    end
end

function evaluate_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r)
    cache.stats[:nonadiabatic_coupling] += 1
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    eigen = get_eigen(cache, r)
    for I in eachindex(cache.adiabatic_derivative)
        cache.nonadiabatic_coupling[I] = evaluate_nonadiabatic_coupling(adiabatic_derivative[I], eigen.values)
    end
end

function evaluate_centroid_nonadiabatic_coupling!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:centroid_nonadiabatic_coupling] += 1
    centroid_adiabatic_derivative = get_centroid_adiabatic_derivative(cache, r)
    centroid_eigen = get_centroid_eigen(cache, r)
    for I in eachindex(centroid_adiabatic_derivative)
        cache.centroid_nonadiabatic_coupling[I] = evaluate_nonadiabatic_coupling(centroid_adiabatic_derivative[I], centroid_eigen.values)
    end
end

function evaluate_nonadiabatic_coupling!(cache::RingPolymer_QuantumModel_Cache, r)
    cache.stats[:nonadiabatic_coupling] += 1
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    eigen = get_eigen(cache, r)
    for i in eachindex(eigen) # Beads
        for I in CartesianIndices(size(adiabatic_derivative)[1:2])
            cache.nonadiabatic_coupling[I,i] = evaluate_nonadiabatic_coupling(adiabatic_derivative[I,i], eigen[i].values)
        end
    end
end

"""
# References

- HammesSchifferTully_JChemPhys_101_4657_1994 Eq. (32)
- SubotnikBellonzi_AnnuRevPhyschem_67_387_2016, section 2.3
"""
function evaluate_nonadiabatic_coupling(adiabatic_derivative::SMatrix, eigenvalues::SVector)
    n = length(eigenvalues)
    SMatrix{n,n}(
        (i != j ? adiabatic_derivative[j,i] / (eigenvalues[i] - eigenvalues[j]) : 0
        for j=1:n, i=1:n))
end

#update functions
"""
Evaluates all electronic properties for the current position `r`.
# Properties evaluated:
- Diabatic potential
- Diabatic derivative
- Eigenvalues and eigenvectors
- Adiabatic derivative
- Nonadiabatic coupling

This should no longer be used, instead access the quantities directly with `get_quantity(cache, r)`.
"""
function update_electronics!(calculator::Abstract_QuantumModel_Cache, r::AbstractArray)
    get_nonadiabatic_coupling(calculator, r)
    return nothing
end

function update_electronics!(calculator::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    get_nonadiabatic_coupling(calculator, r)
    update_centroid_electronics!(calculator, r)
    return nothing
end

function update_centroid_electronics!(calculator::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    get_centroid_nonadiabatic_coupling(calculator, r)
    return nothing
end


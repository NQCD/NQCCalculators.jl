#Definitions of base NQCModels functions on an Abstract Cache.
NQCModels.nstates(cache::Abstract_Cache) = NQCModels.nstates(cache.model)
NQCModels.eachstate(cache::Abstract_Cache) = NQCModels.eachstate(cache.model)
NQCModels.nelectrons(cache::Abstract_Cache) = NQCModels.nelectrons(cache.model)
NQCModels.eachelectron(cache::Abstract_Cache) = NQCModels.eachelectron(cache.model)
NQCModels.mobileatoms(cache::Abstract_Cache) = NQCModels.mobileatoms(cache.model, size(cache.derivative, 2))
NQCModels.dofs(cache::Abstract_Cache) = NQCModels.dofs(cache.model)
NQCModels.fermilevel(cache::Abstract_Cache) = NQCModels.fermilevel(cache.model)
beads(cache) = Base.OneTo(length(cache.potential))
Base.eltype(::Abstract_Cache{T}) where {T} = T

"""
Each of the quantities specified here has functions:
`get_quantity(cache, r)`
`evaluate_quantity!(cache, r)!`

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

#= """
    function: needsupdate(field::DependentField, r)

Function to test if the value stored in the DependentField mutable struct needs to be updated at the given time-step in a dynamics simulation
"""
needsupdate(field::DependentField, r) = field.position != r
=#

for quantity in quantities
    get_quantity = Symbol(:get_, quantity)
    evaluate_quantity! = Symbol(:evaluate_, quantity, :!)
    field = Expr(:call, :getfield, :cache, QuoteNode(quantity))

    @eval function $(get_quantity)(cache, r)
        return $field
    end
end

#Definitions of evaluate functions

function evaluate_potential!(cache::Abstract_ClassicalModel_Cache, r::AbstractMatrix)
    cache.potential .= hcat(NQCModels.potential(cache.model, r))
end

function evaluate_potential!(cache::Abstract_ClassicalModel_Cache, r::AbstractArray{T,3}) where {T}
    @views @inbounds for i in beads(cache)
        cache.potential[i] .= hcat(NQCModels.potential(cache.model, r[:,:,i]))
    end
    return nothing
end

function evaluate_potential!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    NQCModels.potential!(cache.model, cache.potential, r)
end

function evaluate_potential!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    @views @inbounds for i in beads(cache)
        NQCModels.potential!(cache.model, cache.potential[i], r[:,:,i])
    end
    return nothing
end

function evaluate_derivative!(cache::Abstract_ClassicalModel_Cache, R::AbstractMatrix)
    NQCModels.derivative!(cache.model, cache.derivative, R::AbstractMatrix)
end

function evaluate_derivative!(cache::Abstract_ClassicalModel_Cache, r::AbstractArray{T,3}) where {T}
    @views @inbounds for i in beads(cache)
        NQCModels.derivative!(cache.model, cache.derivative[:,:,i], r[:,:,i])
    end
    return nothing
end

function evaluate_derivative!(cache::Abstract_QuantumModel_Cache, R::AbstractMatrix)
    NQCModels.derivative!.(Ref(cache.model), cache.derivative, Ref(R::AbstractMatrix))
end

function evaluate_derivative!(cache::Abstract_QuantumModel_Cache, R::AbstractArray{T,3}) where {T}
    @views @inbounds for i in beads(cache)
        NQCModels.derivative!.(Ref(cache.model), cache.derivative[:,:,i], Ref(R[:,:,i]))
    end
    return nothing
end

function evaluate_friction!(cache::Abstract_Cache, R::AbstractMatrix)
    NQCModels.friction!(cache.model, cache.friction, R::AbstractMatrix)
end

function evaluate_friction!(cache::Abstract_Cache, R::AbstractArray{T,3}) where {T}
    @views for i in axes(R, 3)
        NQCModels.friction!(cache.model, cache.friction[:,:,i], R[:,:,i])
    end
    return nothing
end

function evaluate_centroid_friction!(cache::Abstract_Cache, R::AbstractArray{T,3}) where {T}
    centroid = RingPolymerArrays.get_centroid(R)
    NQCModels.friction!(cache.model, cache.centroid_friction, centroid)
end

#= 
    function correct_phase(new_vectors::SMatrix, old_vectors::SMatrix)
    n = size(new_vectors, 1)
    vect = SVector{n}(sign(LinearAlgebra.dot(new_vectors[:,i], old_vectors[:,i])) for i=1:n)
    return new_vectors .* vect'
end
 =#

function correct_phase!(eig::LinearAlgebra.Eigen, old_eigenvectors::AbstractMatrix)
    @views for i in 1:length(eig.values)
        eig.vectors[:,i] .*= sign(LinearAlgebra.dot(eig.vectors[:,i], old_eigenvectors[:,i]))
    end
    return nothing
end

#= function correct_phase!(eig::LinearAlgebra.Eigen, old_eigenvectors::AbstractMatrix)
    @views for i=1:length(eig.values)
        if LinearAlgebra.dot(eig.vectors[:,i], old_eigenvectors[:,i]) < 0
            eig.vectors[:,i] .*= -1
        end
    end
end =#

function evaluate_eigen!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    potential = get_potential(cache, r)
    eig = LinearAlgebra.eigen(potential)
    correct_phase!(eig, cache.eigen.vectors)
    cache.eigen.values .= eig.values
    cache.eigen.vectors .= eig.vectors
end

function evaluate_eigen!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    potential = get_potential(cache, r)

    @inbounds for i in beads(cache)
        eig = LinearAlgebra.eigen(potential[i])
        correct_phase!(eig, cache.eigen[i].vectors)
        cache.eigen[i].values .= eig.values
        cache.eigen[i].vectors .= eig.vectors
    end
    return nothing
end

function evaluate_adiabatic_derivative!(cache::Abstract_QuantumModel_Cache, r)
    U = get_eigen(cache, r).vectors
    diabatic_derivative = get_derivative(cache, r)
    for I in eachindex(diabatic_derivative)
        cache.adiabatic_derivative[I] = U' * diabatic_derivative[I] * U
    end
    return nothing
end

function evaluate_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    derivative = get_derivative(cache, r)
    eigen = get_eigen(cache, r)
    for i in axes(derivative, 3) # Beads
        for j in axes(derivative, 2) # Atoms
            for k in axes(derivative, 1) # DoFs
                cache.adiabatic_derivative[k,j,i] = eigen[i].vectors' * derivative[k,j,i] * eigen[i].vectors
            end
        end
    end
    return nothing
end

function evaluate_centroid_adiabatic_derivative!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T, 3}) where {T}
    centroid_derivative = get_centroid_derivative(cache, r)
    centroid_eigen = get_centroid_eigen(cache, r)
    for I in eachindex(centroid_derivative)
        cache.centroid_adiabatic_derivative[I] .= centroid_eigen.vectors' * centroid_derivative[I] * centroid_eigen.vectors
    end
    return nothing
end

function evaluate_adiabatic_derivative!(cache::QuantumModel_Cache, r)
    eigen = get_eigen(cache, r)
    derivative = get_derivative(cache, r)
    @inbounds for i in NQCModels.mobileatoms(cache)
        for j in NQCModels.dofs(cache)
            LinearAlgebra.mul!(cache.tmp_mat, derivative[j,i], eigen.vectors)
            LinearAlgebra.mul!(cache.adiabatic_derivative[j,i], eigen.vectors', cache.tmp_mat)
        end
    end
    return nothing
end

function evaluate_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    eigen = get_eigen(cache, r)
    derivative = get_derivative(cache, r)
    @inbounds for i in beads(cache)
        for j in mobileatoms(cache)
            for k in dofs(cache)
                LinearAlgebra.mul!(cache.tmp_mat, derivative[k,j,i], eigen[i].vectors)
                LinearAlgebra.mul!(cache.adiabatic_derivative[k,j,i], eigen[i].vectors', cache.tmp_mat)
            end
        end
    end
    return nothing
end


"""
# References

- HammesSchifferTully_JChemPhys_101_4657_1994 Eq. (32)
- SubotnikBellonzi_AnnuRevPhyschem_67_387_2016, section 2.3
"""
function evaluate_nonadiabatic_coupling(adiabatic_derivative::AbstractMatrix, eigenvalues::AbstractVector)
    n = length(eigenvalues)
    coupling_matrix = zeros(eltype(AbstractMatrix), n, n)
    @. coupling_matrix = adiabatic_derivative / (eigenvalues - eigenvalues')
  
    return coupling_matrix
end

function evaluate_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    eigen = get_eigen(cache, r)
    for I in eachindex(cache.adiabatic_derivative)
        cache.nonadiabatic_coupling[I] = evaluate_nonadiabatic_coupling(adiabatic_derivative[I], eigen.values)
    end
    return nothing
end

#= function evaluate_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    eigen = get_eigen(cache, r)
    for i in eachindex(eigen) # Beads
        for I in CartesianIndices(size(adiabatic_derivative)[1:2])
            cache.nonadiabatic_coupling[I,i] = evaluate_nonadiabatic_coupling(adiabatic_derivative[I,i], eigen[i].values)
        end
    end
end =#

function evaluate_nonadiabatic_coupling!(cache::QuantumModel_Cache, r)
    eigen = get_eigen(cache, r)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)

    evaluate_inverse_difference_matrix!(cache.tmp_mat, eigen.values)

    @inbounds for I in NQCModels.dofs(cache)
        @. cache.nonadiabatic_coupling[I] = adiabatic_derivative[I] * cache.tmp_mat
    end
    return nothing
end

function evaluate_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    eigen = get_eigen(cache, r)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)

    @inbounds for i in beads(cache)
        evaluate_inverse_difference_matrix!(cache.tmp_mat, eigen[i].values)
        for j in mobileatoms(cache)
            for k in dofs(cache)
                @. cache.nonadiabatic_coupling[k,j,i] = adiabatic_derivative[k,j,i] * cache.tmp_mat
            end
        end
    end
    return nothing
end

function evaluate_centroid_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    adiabatic_derivative = get_centroid_adiabatic_derivative(cache, r)
    eigen = get_centroid_eigen(cache, r)
    evaluate_inverse_difference_matrix!(cache.tmp_mat, eigen.values)

    @inbounds for j in mobileatoms(cache)
        for k in dofs(cache)
            @. cache.centroid_nonadiabatic_coupling[j,k] = adiabatic_derivative[j,k] * cache.tmp_mat
        end
    end
    return nothing
end

#= function evaluate_centroid_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T, 3}) where {T}
    centroid_adiabatic_derivative = get_centroid_adiabatic_derivative(cache, r)
    centroid_eigen = get_centroid_eigen(cache, r)
    for I in eachindex(centroid_adiabatic_derivative)
        @. cache.centroid_nonadiabatic_coupling[I] = evaluate_nonadiabatic_coupling(centroid_adiabatic_derivative[I], centroid_eigen.values)
    end
end =#

function evaluate_inverse_difference_matrix!(out, eigenvalues)
    @inbounds for i in eachindex(eigenvalues)
        for j in eachindex(eigenvalues)
            out[j,i] = 1 / (eigenvalues[i] - eigenvalues[j])
        end
        out[i,i] = zero(eltype(out))
    end
    return nothing
end

#RingPolymer specific functions
function evaluate_V̄!(cache::RingPolymer_QuantumModel_Cache, r)
    potential = get_potential(cache, r)
    for i in 1:length(cache.V̄)
        cache.V̄[i] = tr(potential[i]) / nstates(cache.model)
    end
    return nothing
end

function evaluate_traceless_potential!(cache::RingPolymer_QuantumModel_Cache, r)
    n = nstates(cache.model)
    potential = get_potential(cache, r)
    V̄ = get_V̄(cache, r)
    for I in eachindex(potential)
        cache.traceless_potential[I] = Hermitian([i != j ? potential[I][j,i] : potential[I][j,i] - V̄[I] for j=1:n, i=1:n])
    end
    return nothing
end

function evaluate_centroid!(cache::Abstract_Cache, r::AbstractArray{T,3}) where {T}
    cache.centroid .= RingPolymerArrays.get_centroid(r)
end

function evaluate_centroid_potential!(cache::Abstract_ClassicalModel_Cache, r::AbstractArray{T,3}) where {T}
    centroid = RingPolymerArrays.get_centroid(r)
    cache.centroid_potential .= hcat(NQCModels.potential(cache.model, centroid))
    return nothing
end

function evaluate_centroid_potential!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    centroid = RingPolymerArrays.get_centroid(r)
    NQCModels.potential!(cache.model, cache.centroid_potential, centroid)
    return nothing
end

function evaluate_D̄!(cache::RingPolymer_QuantumModel_Cache, r)
    derivative = get_derivative(cache, r)
    for I in eachindex(derivative)
        cache.D̄[I] = tr(derivative[I]) / nstates(cache.model)
    end
    return nothing
end

function evaluate_traceless_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    n = nstates(cache.model)
    derivative = get_derivative(cache, r)
    D̄ = get_D̄(cache, r)
    for I in eachindex(derivative)
        cache.traceless_derivative[I] = Hermitian([i != j ? derivative[I][j,i] : derivative[I][j,i] - D̄[I] for j=1:n, i=1:n])
    end
    return nothing
end

function evaluate_traceless_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r)
    n = nstates(cache.model)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    D̄ = get_D̄(cache, r)
    for I in eachindex(D̄)
        cache.traceless_adiabatic_derivative[I] = [i != j ? adiabatic_derivative[I][j,i] : adiabatic_derivative[I][j,i] - D̄[I] for j=1:n, i=1:n]
    end
    return nothing
end

function evaluate_centroid_derivative!(cache::Abstract_Cache, r::AbstractArray{T,3}) where {T}
    centroid = RingPolymerArrays.get_centroid(r)
    NQCModels.derivative!(cache.model, cache.centroid_derivative[1], centroid)
    return nothing
end

function evaluate_centroid_eigen!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    potential = get_centroid_potential(cache, r)
    eig = LinearAlgebra.eigen(potential)
    correct_phase!(eig, cache.centroid_eigen.vectors)
    cache.centroid_eigen.values .= eig.values
    cache.centroid_eigen.vectors .= eig.vectors
    return nothing
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
function update_electronics!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    evaluate_potential!(cache, r)
    evaluate_derivative!(cache, r)
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
    evaluate_nonadiabatic_coupling!(cache, r)
    return nothing
end

function update_electronics!(cache::QuantumFrictionModel_Cache, r::AbstractMatrix)
    evaluate_potential!(cache, r)
    evaluate_derivative!(cache, r)
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
    evaluate_nonadiabatic_coupling!(cache, r)
    evaluate_friction!(cache, r)
    return nothing
end

function update_electronics!(cache::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_potential!(cache, r)
    evaluate_derivative!(cache, r)
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
    evaluate_nonadiabatic_coupling!(cache, r)

    update_centroid_electronics!(cache, r)
    return nothing
end

function update_electronics!(cache::RingPolymer_QuantumFrictionModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_potential!(cache, r)
    evaluate_derivative!(cache, r)
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
    evaluate_nonadiabatic_coupling!(cache, r)
    evaluate_friction!(cache, r)

    update_centroid_electronics!(cache, r)
    return nothing
end

function update_centroid_electronics!(cache::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_centroid!(cache, r)
    evaluate_centroid_potential!(cache, r)
    evaluate_centroid_derivative!(cache, r)
    evaluate_centroid_eigen!(cache, r)
    evaluate_centroid_adiabatic_derivative!(cache, r)
    evaluate_centroid_nonadiabatic_coupling!(cache, r)
    return nothing
end

function update_centroid_electronics!(cache::RingPolymer_QuantumFrictionModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_centroid!(cache, r)
    evaluate_centroid_potential!(cache, r)
    evaluate_centroid_derivative!(cache, r)
    evaluate_centroid_eigen!(cache, r)
    evaluate_centroid_adiabatic_derivative!(cache, r)
    evaluate_centroid_nonadiabatic_coupling!(cache, r)
    evaluate_centroid_friction!(cache, r)
    return nothing
end
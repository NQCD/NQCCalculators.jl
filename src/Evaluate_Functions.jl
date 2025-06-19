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

function evaluate_derivative!(cache::Abstract_ClassicalModel_Cache, r::AbstractMatrix)
    NQCModels.derivative!(cache.model, cache.derivative, r::AbstractMatrix)
end

function evaluate_derivative!(cache::Abstract_ClassicalModel_Cache, r::AbstractArray{T,3}) where {T}
    @views @inbounds for i in beads(cache)
        NQCModels.derivative!(cache.model, cache.derivative[:,:,i], r[:,:,i])
    end
    return nothing
end

function evaluate_derivative!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    NQCModels.derivative!(cache.model, cache.derivative, r::AbstractMatrix)
end

function evaluate_derivative!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    @views @inbounds for i in beads(cache)
        NQCModels.derivative!(cache.model, cache.derivative[:,:,i], r[:,:,i])
    end
    return nothing
end

function evaluate_friction!(cache::Abstract_ClassicalModel_Cache, R::AbstractMatrix)
    NQCModels.friction!(cache.model, cache.friction, R::AbstractMatrix)
end

function evaluate_friction!(cache::Abstract_ClassicalModel_Cache, R::AbstractArray{T,3}) where {T}
    @views for i in axes(R, 3)
        NQCModels.friction!(cache.model, cache.friction[:,:,i], R[:,:,i])
    end
    return nothing
end

function evaluate_centroid_friction!(cache::Abstract_ClassicalModel_Cache, R::AbstractArray{T,3}) where {T}
    centroid = RingPolymerArrays.get_centroid(R)
    NQCModels.friction!(cache.model, cache.centroid_friction, centroid)
end

function evaluate_friction!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    μ = NQCModels.fermilevel(cache.model)
    if cache.friction_method isa WideBandExact
        potential = get_potential(cache, r)
        derivative = get_derivative(cache, r)
        fill_friction_tensor!(cache.friction, cache.friction_method, potential, derivative, r, μ)
    elseif cache.friction_method isa Nothing
        cache.friction .= zero(cache.friction)
    else
        ∂H = Calculators.get_adiabatic_derivative(sim.calculator, r)
        eigen = Calculators.get_eigen(sim.calculator, r)
        fill_friction_tensor!(cache.friction, sim.method.friction_method, ∂H, eigen, r, μ)
    end
end

function evaluate_friction!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    μ = NQCModels.fermilevel(cache.model)
    if cache.friction_method isa WideBandExact
        potential = get_potential(cache, r)
        derivative = get_derivative(cache, r)
        fill_friction_tensor!(cache.friction, sim.method.friction_method, potential, derivative, r, μ)
    elseif cache.friction_method isa Nothing
        cache.friction .= zero(cache.friction)
    else
        ∂H = Calculators.get_adiabatic_derivative(sim.calculator, r)
        eigen = Calculators.get_eigen(sim.calculator, r)
        fill_friction_tensor!(cache.friction, cache.friction_method, ∂H, eigen, r, μ)
    end
end

function correct_phase!(eig::LinearAlgebra.Eigen, old_eigenvectors::AbstractMatrix)
    @views for i in 1:length(eig.values)
        eig.vectors[:,i] .*= sign(LinearAlgebra.dot(eig.vectors[:,i], old_eigenvectors[:,i]))
    end
    return nothing
end

function evaluate_eigen!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    evaluate_potential!(cache, r)
    potential = get_potential(cache, r)

    eig = LinearAlgebra.eigen(potential)
    correct_phase!(eig, cache.eigen.vectors)
    cache.eigen.values .= eig.values
    cache.eigen.vectors .= eig.vectors
end

function evaluate_eigen!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_potential!(cache, r)
    potential = get_potential(cache, r)

    @inbounds for i in beads(cache)
        eig = LinearAlgebra.eigen(potential[i])
        correct_phase!(eig, cache.eigen[i].vectors)
        cache.eigen[i].values .= eig.values
        cache.eigen[i].vectors .= eig.vectors
    end
    return nothing
end

function evaluate_adiabatic_derivative!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    evaluate_derivative!(cache, r)
    evaluate_eigen!(cache, r)
    U = get_eigen(cache, r).vectors
    diabatic_derivative = get_derivative(cache, r)

    for I in eachindex(diabatic_derivative)
        cache.adiabatic_derivative[I] .= U' * diabatic_derivative[I] * U
    end
    return nothing
end

function evaluate_adiabatic_derivative!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_derivative!(cache, r)
    evaluate_eigen!(cache, r)
    derivative = get_derivative(cache, r)
    eigen = get_eigen(cache, r)

    for i in axes(derivative, 3) # Beads
        for j in axes(derivative, 2) # Atoms
            for k in axes(derivative, 1) # DoFs
                cache.adiabatic_derivative[k,j,i] .= eigen[i].vectors' * derivative[k,j,i] * eigen[i].vectors
            end
        end
    end
    return nothing
end

function evaluate_centroid_adiabatic_derivative!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T, 3}) where {T}
    evaluate_centroid_derivative!(cache, r)
    evaluate_centroid_eigen!(cache, r)
    centroid_derivative = get_centroid_derivative(cache, r)
    centroid_eigen = get_centroid_eigen(cache, r)

    for I in eachindex(centroid_derivative)
        cache.centroid_adiabatic_derivative[I] .= centroid_eigen.vectors' * centroid_derivative[I] * centroid_eigen.vectors
    end
    return nothing
end

function evaluate_inverse_difference_matrix!(out, eigenvalues)
    @inbounds for i in eachindex(eigenvalues)
        for j in eachindex(eigenvalues)
            out[j,i] = 1 / (eigenvalues[i] - eigenvalues[j])
        end
        out[i,i] = zero(eltype(out))
    end
    return nothing
end

"""
# References

- HammesSchifferTully_JChemPhys_101_4657_1994 Eq. (32)
- SubotnikBellonzi_AnnuRevPhyschem_67_387_2016, section 2.3
"""
function evaluate_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r::AbstractMatrix)
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
    eigen = get_eigen(cache, r)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)

    evaluate_inverse_difference_matrix!(cache.tmp_mat, eigen.values)

    @inbounds for I in NQCModels.dofs(cache)
        @. cache.nonadiabatic_coupling[I] = adiabatic_derivative[I] * cache.tmp_mat
    end
    return nothing
end

function evaluate_nonadiabatic_coupling!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
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
    evaluate_eigen!(cache, r)
    evaluate_adiabatic_derivative!(cache, r)
    eigen = get_centroid_eigen(cache, r)
    adiabatic_derivative = get_centroid_adiabatic_derivative(cache, r)

    evaluate_inverse_difference_matrix!(cache.tmp_mat, eigen.values)

    @inbounds for j in mobileatoms(cache)
        for k in dofs(cache)
            @. cache.centroid_nonadiabatic_coupling[j,k] = adiabatic_derivative[j,k] * cache.tmp_mat
        end
    end
    return nothing
end





#RingPolymer specific functions


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

function evaluate_centroid_derivative!(cache::Abstract_Cache, r::AbstractArray{T,3}) where {T}
    centroid = RingPolymerArrays.get_centroid(r)
    NQCModels.derivative!(cache.model, cache.centroid_derivative, centroid)
    return nothing
end

function evaluate_centroid_eigen!(cache::Abstract_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_centroid_potential!(cache, r)
    potential = get_centroid_potential(cache, r)

    eig = LinearAlgebra.eigen(potential)
    correct_phase!(eig, cache.centroid_eigen.vectors)
    cache.centroid_eigen.values .= eig.values
    cache.centroid_eigen.vectors .= eig.vectors
    return nothing
end

#evaluate tilde quantities

function evaluate_V̄!(cache::RingPolymer_QuantumModel_Cache, r)
    evaluate_potential!(cache, r)
    potential = get_potential(cache, r)

    for i in 1:length(cache.V̄)
        cache.V̄[i] = tr(potential[i]) / nstates(cache.model)
    end
    return nothing
end

function evaluate_V̄!(cache::RingPolymer_QuantumFrictionModel_Cache, r)
    evaluate_potential!(cache, r)
    potential = get_potential(cache, r)

    for i in 1:length(cache.V̄)
        cache.V̄[i] = tr(potential[i]) / nstates(cache.model)
    end
    return nothing
end

function evaluate_D̄!(cache::RingPolymer_QuantumModel_Cache, r)
    evaluate_derivative!(cache, r)
    derivative = get_derivative(cache, r)

    for I in eachindex(derivative)
        cache.D̄[I] = tr(derivative[I]) / nstates(cache.model)
    end
    return nothing
end

function evaluate_D̄!(cache::RingPolymer_QuantumFrictionModel_Cache, r)
    evaluate_derivative!(cache, r)
    derivative = get_derivative(cache, r)

    for I in eachindex(derivative)
        cache.D̄[I] = tr(derivative[I]) / nstates(cache.model)
    end
    return nothing
end

#evaluate traceless quantities

function evaluate_traceless_potential!(cache::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_potential!(cache, r)
    evaluate_V̄!(cache, r)
    potential = get_potential(cache, r)
    V̄ = get_V̄(cache, r)

    n = nstates(cache)
    for i in eachindex(potential)
        cache.traceless_potential[i] = potential[i] - V̄[i].*I(n)
    end
    return nothing
end

function evaluate_traceless_potential!(cache::RingPolymer_QuantumFrictionModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_potential!(cache, r)
    evaluate_V̄!(cache, r)
    potential = get_potential(cache, r)
    V̄ = get_V̄(cache, r)

    n = nstates(cache)
    for i in eachindex(potential)
        cache.traceless_potential[i] = potential[i] - V̄[i].*I(n)
    end
    return nothing
end

function evaluate_traceless_derivative!(cache::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_derivative!(cache, r)
    evaluate_D̄!(cache, r)
    derivative = get_derivative(cache, r)
    D̄ = get_D̄(cache, r)

    n = nstates(cache)
    for i in eachindex(derivative)
        cache.traceless_derivative[i] = derivative[i] - D̄[i].*I(n)
    end
    return nothing
end

function evaluate_traceless_derivative!(cache::RingPolymer_QuantumFrictionModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_derivative!(cache, r)
    evaluate_D̄!(cache, r)
    derivative = get_derivative(cache, r)
    D̄ = get_D̄(cache, r)

    n = nstates(cache)
    for i in eachindex(derivative)
        cache.traceless_derivative[i] = derivative[i] - D̄[i].*I(n)
    end
    return nothing
end

function evaluate_traceless_adiabatic_derivative!(cache::RingPolymer_QuantumModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_adiabatic_derivative!(cache, r)
    evaluate_D̄!(cache, r)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    D̄ = get_D̄(cache, r)

    n = nstates(cache)
    for i in eachindex(D̄)
        cache.traceless_adiabatic_derivative[i] = adiabatic_derivative[i] - D̄[i].*I(n)
    end
    return nothing
end


function evaluate_traceless_adiabatic_derivative!(cache::RingPolymer_QuantumFrictionModel_Cache, r::AbstractArray{T,3}) where {T}
    evaluate_adiabatic_derivative!(cache, r)
    evaluate_D̄!(cache, r)
    adiabatic_derivative = get_adiabatic_derivative(cache, r)
    D̄ = get_D̄(cache, r)

    n = nstates(cache)
    for i in eachindex(D̄)
        cache.traceless_adiabatic_derivative[i] = adiabatic_derivative[i] - D̄[i].*I(n)
    end
    return nothing
end
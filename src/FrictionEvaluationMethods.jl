abstract type FrictionEvaluationMethod end

function fill_friction_tensor!(Λ, friction_method::FrictionEvaluationMethod, ∂H, eigen, r, μ)
    for I in eachindex(r)
        for J in eachindex(r)
            Λ[J,I] = friction_method(∂H[J], ∂H[I], eigen.values, μ, friction_method.β)
        end
    end
end

function fill_friction_tensor!(Λ, friction_method::WideBandExact, potential, derivative, r, μ)
    for I in eachindex(r)
        for J in eachindex(r)
            Λ[J,I] = friction_method(potential, derivative[J], derivative[I], μ, friction_method.β)
        end
    end
end

struct GaussianBroadening{T} <: FrictionEvaluationMethod 
    σ::T
    β::T
end
function (friction_method::GaussianBroadening)(∂Hᵢ, ∂Hⱼ, eigenvalues, μ, β)
    out = zero(eltype(eigenvalues))
    for n in eachindex(eigenvalues)
        for m in eachindex(eigenvalues)
            ϵₙ = eigenvalues[n]
            ϵₘ = eigenvalues[m]
            Δϵ = ϵₙ - ϵₘ
            out += -π * ∂Hᵢ[n,m] * ∂Hⱼ[m,n] * gauss(Δϵ, friction_method) * ∂fermi(ϵₙ, μ, β)
        end
    end
    return out
end

struct OffDiagonalGaussianBroadening{T} <: FrictionEvaluationMethod
    σ::T
    β::T
end
function (friction_method::OffDiagonalGaussianBroadening)(∂Hᵢ, ∂Hⱼ, eigenvalues, μ, β)
    out = zero(eltype(eigenvalues))
    for n in eachindex(eigenvalues)
        for m=n+1:length(eigenvalues)
            ϵₙ = eigenvalues[n]
            ϵₘ = eigenvalues[m]
            Δϵ = ϵₙ - ϵₘ

            fₙ = fermi(ϵₙ, μ, β)
            fₘ = fermi(ϵₘ, μ, β)
            Δf = (fₘ - fₙ)

            out += 2π * ∂Hᵢ[n,m] * ∂Hⱼ[m,n] * gauss(Δϵ, friction_method) * Δf / Δϵ
        end
    end
    return out
end

struct DirectQuadrature{T} <: FrictionEvaluationMethod
    ρ::T    
    β::T
end
function (friction_method::DirectQuadrature)(∂Hᵢ, ∂Hⱼ, eigenvalues, μ, β)
    out = zero(eltype(eigenvalues))
    for n in eachindex(eigenvalues)
        ϵₙ = eigenvalues[n]
        out += -π * ∂Hᵢ[n,n] * ∂Hⱼ[n,n] * friction_method.ρ * ∂fermi(ϵₙ, μ, β)
    end
    return out
end

struct WideBandExact{T} <: FrictionEvaluationMethod
    ρ::T
    β::T
end
function (friction_method::WideBandExact)(potential, ∂potentialᵢ, ∂potentialⱼ, μ, β)
    h = potential[1,1]
    ∂hᵢ = ∂potentialᵢ[1,1]
    ∂hⱼ = ∂potentialⱼ[1,1]

    ρ = friction_method.ρ
    Γ = 2π * potential[2,1]^2 * ρ
    ∂Γ∂potential = sqrt(8π * ρ * Γ)
    ∂Γᵢ = ∂Γ∂potential * ∂potentialᵢ[2,1]
    ∂Γⱼ = ∂Γ∂potential * ∂potentialⱼ[2,1]

    A(ϵ) = 1/π * Γ/2 / ((ϵ-h)^2 + (Γ/2)^2)
    kernel(ϵ) = -π * (∂hᵢ + (ϵ-h)*∂Γᵢ/Γ) * (∂hⱼ + (ϵ-h)*∂Γⱼ/Γ) * A(ϵ)^2 * ∂fermi(ϵ, μ, β)
    diagonal = @view potential[diagind(potential)]
    integral, _ = QuadGK.quadgk(kernel, extrema(diagonal)...)
    return integral
end
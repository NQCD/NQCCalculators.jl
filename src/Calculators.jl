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
`evaluate_quantity!(cache, r)`
`update_quantity!(cache, r)`

The user should mostly access the get_quantity!() function, or in rare circumstances the evaluate_quantity!() fucntion.
This will ensure quantities are correctly evaluated and cached accordingly.

The latter is called by the former and is where the details required to calculate the quantity are found.
"""
const quantities = [
    (:potential, Union{AbstractArray, LinearAlgebra.Hermitian}),
    (:derivative, Union{AbstractArray, AbstractArray{LinearAlgebra.Hermitian}}),
    (:eigen, Union{FastLapackInterface.HermitianEigenWs, AbstractArray{FastLapackInterface.HermitianEigenWs}}),
    (:adiabatic_derivative, AbstractArray),
    (:nonadiabatic_coupling, AbstractArray),

    (:traceless_potential, AbstractArray{LinearAlgebra.Hermitian}),
    (:V̄, AbstractArray),
    (:traceless_derivative, AbstractArray{LinearAlgebra.Hermitian}),
    (:D̄, AbstractArray),
    (:traceless_adiabatic_derivative, AbstractArray),

    (:centroid, AbstractArray),
    (:centroid_potential, AbstractArray),
    (:centroid_derivative, AbstractArray),
    (:centroid_eigen, AbstractArray),
    (:centroid_adiabatic_derivative, AbstractArray),
    (:centroid_nonadiabatic_coupling, AbstractArray),

    (:friction, AbstractArray),
    (:centroid_friction, AbstractArray),

    (:phase_ref, AbstractVector),
    (:tmp_mat, AbstractMatrix),

]

for (quantity,T) in quantities
    get_quantity = Symbol(:get_, quantity)
    field = Expr(:call, :getfield, :cache, QuoteNode(quantity))
    

    @eval function $(get_quantity)(cache)
        return $(field)::$(T)
    end
end

include("Evaluate_Functions.jl")

include("Update_Functions.jl")
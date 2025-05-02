"""
    QCEngine

This module exists to bridge the gap between the `NQCModels` and the `NQCDynamics`.

Here, we provide functions and types for evaluating and storing quantities obtained from the
`NQCModels`. In addition any further manipulation of those quantities, such as computing eigenvalues,
is included here.

This module is largely needed to facilitate the integration of different methods for
calculating the properties of both correlated and uncorrelated electronic states with the 
dynamics methods implemented in NQCDynamics.jl that require them.
Alongside this the module integrates ring polymer and classical dynamics to allow using the same 
models and functions for both. Specific ring polymer types are provided that have the extra fields 
and methods needed to evaluate the quantities for each bead. 
"""
module QCEngine

using LinearAlgebra: LinearAlgebra, Hermitian, I, Eigen, tr
using StaticArrays: SMatrix, SVector
using RingPolymerArrays: get_centroid!

using NQCModels: NQCModels, Model, nstates, mobileatoms, dofs, Subsystem, CompositeModel
using NQCModels.AdiabaticModels: AdiabaticModel
using NQCModels.DiabaticModels: DiabaticModel, DiabaticFrictionModel
using NQCModels.FrictionModels: AdiabaticFrictionModel

using NQCDynamics: ndofs

include("Caches.jl")
include("Calculators.jl")

end
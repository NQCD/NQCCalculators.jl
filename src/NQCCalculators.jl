"""
    NQCCalculators

This module exists to bridge the gap between the `NQCModels` and the `NQCDynamics`.

Here, we provide functions and types for evaluating and storing quantities obtained from the
`NQCModels`. In addition any further manipulation of those quantities, such as computing eigenvalues,
is included here.

Alongside this the module integrates ring polymer and classical dynamics to allow using the same 
models and functions for both. Specific ring polymer types are provided that have the extra fields 
and methods needed to evaluate the quantities for each bead. 
"""
module NQCCalculators

using LinearAlgebra
using StaticArrays
using RingPolymerArrays

using NQCModels: NQCModels, Model, nstates, mobileatoms, dofs, ndofs, Subsystem, CompositeModel
using NQCModels.ClassicalModels: ClassicalModel
using NQCModels.QuantumModels: QuantumModel, QuantumFrictionModel, LargeQuantumModel
using NQCModels.FrictionModels: ClassicalFrictionModel

include("Caches.jl")
export Abstract_Cache, Abstract_ClassicalModel_Cache, Abstract_Friction_Cache
export Abstract_ExactQuantumModel_Cache, Abstract_SmallQuantumModel_Cache, Abstract_LargeQuantumModel_Cache, Abstract_CorrelatedQuantumModel_Cache

include("Calculators.jl")
export ClassicalModel_Cache, RingPolymer_ClassicalModel_Cache
export Friction_Cache, RingPolymer_Friction_Cache
export SmallQuantumModel_Cache, RingPolymer_SmallQuantumModel_Cache
export LargeQuantumModel_Cache, RingPolymer_LargeQuantumModel_Cache
export Create_Cache

end
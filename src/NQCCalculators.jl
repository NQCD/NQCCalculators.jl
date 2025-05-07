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

using NQCModels: NQCModels, Model, nstates, mobileatoms, dofs, Subsystem, CompositeModel
using NQCModels.AdiabaticModels: AdiabaticModel
using NQCModels.DiabaticModels: DiabaticModel, DiabaticFrictionModel, LargeDiabaticModel
using NQCModels.FrictionModels: AdiabaticFrictionModel

include("Caches.jl")
export Caches

include("Calculators.jl")
export Calculators

end
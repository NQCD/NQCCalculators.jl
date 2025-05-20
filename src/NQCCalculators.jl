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
using RingPolymerArrays
using NQCBase

using NQCModels: NQCModels, Model, nstates, mobileatoms, dofs, ndofs, Subsystem, CompositeModel
using NQCModels.ClassicalModels: ClassicalModel
using NQCModels.QuantumModels: QuantumModel, QuantumFrictionModel
using NQCModels.FrictionModels: ClassicalFrictionModel

include("Caches.jl")
export Abstract_Cache, Abstract_ClassicalModel_Cache
export Abstract_QuantumModel_Cache, Abstract_CorrelatedQuantumModel_Cache
export ClassicalModel_Cache, RingPolymer_ClassicalModel_Cache
export ClassicalFrictionModel_Cache, RingPolymer_ClassicalFrictionModel_Cache
export QuantumModel_Cache, RingPolymer_QuantumModel_Cache
export QuantumFrictionModel_Cache, RingPolymer_QuantumFrictionModel_Cache
export Create_Cache
export needsupdate

include("Calculators.jl")
export evaluate_friction!, evaluate_derivative!
export evaluate_eigen!, correct_phase!, correct_phase
export evaluate_adiabatic_derivative!, evaluate_centroid_adiabatic_derivative!
export evaluate_nonadiabatic_coupling!, evaluate_inverse_difference_matrix!, evaluate_nonadiabatic_coupling
export evaluate_centroid_potential!, evaluate_centroid!, evaluate_centroid_nonadiabatic_coupling!
export evaluate_traceless_adiabatic_derivative!, evaluate_traceless_potential!, evaluate_traceless_derivative!,  evaluate_D̄!, evaluate_V̄!
export evaluate_centroid_potential!, evaluate_centroid_eigen!, evaluate_centroid_derivative!

export update_electronics!, update_centroid_electronics!
end
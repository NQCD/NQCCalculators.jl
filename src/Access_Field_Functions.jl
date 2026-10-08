# ---------------------------------------------------------------------------------------------------------- #
#                Return functions for accessing fields of a struct such that it is type stable               #
# ---------------------------------------------------------------------------------------------------------- #

# Turns out accessing fields from a struct via `x.a` (`getproperty(x,:a)`) can be type unstable if the different 
# fields of the stuct have different types. This is very much the case for caches by design. 
# As such, return (`access_<fieldname>`) functions which specify the expected type of the field that is being 
# returned are essential to reduce type instability and allocations.


function access_cache(x)
    return x.cache::Abstract_Cache
end

# There should be an access function for every

function access_method(x)
    return x.method::DynamicsMethods.Method
end

function access_adiabatic_derivative(x)
    return x.adiabatic_derivative ::AbstractMatrix
end

function access_model(x)
    return x.model ::QuantumModels.QuantumModel
end

function access_potential(x)
    return x.potential ::LinearAlgebra.Hermitian
end

function access_derivative(x)
    return x.derivative ::AbstractMatrix{LinearAlgebra.Hermitian}
end

function access_eigen(x)
    return x.model ::FastLapackInterface.Hermitian
end

function access_phase_ref(x)
    return x.potential ::AbstractVector
end

function access_nonadiabatic_coupling(x)
    return x.derivative ::AbstractMatrix
end

function access_tmp_mat(x)
    return x.derivative ::AbstractMatrix
end

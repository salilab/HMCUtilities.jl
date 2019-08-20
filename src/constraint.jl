#
# Constraint interface and implementations
#

using LinearAlgebra:
    dot,
    norm2,
    normalize,
    logabsdet,
    logdet
using StatsFuns:
    logit,
    logistic

"""
    VariableConstraint{NC,NF}

Abstract type for transformations on constrained variables represented with
`NC`-dimensional vectors that produce `NF`-dimensional free (unconstrained)
vectors.

To implement a new constraint, simply create a new type of
`VariableConstraint` and implement `constrain` and `free`.

Functional defaults are provided for `free_jacobian`, `free_logpdf_correction`,
and `free_logpdf_gradient`. Any of these functions may be specialized for
increased efficiency.
"""
abstract type VariableConstraint{NC,NF} end

"""
    OneToOneConstraint{N}

Alias for dispatch on one-to-one constraints `f: ℝⁿ → ℝⁿ`.
"""
const OneToOneConstraint{N} = VariableConstraint{N,N}

"""
    UnivariateConstraint

Alias for dispatch on univariate constraints `f: ℝ → ℝ`. Univariate
constraints can take scalar inputs and produce scalar outputs.
"""
const UnivariateConstraint = OneToOneConstraint{1}


###
### Basic interface for constraints
###

"""
    constrain(c::VariableConstraint, y)

From free variable `y`, construct constrained variable.
"""
function constrain end

"""
    free(c::VariableConstraint, x)

From constrained variable `x`, construct free variable.
"""
function free end

"""
    free_dimension(c::VariableConstraint)

Get the number of dimensions (length) of the freed vector.
"""
free_dimension(::VariableConstraint{NC,NF}) where {NC,NF} = NF

"""
    constrain_dimension(c::VariableConstraint)

Get the number of dimensions (length) of the constrained vector.
"""
constrain_dimension(::VariableConstraint{NC}) where {NC} = NC

"""
    free_jacobian(c::VariableConstraint, y)

From free vector `y = f(x)`, compute the Jacobian matrix of the inverse
transformation `x = f⁻¹(y)` with entries `Jᵢⱼ = ∂xᵢ/∂yⱼ`.
"""
function free_jacobian(c::VariableConstraint, y)
    nf = free_dimension(c)
    # NOTE: work-around to make forward_jacobian type-inferrable
    # see https://github.com/FluxML/Zygote.jl/issues/299
    v = Val(min(nf, ForwardDiff.DEFAULT_CHUNK_THRESHOLD))
    # NOTE: Zygote's (reverse-mode) Jacobians are adjoints
    J′ = last(Zygote.forward_jacobian(y -> constrain(c, y), y, v))
    return adjoint(J′)
end

# NOTE: Workaround until Zygote supports nesting Jacobians
# see https://github.com/FluxML/Zygote.jl/issues/305
Zygote.@adjoint function free_jacobian(c::VariableConstraint, y::AbstractVector)
    nf = free_dimension(c)
    nc = constrain_dimension(c)

    jac(y) = ForwardDiff.jacobian(y->constrain(c, y), y)
    J = similar(y, (nc, nf))
    diffres = DiffResults.JacobianResult(J, y)
    diffres = ForwardDiff.jacobian!(diffres, jac, y)
    J = DiffResults.value(diffres)
    ∇y_J = reshape(DiffResults.jacobian(diffres), (nc, nf, nf))

    return J, function (J̄)
        @einsum ȳ[k] := J̄[i,j] * ∇y_J[i,j,k]
        return (nothing, ȳ)
    end
end

"""
    free_jacobian(c::UnivariateConstraint, y)

From free scalar `y = f(x)`, compute the derivative of the inverse
transformation `x = f⁻¹(y)`, `∂x/∂y`.
"""
function free_jacobian(c::UnivariateConstraint, y)
    ∂x_∂y = first(Zygote.gradient(y -> constrain(c, y), y))
    return ∂x_∂y
end

"""
    halflogdetmul(x::AbstractMatrix)

For a matrix `x`, compute `½log(det (x' x))`.
"""
halflogdetmul(x) = logdet(x' * x) / 2

# custom adjoint for slight speed-up
Zygote.@adjoint function halflogdetmul(x::AbstractArray)
    s = x' * x
    return logdet(s) / 2, Δ -> (Δ * (x * inv(s)),) # `Δ * x⁺ᵀ`
end

"""
    free_logpdf_correction(c::VariableConstraint, y)

From free vector `y`, compute correction to log pdf for transformation. Given
a transformation `f: x ↦ y`, its inverse `f⁻¹: y ↦ x`, and pdf `π(x)`, the log
pdf of the transformed density is

`log π(y) = log π(f⁻¹(y)) + ½log(det G)`,

where `det G` is the determinant of the matrix `G = Jᵀ J`, and `J` is the
Jacobian matrix of the inverse transformation with entries `Jᵢⱼ = ∂xᵢ/∂yⱼ`.
This result is known as the area formula.

This function returns `½log(det G)` for the general case of `f: ℝᵐ → ℝⁿ`.
"""
function free_logpdf_correction(c::VariableConstraint, y)
    J = free_jacobian(c, y)
    return halflogdetmul(J)
end

_logabsdet(x) = first(logabsdet(x))
_logabsdet(x::Real) = log(abs(x))

"""
    free_logpdf_correction(c::OneToOneConstraint, y)

From free variable `y`, compute correction to log pdf for transformation. For
a one-to-one transformation with a square Jacobian, the correction simplifies
to `log |det J|`.
"""
function free_logpdf_correction(c::OneToOneConstraint, y)
    J = free_jacobian(c, y)
    return _logabsdet(J)
end

"""
    free_logpdf_gradient(c::VariableConstraint, y, logπx, ∇x_logπx)

Compute the log density and its gradient on the free variable from the
corresponding log density and its gradient on the constrained variable.

From free vector `y = f(x)`, log pdf `log π(x)`, and gradient `∇x log π(x)` of
`log π(x)` with respect to `x`, compute corrected log pdf `log π(y)` and its
gradient `∇y log π(y)` wrt `y`.
"""
function free_logpdf_gradient(c::VariableConstraint, y, logπx, ∇x_logπx)
    TL = typeof(zero(eltype(logπx)) + zero(eltype(y)))
    TG = typeof(zero(eltype(∇x_logπx)) * zero(eltype(y)))

    x, back_constrain = Zygote.forward(y -> constrain(c, y), y)
    ∇y_logπx = first(back_constrain(∇x_logπx))

    # NOTE: currently, this only works for multivariate constraints if
    # `free_logpdf_correction` or `free_jacobian` are implemented explicitly.
    logdetJ, back_logdetJ = Zygote.forward(y -> free_logpdf_correction(c, y), y)
    s = Zygote.sensitivity(logdetJ)  # 1
    ∇y_logdetJ = first(back_logdetJ(s))

    logπy = logπx + logdetJ
    ∇y_logπy = Zygote.accum(∇y_logπx, ∇y_logdetJ)

    return logπy, ∇y_logπy
end


###
### Constraint implementations
###

"""
    IdentityConstraint{N} <: OneToOneConstraint{N}

Do-nothing constraint on `ℝⁿ`, corresponding to the identity function on
`n`-dimensional variables. Included for convenient bundling of constrained
with unconstrained variables.

# Constructor

    IdentityConstraint(n::Int)
"""
struct IdentityConstraint{N} <: OneToOneConstraint{N} end

IdentityConstraint(n) = IdentityConstraint{n}()

constrain(::IdentityConstraint, y) = copy(y)

free(::IdentityConstraint, x) = copy(x)

free_logpdf_correction(::IdentityConstraint, y) = zero(eltype(y))


"""
    LowerBoundedConstraint{T} <: UnivariateConstraint

Constraint on a scalar that is strictly greater than a lower bound.

# Constructor

    LowerBoundedConstraint(lb)
"""
struct LowerBoundedConstraint{T} <: UnivariateConstraint
    lb::T
end

constrain(c::LowerBoundedConstraint, y) = exp(y) + c.lb

free(c::LowerBoundedConstraint, x) = log(x - c.lb)

free_logpdf_correction(::LowerBoundedConstraint{T}, y) where {T} = y + zero(T)


"""
    UpperBoundedConstraint{T} <: UnivariateConstraint

Constraint on a scalar that is strictly less than an upper bound.

# Constructor

    UpperBoundedConstraint(ub)
"""
struct UpperBoundedConstraint{T} <: UnivariateConstraint
    ub::T
end

free(c::UpperBoundedConstraint, x) = log(c.ub - x)

constrain(c::UpperBoundedConstraint, y) = c.ub - exp(y)

free_logpdf_correction(::UpperBoundedConstraint{T}, y) where {T} = y + zero(T)


"""
    BoundedConstraint{TL,TU,TD} <: UnivariateConstraint

Constraint on a scalar that is has both an upper and lower bound.

# Constructor

    BoundedConstraint(lb, ub)
"""
struct BoundedConstraint{TL,TU,TD} <: UnivariateConstraint
    lb::TL
    ub::TU
    delta::TD
end

function BoundedConstraint(lb::Real, ub::Real)
    @assert lb < ub
    return BoundedConstraint(lb, ub, ub - lb)
end

free(c::BoundedConstraint, x) = logit((x - c.lb) / c.delta)

constrain(c::BoundedConstraint, y) = c.delta * logistic(y) + c.lb


"""
    TransformConstraint(lb::Real, ub::Real)

Convenient constructor for lower-, upper-, lower- and upper-, and un-bounded
univariate constraints. The correct type is chosen based on the arguments.
"""
function TransformConstraint(lb=-Inf, ub=Inf)
    @assert lb < ub
    has_lb, has_ub = isfinite(lb), isfinite(ub)
    if has_lb && has_ub
        return BoundedConstraint(lb, ub)
    elseif has_lb
        return LowerBoundedConstraint(lb)
    elseif has_ub
        return UpperBoundedConstraint(ub)
    else
        return IdentityConstraint(1)
    end
end


"""
    UnitVectorConstraint{N} <: OneToOneConstraint{N}

Transformation from an `n`-dimensional unit-vector to an unconstrained
`n`-dimensional vector. Note that in this case the inverse transformation
(ℓ²-normalization) is not unique, and therefore the usual log correction to
the density is undefined.

However, using the fact that a standard multivariate normally distributed
vector when normalized is uniformly distributed on a sphere, we can correct
the log pdf wrt `y` by applying a standard multivariate normal prior to `y`.
The corresponding log density correction is `-½ yᵀ y = -½ |y|²`.

Gradients wrt `x` are pulled back to `y` by Jacobian multiplication. The
Jacobian in this case is a normalized projection matrix onto the tangent space
to the sphere at `x`: `J = Πₓ / |y|`, where `Πₓ = I - xᵀ x`.

# Constructor

    UnitVectorConstraint(n::Int)
"""
struct UnitVectorConstraint{N} <: OneToOneConstraint{N} end

UnitVectorConstraint(n) = UnitVectorConstraint{n}()

"""
    normalize2(v)

Compute the ℓ²-norm (length) of `v`.
"""
normalize2(v) = normalize(v)

# Workaround for Zygote not currently supporting `rmul!`.
Zygote.@adjoint function normalize2(v)
    n = norm2(v)
    if !isempty(v)
        vv = v ./ n
        return vv, Δ -> ((Δ .- vv * (vv' * Δ)) ./ n,)  # normalized projection
    else
        T = typeof(zero(eltype(v)) / n)
        return T[], Δ -> (nothing,)
    end
end

"""
    half_norm2_sqr(v)

For vector `v`, compute the half of its square ℓ²-norm, `½ |v|²`.
"""
half_norm2_sqr(v) = dot(v, v) / 2

Zygote.@adjoint half_norm2_sqr(v) = half_norm2_sqr(v), Δ -> (Δ .* v,)

constrain(::UnitVectorConstraint, y) = normalize2(y)

function free(::UnitVectorConstraint, x)
    snx = dot(x, x)
    @assert snx ≈ 1
    return x
end

free_logpdf_correction(::UnitVectorConstraint, y) = -half_norm2_sqr(y)

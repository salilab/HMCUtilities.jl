#
# Constraint interface and implementations
#

using LinearAlgebra:
    dot,
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

The supported interface is the 3 functions `constrain`, `free`, and
`constrain_with_pushlogpdf`.

To implement a new constraint, simply create a new type of `VariableConstraint`
and implement `constrain` and `free`. Various internal functions are used to
provide efficient and accurate defaults; these may be overriden for increased
efficiency when analytical gradients/jacobian determinants are known.
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
    constrain_with_pushlogpdf(c::VariableConstraint, y)

From free variable `y = f(x)` get the constrained variable `x` and a new
function that pushes forward the log density `log π(x)` with respect to `x` and
its gradient to the corresponding log density `log π(y)` with respect to `y` and
its gradient.

```julia
x, pushlogpdf = constrain_with_pushlogpdf(c, y)
logπx, ∇x_logπx = ... # Compute density and its gradient in constrained space
logπy, ∇y_logπy = pushlogpdf(logπx, ∇x_logπx)
```
"""
function constrain_with_pushlogpdf(c::VariableConstraint, y)
    x, pushgrad = constrain_with_pushgrad(c, y)
    logdetJ, ∇y_logdetJ = free_logpdf_correction_with_grad(c, y)

    return x, function (logπx, ∇x_logπx)
        ∇y_logπx = pushgrad(∇x_logπx)
        logπy = logπx + logdetJ
        ∇y_logπy = ∇y_logπx .+ ∇y_logdetJ
        return logπy, ∇y_logπy
    end
end

function constrain_with_pushlogpdf(c::UnivariateConstraint, y::AbstractArray)
    x, scalar_push = constrain_with_pushlogpdf(c, y[1])

    return [x], function (logπx, ∇x_logπx::AbstractArray)
        logπy, ∇y_logπy = scalar_push(logπx, ∇x_logπx[1])
        return logπy, [∇y_logπy]
    end
end

"""
    constrain_jacobian(c::VariableConstraint, y)

From free vector `y = f(x)`, compute the Jacobian matrix of the inverse
transformation `x = f⁻¹(y)` with entries `Jᵢⱼ = ∂xᵢ/∂yⱼ`.
"""
function constrain_jacobian(c::VariableConstraint, y)
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
Zygote.@adjoint function constrain_jacobian(c::VariableConstraint, y::AbstractVector)
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
    constrain_jacobian(c::UnivariateConstraint, y)

From free scalar `y = f(x)`, compute the derivative of the inverse
transformation `x = f⁻¹(y)`, `∂x/∂y`.
"""
function constrain_jacobian(c::UnivariateConstraint, y)
    ∂x_∂y = first(Zygote.gradient(y -> constrain(c, y), y))
    return ∂x_∂y
end

function constrain_jacobian(c::UnivariateConstraint, y::AbstractArray)
    return [constrain_jacobian(c, y[1])]
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
    @assert free_dimension(c) <= constrain_dimension(c)
    J = constrain_jacobian(c, y)
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
    J = constrain_jacobian(c, y)
    return _logabsdet(J)
end

_format_grad(x, ∇x) = Zygote.accum(zero(x), ∇x)

function free_logpdf_correction_with_grad(c, y)
    logdetJ, back = Zygote.forward(y) do (y)
        return free_logpdf_correction(c, y)
    end

    s = Zygote.sensitivity(logdetJ)  # 1
    ∇y_logdetJ = _format_grad(y, first(back(s)))

    return logdetJ, ∇y_logdetJ
end

function constrain_with_pushgrad(c::VariableConstraint, y)
    x, pushgrad = Zygote.forward(y -> constrain(c, y), y)
    return x, ∇x -> _format_grad(y, first(pushgrad(∇x)))
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

constrain(::IdentityConstraint, y) = y

free(::IdentityConstraint, x) = x

function constrain_with_pushlogpdf(::IdentityConstraint, y)
    return y, (logπx, ∇x_logπx) -> (logπx, ∇x_logπx)
end

function constrain_with_pushlogpdf(::IdentityConstraint{1}, y::AbstractArray)
    return y, (logπx, ∇x_logπx) -> (logπx, ∇x_logπx)
end

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

function constrain_with_pushlogpdf(c::LowerBoundedConstraint, y)
    expy = exp(y)
    x = expy + c.lb
    return x, function (logπx, dx_logπx)
        return logπx + y, dx_logπx * expy + 1
    end
end


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

function constrain_with_pushlogpdf(c::UpperBoundedConstraint, y)
    nexpy = -exp(y)
    x = nexpy + c.ub
    return x, function (logπx, dx_logπx)
        return logπx + y, dx_logπx * nexpy + 1
    end
end


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

# Avoid recalculating logistic
function constrain_with_pushlogpdf(c::BoundedConstraint, y)
    z = logistic(y)
    delz = c.delta * z
    x = delz + c.lb
    dy_dx = delz * (1 - z)
    logdetJ = log(dy_dx)
    dy_logdetJ = 1 - 2z
    return x, function (logπx, dx_logπx)
        return logπx + logdetJ, dx_logπx * dy_dx + dy_logdetJ
    end
end


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
(ℓ²-normalization) is not unique, and therefore the pushforward density cannot
be obtained using the usual Jacobian technique.

However, using the fact that a standard multivariate normally distributed
vector when normalized is uniformly distributed on a sphere, we can push
forward the uniform measure on the sphere by applying a standard multivariate
normal prior to `y`. The corresponding log density correction is `-½ yᵀ y = -½
|y|²`.

The Jacobian of the inverse transformation is a normalized projection matrix
onto the tangent space to the sphere at `x`: `J = Πₓ / |y|`, where
`Πₓ = I - xᵀ x`.

# Constructor

    UnitVectorConstraint(n::Int)
"""
struct UnitVectorConstraint{N} <: OneToOneConstraint{N} end

UnitVectorConstraint(n) = UnitVectorConstraint{n}()

constrain(::UnitVectorConstraint, y) = normalize(y)

function free(::UnitVectorConstraint, x)
    snx = dot(x, x)
    @assert snx ≈ 1
    return x
end

# Avoid re-normalizing
function constrain_with_pushlogpdf(::UnitVectorConstraint, y)
    norm_sqr = dot(y, y)
    ny = sqrt(norm_sqr)
    x = y ./ ny
    logdetJ = -norm_sqr / 2
    pushgrad = Δ -> Δ .- x .* (dot(x, Δ) + 1)
    return x, function (logπx, ∇x_logπx)
        return logπx + logdetJ, pushgrad(∇x_logπx ./ ny)
    end
end

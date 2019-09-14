#
# Constraint interface and implementations
#

import Base: clamp,
    show
using LinearAlgebra:
    dot,
    norm,
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

The supported interface is the 4 functions `constrain`, `free`,
`constrain_with_pushlogpdf` and `constrain_with_pushlogpdf_grad`.

To implement a new constraint, simply create a new type of `VariableConstraint`
and implement `constrain` and `free`. Various internal functions are used to
provide efficient and accurate defaults; these may be overriden for increased
efficiency when analytical gradients/jacobian determinants are known. The most
common override is `constrain_with_logpdf_correction`.
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
    clamp(c::VariableConstraint, x)

Return `x`, ensuring that its value satisfies the constraint. This is useful
to avoid numerical instability when a value nears the boundary condition.
The clamp is invisible during differentiation.
"""
function clamp end

# Passthrough adjoints for clamp
Zygote.@adjoint clamp(c::VariableConstraint, x) = clamp(c, x), Δ -> (nothing, Δ)

"""
    free(c::VariableConstraint, x)

From constrained variable `x`, construct free variable.
"""
function free end

Base.@propagate_inbounds function free(c::UnivariateConstraint,
                                       x::AbstractVector)
    return [free(c, x[1])]
end

"""
    constrain(c::VariableConstraint, y)

From free variable `y`, construct constrained variable.
"""
function constrain end

Base.@propagate_inbounds function constrain(c::UnivariateConstraint,
                                            y::AbstractVector)
    return [constrain(c, y[1])]
end

"""
    constrain_with_logpdf_correction(c::VariableConstraint, y)

From free variable `y = f(x)` get the constrained variable `x` and the
addtive correction to the log density `log π(x)` to get `log π(x)`. See
[`free_logpdf_correction`](@free_logpdf_correction).
"""
function constrain_with_logpdf_correction(c, y)
    x = constrain(c, y)
    logdetJ = free_logpdf_correction(c, y)
    return x, logdetJ
end

Base.@propagate_inbounds function constrain_with_logpdf_correction(
        c::UnivariateConstraint,
        y::AbstractVector
    )
    x, logdetJ = constrain_with_logpdf_correction(c, y[1])
    return [x], logdetJ
end

"""
    constrain_with_pushlogpdf(c::VariableConstraint, y)

From free variable `y = f(x)` get the constrained variable `x` and a new
function that pushes forward the log density `log π(x)` with respect to `x` and
to the corresponding log density `log π(y)` with respect to `y`.

```julia
x, pushlogpdf = constrain_with_pushlogpdf(c, y)
logπx = ... # Compute density in constrained space
logπy = pushlogpdf(logπx)
```
"""
function constrain_with_pushlogpdf(c, y)
    x, logdetJ = constrain_with_logpdf_correction(c, y)
    return x, logπx -> logπx + logdetJ
end

Base.@propagate_inbounds function constrain_with_pushlogpdf(
        c::UnivariateConstraint,
        y::AbstractVector
    )
    x, pushlogpdf = constrain_with_pushlogpdf(c, y[1])
    return [x], pushlogpdf
end

"""
    constrain_with_pushlogpdf_grad(c::VariableConstraint, y)

From free variable `y = f(x)` get the constrained variable `x` and a new
function that pushes forward the log density `log π(x)` with respect to `x` and
its gradient to the corresponding log density `log π(y)` with respect to `y` and
its gradient.

```julia
x, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, y)
logπx, ∇x_logπx = ... # Compute density and its gradient in constrained space
logπy, ∇y_logπy = pushlogpdf_grad(logπx, ∇x_logπx)
```
"""
function constrain_with_pushlogpdf_grad(c, y)
    (x, logdetJ), back = Zygote.forward(constrain_with_logpdf_correction, c, y)
    nf = free_dimension(c)

    return x, function (logπx, ∇x_logπx)
        s = Zygote.sensitivity(logdetJ)
        logπy = logπx + logdetJ
        T = eltype(logπy)
        ∇y_logπy = similar(∇x_logπx, T, nf)
        copyto!(∇y_logπy, back((∇x_logπx, s))[2])
        return logπy, ∇y_logπy
    end
end

function constrain_with_pushlogpdf_grad(c::UnivariateConstraint, y::Real)
    (x, logdetJ), back = Zygote.forward(constrain_with_logpdf_correction, c, y)

    return x, function (logπx, ∇x_logπx::Real)
        s = Zygote.sensitivity(logdetJ)
        logπy = logπx + logdetJ
        ∇y_logπy = back((∇x_logπx, s))[2]
        return logπy, ∇y_logπy
    end
end

Base.@propagate_inbounds function constrain_with_pushlogpdf_grad(
        c::UnivariateConstraint,
        y::AbstractVector
    )
    x, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, y[1])
    return [x], function (logπx, ∇x_logπx::AbstractVector)
        logπy, ∇y_logπy = pushlogpdf_grad(logπx, ∇x_logπx[1])
        return logπy, [∇y_logπy]
    end
end

"""
    constrain_jacobian(c::VariableConstraint, y)

From free vector `y = f(x)`, compute the Jacobian matrix of the inverse
transformation `x = f⁻¹(y)` with entries `Jᵢⱼ = ∂xᵢ/∂yⱼ`.
"""
function constrain_jacobian(c, y)
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
Zygote.@adjoint function constrain_jacobian(c,
                                            y::AbstractVector)
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
transformation `x = f⁻¹(y)`, `dx/dy`.
"""
function constrain_jacobian(c::UnivariateConstraint, y)
    dx_dy = Zygote.gradient(constrain, c, y)[2]
    return dx_dy
end

Base.@propagate_inbounds function constrain_jacobian(
        c::UnivariateConstraint,
        y::AbstractArray
    )
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
function free_logpdf_correction(c, y)
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
function free_logpdf_correction(c::OneToOneConstraint, y::AbstractVector)
    J = constrain_jacobian(c, y)
    return first(logabsdet(J))
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

function Base.show(io::IO, mime::MIME"text/plain",
                    c::IdentityConstraint{N}) where {N}
    print(io, "IdentityConstraint($N)")
end

clamp(::IdentityConstraint, x) = x
clamp(::IdentityConstraint, x::ForwardDiff.Dual) = x

free(::IdentityConstraint{1}, x::Real) = x
free(::IdentityConstraint, x::AbstractVector) = x

constrain(::IdentityConstraint{1}, y::Real) = y
constrain(::IdentityConstraint, y::AbstractVector) = y

free_logpdf_correction(::IdentityConstraint{1}, y::Real) = zero(eltype(y))
free_logpdf_correction(::IdentityConstraint, y::AbstractVector) = zero(eltype(y))

constrain_with_pushlogpdf(::IdentityConstraint{1}, y::Real) = y, identity
constrain_with_pushlogpdf(::IdentityConstraint, y::AbstractVector) = y, identity

function constrain_with_pushlogpdf_grad(::IdentityConstraint{1}, y::Real)
    return y, (logπx, ∇x_logπx) -> (logπx, ∇x_logπx)
end

function constrain_with_pushlogpdf_grad(::IdentityConstraint, y::AbstractVector)
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

function Base.show(io::IO, mime::MIME"text/plain", c::LowerBoundedConstraint)
    print(io, "LowerBoundedConstraint($(c.lb))")
end

clamp(c::LowerBoundedConstraint, x::Real) = max(x, c.lb + eps(x))
clamp(::LowerBoundedConstraint, x::ForwardDiff.Dual) = x

free(c::LowerBoundedConstraint, x::Real) = log(clamp(c, x) - c.lb)

constrain(c::LowerBoundedConstraint, y::Real) = clamp(c, exp(y) + c.lb)

free_logpdf_correction(c::LowerBoundedConstraint, y::Real) = y


"""
    UpperBoundedConstraint{T} <: UnivariateConstraint

Constraint on a scalar that is strictly less than an upper bound.

# Constructor

    UpperBoundedConstraint(ub)
"""
struct UpperBoundedConstraint{T} <: UnivariateConstraint
    ub::T
end

function Base.show(io::IO, mime::MIME"text/plain", c::UpperBoundedConstraint)
    print(io, "UpperBoundedConstraint($(c.ub))")
end

clamp(c::UpperBoundedConstraint, x::Real) = min(x, c.ub - eps(x))
clamp(::UpperBoundedConstraint, x::ForwardDiff.Dual) = x

free(c::UpperBoundedConstraint, x::Real) = log(c.ub - clamp(c, x))

constrain(c::UpperBoundedConstraint, y::Real) = clamp(c, c.ub - exp(y))

free_logpdf_correction(c::UpperBoundedConstraint, y::Real) = y


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

function Base.show(io::IO, mime::MIME"text/plain", c::BoundedConstraint)
    print(io, "BoundedConstraint($(c.lb), $(c.ub))")
end

function BoundedConstraint(lb::Real, ub::Real)
    @assert lb < ub
    return BoundedConstraint(lb, ub, ub - lb)
end

clamp(c::BoundedConstraint, x::Real) = clamp(x, c.lb + eps(x), c.ub - eps(x))
clamp(::BoundedConstraint, x::ForwardDiff.Dual) = x

free(c::BoundedConstraint, x::Real) = logit((clamp(c, x) - c.lb) / c.delta)

constrain(c::BoundedConstraint, y::Real) = clamp(c, c.delta * logistic(y) + c.lb)

function constrain_with_logpdf_correction(c::BoundedConstraint, y::Real)
    z = logistic(y)
    delz = c.delta * z
    x = delz + c.lb
    dx_dy = delz * (1 - z)
    return clamp(c, x), log(dx_dy)
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

function Base.show(io::IO, mime::MIME"text/plain",
                   c::UnitVectorConstraint{N}) where {N}
    print(io, "UnitVectorConstraint($N)")
end

clamp(::UnitVectorConstraint, x) = normalize(x)
clamp(::UnitVectorConstraint, x::AbstractArray{<:ForwardDiff.Dual}) = x

free(c::UnitVectorConstraint, x) = clamp(c, x)

function constrain(::UnitVectorConstraint, y)
    x = y ./ norm(y)
    return x
end

function normalize_with_norm(y)
    ny = norm(y)
    return y ./ ny, ny
end

Zygote.@adjoint function normalize_with_norm(y)
    x, ny = normalize_with_norm(y)
    return (x, ny), function (Δ)
        x̄, n̄ȳ = Δ
        return (x̄ ./ ny .- x .* (dot(x, x̄) / ny - n̄ȳ),)
    end
end

function constrain_with_logpdf_correction(::UnitVectorConstraint, y)
    x, ny = normalize_with_norm(y)
    logdetJ = -ny^2 / 2
    return x, logdetJ
end


"""
    UnitSimplexConstraint{N,M} <: VariableConstraint{N,M}

Transformation from an `n`-dimensional vector of positive reals with a unit
ℓ¹-norm to an unconstrained `m = n - 1`-dimensional vector.

This constraint uses the stick-breaking process to define the transformation.
See https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process
for more details.

# Constructor

    UnitSimplexConstraint(n::Int)
"""
struct UnitSimplexConstraint{N,M} <: VariableConstraint{N,M} end

UnitSimplexConstraint(n::Int) = UnitSimplexConstraint{n,n-1}()

function Base.show(io::IO, mime::MIME"text/plain",
                   c::UnitSimplexConstraint{N}) where {N}
    print(io, "UnitSimplexConstraint($N)")
end

"""
    stick_ratio(x, Σx)

Break a piece of length `x` off of a unit-length stick off of which pieces of
total length `Σx` have already been broken. Return the ratio of the length `x`
to the remaining length of the stick.
"""
stick_ratio(x, Σx) = x / (1 - Σx)

"""
    stick_length(r, Σx)

Inverse of `stick_ratio`.
"""
stick_length(r, Σx) = r * (1 - Σx)

"""
    constrain_stick_ratio(k, y, K)

Constrain the free variable `y` to the `k`th of `K` stick ratios.
"""
constrain_stick_ratio(k, y, K) = logistic(y - log(K - k))

"""
    free_stick_ratio(k, r, K)

Free the `k`th stick ratio `r` of `K` stick ratios to an unconstrained variable.
"""
free_stick_ratio(k, r, K) = logit(r) + log(K - k)

function clamp(::UnitSimplexConstraint, x)
    ϵ = eps(eltype(x))
    return normalize(clamp.(x, ϵ, 1 - ϵ), 1)
end

clamp(::UnitSimplexConstraint, x::AbstractArray{<:ForwardDiff.Dual}) = x

function free(c::UnitSimplexConstraint, x)
    K = constrain_dimension(c)
    @assert length(x) == K

    x = clamp(c, x)
    T = eltype(x)
    y = similar(x, K - 1)
    Σx = zero(T)
    @inbounds begin
        y[1] = free_stick_ratio(1, x[1], K)
        @simd for k = 2:(K - 1)
            Σx += x[k - 1]
            zₖ = stick_ratio(x[k], Σx)
            y[k] = free_stick_ratio(k, zₖ, K)
        end
    end
    return y
end

function constrain(c::UnitSimplexConstraint, y)
    K = constrain_dimension(c)
    @assert length(y) == free_dimension(c)

    x = Zygote.Buffer(y, K)
    Σx = zero(eltype(y))
    @inbounds begin
        for k = 1:(K - 1)
            zₖ = constrain_stick_ratio(k, y[k], K)
            xₖ = stick_length(zₖ, Σx)
            x[k] = xₖ
            Σx += xₖ
        end
        x[K] = 1 - Σx
    end
    return clamp(c, copy(x))
end

function constrain_with_logpdf_correction(c::UnitSimplexConstraint, y)
    K = constrain_dimension(c)
    @assert length(y) == free_dimension(c)
    T = eltype(y)

    x = Zygote.Buffer(y, K)
    Σx = zero(eltype(y))
    logdetJ = log(K) / 2
    @inbounds begin
        for k = 1:(K - 1)
            zₖ = constrain_stick_ratio(k, y[k], K)
            xₖ = stick_length(zₖ, Σx)
            x[k] = xₖ
            logdetJ += log(xₖ * (1 - zₖ))
            Σx += xₖ
        end
        x[K] = 1 - Σx
    end

    return clamp(c, copy(x)), logdetJ
end

function constrain_with_pushlogpdf_grad(
        c::UnitSimplexConstraint,
        y::SubArray
    )
    return constrain_with_pushlogpdf_grad(c, collect(y))
end

"""
    JointConstraint{TC,RC,RF,NC,NF} <: VariableConstraint{NC,NF}

Joint transformation on a series of constraints. This constraint type
conveniently binds together a series of transformations on non-overlapping
subarrays of a longer parameter array.

# Constructor

    JointConstraint(constraints::VariableConstraint...)
"""
struct JointConstraint{TC,CR,CF,NC,NF} <: VariableConstraint{NC,NF}
    constraints::TC
    cranges::CR
    franges::CF

    function JointConstraint(constraints...)
        cs = merge_constraints(constraints...)

        ncs = map(constrain_dimension, cs)
        nc = sum(ncs)
        cranges = _ranges_from_lengths(ncs)

        nfs = map(free_dimension, cs)
        nf = sum(nfs)
        franges = _ranges_from_lengths(nfs)

        return new{typeof(cs),typeof(cranges),typeof(franges),nc,nf}(cs, cranges, franges)
    end
end

JointConstraint(c) = c

"""
    merge_constraints(cs::VariableConstraint...)

Merge adjacent constraints that can be merged. For example,
`IdentityConstraint(3)` and `IdentityConstraint(2)` can be merged into a single
`IdentityConstraint(5)`.
"""
merge_constraints(c1, c2, cs...) = reduce(merge_constraints, (c1, c2, cs...))

@inline merge_constraints(c1) = (c1,)

@inline merge_constraints(c1, c2) = (c1, c2)

@inline function merge_constraints(c1, c2::Tuple)
    return (merge_constraints(c1, c2[1])..., Base.tail(c2)...)
end

@inline function merge_constraints(c1::Tuple, c2)
    return (Base.front(c1)..., merge_constraints(c1[end], c2)...)
end

@inline function merge_constraints(c1::Tuple, c2::Tuple)
    return (Base.front(c1)...,
            merge_constraints(c1[end], c2[1])...,
            Base.tail(c2)...)
end

@inline function merge_constraints(::IdentityConstraint{M},
                                   ::IdentityConstraint{N}) where {M,N}
    return (IdentityConstraint(M + N),)
end

function _ranges_from_lengths(lengths)
    ranges = UnitRange{Int}[]
    k = 1
    for len in lengths
        push!(ranges, k:(k + len - 1))
        k += len
    end
    return tuple(ranges...)
end

function Base.show(io::IO, mime::MIME"text/plain", jc::JointConstraint)
    print(io, "JointConstraint(")
    nconstraints = length(jc.constraints)
    if nconstraints > 0
        print(io, "$(repr(mime, jc.constraints[1]))")
        if nconstraints > 10
            print(io, ", $(repr(mime, jc.constraints[2]))")
            print(io, ", ...")
            print(io, ", $(repr(mime, jc.constraints[end-1]))")
            print(io, ", $(repr(mime, jc.constraints[end]))")
        else
            for i in 2:nconstraints
                print(io, ", $(repr(mime, jc.constraints[i]))")
            end
        end
    end
    print(io, ")")
end

function _parallel_free(cs, cranges, franges, nf, x)
    y = similar(x, nf)
    @simd for i in 1:length(cs)
        @inbounds begin
            xᵢ = view(x, cranges[i])
            setindex!(y, free(cs[i], xᵢ), franges[i])
        end
    end
    return y
end

function free(jc::JointConstraint, x)
    @assert length(x) == constrain_dimension(jc)
    return _parallel_free(
        jc.constraints,
        jc.cranges,
        jc.franges,
        free_dimension(jc),
        x
    )
end

function _parallel_constrain(cs, cranges, franges, nc, y)
    x = similar(y, nc)
    @simd for i in 1:length(cs)
        @inbounds begin
            yᵢ = view(y, franges[i])
            setindex!(x, constrain(cs[i], yᵢ), cranges[i])
        end
    end
    return x
end

function constrain(jc::JointConstraint, y)
    @assert length(y) == free_dimension(jc)
    return _parallel_constrain(
        jc.constraints,
        jc.cranges,
        jc.franges,
        constrain_dimension(jc),
        y
    )
end

function _parallel_constrain_with_logpdf_correction(cs, cranges, franges, nc, nf, y)
    x = similar(y, nc)

    TL = eltype(y)
    cinds = 1:length(cs)

    logdetJ = zero(TL)
    @simd for i in cinds
        @inbounds begin
            yᵢ = view(y, franges[i])
            xᵢ, logdetJᵢ = constrain_with_logpdf_correction(cs[i], yᵢ)
            setindex!(x, xᵢ, cranges[i])
        end
        logdetJ += logdetJᵢ
    end

    return x, logdetJ
end

Zygote.@adjoint function _parallel_constrain_with_logpdf_correction(cs, cranges, franges, nc, nf, y)
    x = similar(y, nc)

    TL = eltype(y)
    cinds = 1:length(cs)

    backs = []
    logdetJ = zero(TL)
    @simd for i in cinds
        @inbounds begin
            yᵢ = view(y, franges[i])
            (xᵢ, logdetJᵢ), backᵢ = (
                Zygote.forward(constrain_with_logpdf_correction, cs[i], yᵢ)
            )
            setindex!(x, xᵢ, cranges[i])
        end
        push!(backs, backᵢ)
        logdetJ += logdetJᵢ
    end

    return (x, logdetJ), function (Δ)
        x̄, logdetJ̄ = Δ
        TA = Base.promote_eltypeof(x̄, logdetJ̄)
        ȳ = similar(x̄, TA, nf)

        @simd for i in cinds
            @inbounds begin
                x̄ᵢ = view(x̄, cranges[i])
                backᵢ = backs[i]
                ȳᵢ = backᵢ((x̄ᵢ, logdetJ̄))[2]
                setindex!(ȳ, ȳᵢ, franges[i])
            end
        end

        return (nothing, nothing, nothing, nothing, nothing, ȳ)
    end
end

function constrain_with_logpdf_correction(jc::JointConstraint, y)
    nf = free_dimension(jc)
    @assert length(y) == nf
    return _parallel_constrain_with_logpdf_correction(
        jc.constraints,
        jc.cranges,
        jc.franges,
        constrain_dimension(jc),
        nf,
        y
    )
end

function _parallel_constrain_with_pushlogpdf(cs, cranges, franges, nc, nf, y)
    x = similar(y, nc)
    pushes = []
    cinds = 1:length(cs)
    logdetJs = []
    @simd for i in cinds
        @inbounds begin
            yᵢ = view(y, franges[i])
            xᵢ, pushᵢ = constrain_with_pushlogpdf(cs[i], yᵢ)::Tuple
            setindex!(x, xᵢ, cranges[i])
        end
        push!(pushes, pushᵢ)
    end
    y1 = @inbounds y[1]

    function pushlogpdf(logπx)
        TL = Base.promote_eltypeof(y1, logπx)

        logdetJ = zero(TL)
        @simd for i in cinds
            @inbounds pushᵢ = pushes[i]
            logdetJᵢ = pushᵢ(zero(TL))
            logdetJ += logdetJᵢ
        end

        logπy::TL = logπx + logdetJ
        return logπy
    end

    return x, pushlogpdf
end

function constrain_with_pushlogpdf(jc::JointConstraint, y)
    @assert length(y) == free_dimension(jc)
    return _parallel_constrain_with_pushlogpdf(
        jc.constraints,
        jc.cranges,
        jc.franges,
        constrain_dimension(jc),
        free_dimension(jc),
        y
    )
end

function _parallel_constrain_with_pushlogpdf_grad(cs, cranges, franges, nc, nf, y)
    x = similar(y, nc)
    pushes = []
    cinds = 1:length(cs)
    @simd for i in cinds
        @inbounds begin
            yᵢ = view(y, franges[i])
            xᵢ, pushᵢ = constrain_with_pushlogpdf_grad(cs[i], yᵢ)::Tuple
            setindex!(x, xᵢ, cranges[i])
        end
        push!(pushes, pushᵢ)
    end
    y1 = @inbounds y[1]

    function pushlogpdf_grad(logπx, ∇x_logπx)
        TL = Base.promote_eltypeof(y1, logπx, ∇x_logπx)
        ∇y_logπy = similar(∇x_logπx, nf)

        logdetJ = zero(TL)
        @simd for i in cinds
            @inbounds begin
                ∇x_logπxᵢ = view(∇x_logπx, cranges[i])
                pushᵢ = pushes[i]
                logdetJᵢ, ∇y_logπyᵢ = pushᵢ(zero(logπx), ∇x_logπxᵢ)::Tuple
                setindex!(∇y_logπy, ∇y_logπyᵢ, franges[i])
            end
            logdetJ += logdetJᵢ
        end

        logπy::TL = logπx + logdetJ
        return logπy, ∇y_logπy
    end

    return x, pushlogpdf_grad
end

function constrain_with_pushlogpdf_grad(jc::JointConstraint, y)
    @assert length(y) == free_dimension(jc)
    return _parallel_constrain_with_pushlogpdf_grad(
        jc.constraints,
        jc.cranges,
        jc.franges,
        constrain_dimension(jc),
        free_dimension(jc),
        y
    )
end

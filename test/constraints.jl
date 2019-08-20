import HMCUtilities:
    VariableConstraint,
    constrain,
    free

struct TestAffineConstraint{N,TM,TMinv,Tb} <: HMCUtilities.VariableConstraint{N,N}
    M::TM
    Minv::TMinv
    b::Tb

    function TestAffineConstraint(M, b)
        n = length(b)
        @assert size(M) == (n, n)
        Minv = inv(M)
        return new{n,typeof(M),typeof(Minv),typeof(b)}(M, Minv, b)
    end
end

HMCUtilities.constrain(c::TestAffineConstraint, y) = c.Minv * (y - c.b)
HMCUtilities.free(c::TestAffineConstraint, x) = c.M * x + c.b

struct TestSphericalConstraint <: HMCUtilities.VariableConstraint{3,2} end

function HMCUtilities.constrain(c::TestSphericalConstraint, y)
    sinθ, cosθ = sincos(y[1])
    sinϕ, cosϕ = sincos(y[2])
    return [sinθ * cosϕ, sinθ * sinϕ, cosθ]
end

function HMCUtilities.free(c::TestSphericalConstraint, x)
    ϕ = atan(x[2], x[1])
    θ = acos(x[3])
    return [θ, ϕ]
end


struct TestSquareConstraint <: HMCUtilities.UnivariateConstraint end

HMCUtilities.constrain(c::TestSquareConstraint, y::Real) = y^2

HMCUtilities.free(c::TestSquareConstraint, x::Real) = sqrt(x)


@testset "defaults" begin
    @testset "one-to-one" begin
        M = [1.0 2.0;
             3.0 4.0]
        b = [5.0, 6.0]

        c = TestAffineConstraint(M, b)
        xs = [zeros(2), ones(2), [1.0, 2.0]]
        ys = [b, [8.0, 13.0], [10.0, 17.0]]
        logπxs = [-dot(x, x) / 2 for x in xs]
        ∇x_logπxs = [-x for x in xs]
        Minv = inv(M)
        logπys = logπxs .- log(2)
        ∇y_logπys = [Minv'x for x in ∇x_logπxs]

        for (x, y, logπx, ∇x_logπx, y_exp, logπy_exp, ∇y_logπy_exp) in zip(xs, ys, logπxs, ∇x_logπxs, ys, logπys, ∇y_logπys)
            test_constraint(
                c,
                x,
                y,
                logπx,
                ∇x_logπx,
                y_exp,
                logπy_exp,
                ∇y_logπy_exp
            )
        end
    end

    @testset "non one-to-one" begin
        c = TestSphericalConstraint()
        xs = [[sqrt(3 / 2), sqrt(3 / 2), 1] ./ 2]
        ys = [π .* [1 / 3, 1 / 4]]
        κμ = 10.0 * [1.0, 0.0, 0.0]
        logπxs = [dot(x, κμ) for x in xs]  # vMF
        ∇x_logπxs = [κμ for x in xs]
        logπys = [(5 * sqrt(6) - log(4 / 3)) / 2]
        ∇y_logπys = [[5 / sqrt(2) + 1 / sqrt(3), -5 * sqrt(3 / 2)]]

        for (x, y, logπx, ∇x_logπx, y_exp, logπy_exp, ∇y_logπy_exp) in zip(xs, ys, logπxs, ∇x_logπxs, ys, logπys, ∇y_logπys)
            test_constraint(
                c,
                x,
                y,
                logπx,
                ∇x_logπx,
                y_exp,
                logπy_exp,
                ∇y_logπy_exp
            )
        end
    end

    @testset "univariate" begin
        c = TestSquareConstraint()
        x = rand()
        y = -sqrt(x)
        y_exp = abs(y)
        logπx = x^2 / 2
        ∇x_logπx = x
        logπy_exp = logπx + log(2 * abs(y))
        ∇y_logπy_exp = 2 * y^3 + inv(y)

        @testset "scalar" begin
            vtypes = [Float64]
            test_constraint(
                c,
                x,
                y,
                logπx,
                ∇x_logπx,
                y_exp,
                logπy_exp,
                ∇y_logπy_exp;
                cvtypes=vtypes,
                fvtypes=vtypes
            )
        end

        @testset "vector" begin
            vtypes = [Vector{Float64}]
            test_constraint(
                c,
                [x],
                [y],
                logπx,
                [∇x_logπx],
                [y_exp],
                logπy_exp,
                [∇y_logπy_exp];
                cvtypes=vtypes,
                fvtypes=vtypes
            )
        end
    end
end

@testset "IdentityConstraint" begin
    @testset "vector n = $n" for n in 1:3
        for m in 1:3
            c = HMCUtilities.IdentityConstraint(n)
            x = randn(n)
            y = copy(x)
            logπx = -dot(x, x) / 2
            ∇x_logπx = -x
            logπy_exp = copy(logπx)
            ∇y_logπy_exp = copy(∇x_logπx)
            test_constraint(c, x, y, logπx, ∇x_logπx, y, logπy_exp, ∇y_logπy_exp)
        end
    end

    @testset "scalar" begin
        c = HMCUtilities.IdentityConstraint(1)
        vtypes = [Float64, Float32]
        x = randn()
        y = copy(x)
        logπx = -dot(x, x) / 2
        ∇x_logπx = -x
        logπy_exp = copy(logπx)
        ∇y_logπy_exp = copy(∇x_logπx)
        test_constraint(
            c,
            x,
            y,
            logπx,
            ∇x_logπx,
            y,
            logπy_exp,
            ∇y_logπy_exp;
            cvtypes=vtypes,
            fvtypes=vtypes
        )
    end
end

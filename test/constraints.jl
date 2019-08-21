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
            vtypes = [Float64, Float32]
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
            vtypes = [Vector{Float64}, Vector{Float32}]
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

@testset "LowerBoundedConstraint" begin
    vtypes = [Float64, Float32]

    @testset "lb=-5, y=0" begin
        c = HMCUtilities.LowerBoundedConstraint(-5.0)
        (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (0, -4, -8, 4, -8, 5)
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

    @testset "lb=5, y=-5" begin
        c = HMCUtilities.LowerBoundedConstraint(5.0)
        (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (
            -5, 5.00673794700, -12.5337124350, -5.00673794700, -17.5337124350, 0.966264865075
        )
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


@testset "UpperBoundedConstraint" begin
    vtypes = [Float64, Float32]

    @testset "ub=-5, y=0" begin
        c = HMCUtilities.UpperBoundedConstraint(-5.0)
        (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (0, -6, -18, 6, -18, -5)
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

    @testset "ub=5, y=-5" begin
        c = HMCUtilities.UpperBoundedConstraint(5.0)
        (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (
            -5, 4.99326205300, -12.4663329650, -4.99326205300, -17.4663329650, 1.03364433507
        )
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

@testset "BoundedConstraint" begin
    vtypes = [Float64]

    @testset "lb=0, ub=1" begin
        c = HMCUtilities.BoundedConstraint(0.0, 1.0)
        @testset "y=-20" begin
            (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (
                -20, 2.0611536181902035814e-9, 0.0, 0.0, -20, 1.0
            )
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

        @testset "y=20" begin
            (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (
                20, 0.99999999793884638181, -0.5, -1, -20.5, -1
            )
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

    @testset "lb=-5, ub=5" begin
        c = HMCUtilities.BoundedConstraint(-5.0, 5.0)
        @testset "y=1" begin
            (y, x, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) = (
                1, 2.310585786, -2.669403338, -2.310585786, -1.993341620, -5.005004541
            )
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
end

@testset "TransformConstraint" begin
    (lb, ub) = (-rand(), rand())
    @test_throws AssertionError HMCUtilities.TransformConstraint(ub, lb)

    c = HMCUtilities.TransformConstraint(lb, Inf)
    @test c === HMCUtilities.LowerBoundedConstraint(lb)

    c = HMCUtilities.TransformConstraint(-Inf, ub)
    @test c === HMCUtilities.UpperBoundedConstraint(ub)

    c = HMCUtilities.TransformConstraint(lb, ub)
    @test c === HMCUtilities.BoundedConstraint(lb, ub)

    c = HMCUtilities.TransformConstraint()
    @test c === HMCUtilities.IdentityConstraint(1)
    c = HMCUtilities.TransformConstraint(-Inf, Inf)
    @test c === HMCUtilities.IdentityConstraint(1)
end

@testset "UnitVectorConstraint" begin
    vtypes = [Vector{Float64}, Vector{Float32}]

    @testset "n=$n" for n in 2:4
        c = HMCUtilities.UnitVectorConstraint(n)
        κμ = rand() * normalize(randn(n))

        y = randn(n)
        x = normalize(y)
        y_exp = copy(x)
        logπx = dot(x, κμ)  # vMF
        ∇x_logπx = κμ
        logπy_exp = logπx - dot(y, y) / 2
        ∇y_logπy_exp = (I - x * x') * ∇x_logπx / norm(y) - y

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
end

@testset "JointConstraint" begin
    vtypes = [Vector{Float64}, Vector{Float32}]

    cs = [HMCUtilities.IdentityConstraint(3),
          HMCUtilities.LowerBoundedConstraint(0.0),
          HMCUtilities.UpperBoundedConstraint(1.0),
          HMCUtilities.BoundedConstraint(0.0, 1.0),
          HMCUtilities.UnitVectorConstraint(4)]

    y = Float64[]
    x = Float64[]
    y_exp = Float64[]
    logπx = randn()
    ∇x_logπx = Float64[]
    ∇y_logπy_exp = Float64[]
    logdetJ_exp = sum(cs) do (c)
        yi = randn(HMCUtilities.free_dimension(c))
        xi = constrain(c, yi)
        yi_exp = free(c, xi)
        ∇x_logπxi = randn(length(xi))
        _, pushlogpdf = constrain_with_pushlogpdf(c, yi)
        logdetJi, ∇y_logπyi = pushlogpdf(0.0, ∇x_logπxi)
        push!(y, yi...)
        push!(x, xi...)
        push!(y_exp, yi_exp...)
        push!(∇x_logπx, ∇x_logπxi...)
        push!(∇y_logπy_exp, ∇y_logπyi...)
        return logdetJi
    end
    logπy_exp = logπx + logdetJ_exp

    jc = HMCUtilities.JointConstraint(cs...)

    test_constraint(
        jc,
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

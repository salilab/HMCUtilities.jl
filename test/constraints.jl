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

@testset "defaults" begin
    @testset "one-to-one" begin
        vtypes = [Vector]

        M = [1.0 2.0;
             3.0 4.0]
        b = [5.0, 6.0]

        c = TestAffineConstraint(M, b)
        xs = [zeros(2), ones(2), [1.0, 2.0]]
        ys = [b, [8.0, 13.0], [10.0, 17.0]]
        logπxs = [-dot(x, x) / 2 for x in xs]
        ∇x_logπxs = [-x for x in xs]
        Minv = inv(M)
        jacobians = [Minv, Minv, Minv]

        logπys = logπxs .- log(2)
        ∇y_logπys = [Minv'x for x in ∇x_logπxs]

        test_constraint(
            c,
            xs,
            ys,
            logπxs,
            ∇x_logπxs,
            logπys,
            ∇y_logπys;
            jacobians=jacobians,
            cvtypes=vtypes,
            fvtypes=vtypes
        )
    end

    @testset "non one-to-one" begin
        cvtypes = [Vector]
        fvtypes = [Vector]

        c = TestSphericalConstraint()
        xs = [[sqrt(3 / 2), sqrt(3 / 2), 1] ./ 2]
        ys = [π .* [1 / 3, 1 / 4]]
        κμ = 10.0 * [1.0, 0.0, 0.0]
        logπxs = [dot(x, κμ) for x in xs]  # vMF
        ∇x_logπxs = [κμ for x in xs]

        jacobians = [[1 -sqrt(3); 1 sqrt(3); -sqrt(6) 0] ./ (2 * sqrt(2))]

        logπys = [(5 * sqrt(6) - log(4 / 3)) / 2]
        ∇y_logπys = [[5 / sqrt(2) + 1 / sqrt(3), -5 * sqrt(3 / 2)]]

        test_constraint(
            c,
            xs,
            ys,
            logπxs,
            ∇x_logπxs,
            logπys,
            ∇y_logπys;
            jacobians=jacobians,
            cvtypes=cvtypes,
            fvtypes=fvtypes
        )
    end
end

import HMCUtilities:
    VariableConstraint,
    constrain,
    free

struct TestAffineConstraint{N,TM,TMinv,Tb,F} <: HMCUtilities.VariableConstraint{N,N}
    M::TM
    Minv::TMinv
    b::Tb

    function TestAffineConstraint(M, b, jac_flag)
        n = length(b)
        @assert size(M) == (n, n)
        Minv = inv(M)
        return new{n,typeof(M),typeof(Minv),typeof(b),jac_flag}(M, Minv, b)
    end
end

TestAffineConstraintWithJac{N,TM,TMinv,Tb} = TestAffineConstraint{N,TM,TMinv,Tb,true}
TestAffineConstraintWithNoJac{N,TM,TMinv,Tb} = TestAffineConstraint{N,TM,TMinv,Tb,false}

HMCUtilities.constrain(c::TestAffineConstraint, y) = c.Minv * (y - c.b)
HMCUtilities.free(c::TestAffineConstraint, x) = c.M * x + c.b

HMCUtilities.free_jacobian(c::TestAffineConstraintWithJac, y) = c.Minv

function HMCUtilities.free_logpdf_correction(c::TestAffineConstraintWithNoJac, y)
    return first(logabsdet(c.Minv))
end

@testset "defaults" begin
    M = [1.0 2.0;
         3.0 4.0]
    b = [5.0, 6.0]

    xs = [zeros(2), ones(2), [1.0, 2.0]]
    ys = [b, [8.0, 13.0], [10.0, 17.0]]
    logπxs = [-dot(x, x) / 2 for x in xs]
    ∇x_logπxs = [-x for x in xs]
    Minv = inv(M)
    jacobians = [Minv, Minv, Minv]

    logπys = logπxs .- log(2)
    ∇y_logπys = [Minv'x for x in ∇x_logπxs]

    @testset "explicit `free_jacobian`" begin
        c = TestAffineConstraint(M, b, true)
        test_constraint(
            c,
            xs,
            ys,
            logπxs,
            ∇x_logπxs,
            logπys,
            ∇y_logπys;
            jacobians=jacobians
        )
    end

    @testset "explicit `free_logpdf_correction`" begin
        c2 = TestAffineConstraint(M, b, false)
        test_constraint(
            c2,
            xs,
            ys,
            logπxs,
            ∇x_logπxs,
            logπys,
            ∇y_logπys;
            jacobians=jacobians
        )
    end
end

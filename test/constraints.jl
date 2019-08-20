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


end


@testset "defaults" begin
    vtypes = [Vector, SVector{2}, MVector{2}]
    mtypes = [Matrix, SMatrix{2,2}, MMatrix{2,2}]

    @testset "TestAffineConstraint" begin
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
            vtypes=vtypes,
            mtypes=mtypes
        )
    end
end

using Zygote
using HMCUtilities:
    VariableConstraint,
    free_dimension,
    constrain_dimension,
    free,
    constrain,
    constrain_with_pushlogpdf,
    constrain_with_pushlogpdf_grad,
    _logabsdet

_size(x) = size(x)
_size(x::Number) = (length(x),)

function logabsdetjac(c, y)
    J = HMCUtilities.constrain_jacobian(c, y)
    return _logabsdet(J'J)[1] / 2
end

_format_grad(z, x) = Zygote.accum(z, x)

function test_constraint(
        c,
        x,
        y,
        logπx,
        ∇x_logπx,
        y_exp,
        logπy_exp,
        ∇y_logπy_exp;
        cvtypes=[Vector{Float64}, Vector{Float32}],
        fvtypes=[Vector{Float64}, Vector{Float32}],
        test_type_stability = true,
        test_logdetJ_consistency = true
    )
    @test isa(c, VariableConstraint)

    @testset "$CVType, $FVType" for (CVType, FVType) in zip(cvtypes, fvtypes)
        _test_constraint(
            c,
            convert(CVType, x),
            convert(FVType, y),
            convert(eltype(CVType), logπx),
            convert(CVType, ∇x_logπx),
            convert(FVType, y_exp),
            convert(eltype(FVType), logπy_exp),
            convert(FVType, ∇y_logπy_exp);
            test_type_stability = test_type_stability,
            test_logdetJ_consistency = test_logdetJ_consistency
        )
    end
end

function _test_constraint(
    c,
    x,
    y,
    logπx,
    ∇x_logπx,
    y_exp,
    logπy_exp,
    ∇y_logπy_exp;
    test_type_stability = true,
    test_logdetJ_consistency = true
)

    @testset "free" begin
        y2 = free(c, x)
        @test _size(y2) == (free_dimension(c),)
        @test y2 ≈ y_exp
        x3 = constrain(c, y2)
        @test _size(x3) == (constrain_dimension(c),)
        @test x3 ≈ x
    end

    @testset "constrain" begin
        x2 = constrain(c, y)
        @test _size(x2) == (constrain_dimension(c),)
        @test x2 ≈ x
    end

    @testset "constrain_with_pushlogpdf" begin
        x2, pushlogpdf = constrain_with_pushlogpdf(c, y)
        @test x2 ≈ constrain(c, y)
        logπy = pushlogpdf(logπx)
        @test isreal(logπy)
        @test logπy ≈ logπy_exp
    end

    @testset "constrain_with_pushlogpdf_grad" begin
        x2, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, y)
        @test x2 ≈ constrain(c, y)
        logπy, ∇y_logπy = pushlogpdf_grad(logπx, ∇x_logπx)
        @test isreal(logπy)
        @test logπy ≈ logπy_exp
        @test _size(∇y_logπy) == (free_dimension(c),)
        @test ∇y_logπy ≈ ∇y_logπy_exp
    end

    @testset "jacobian consistency" begin
        J = HMCUtilities.constrain_jacobian(c, y)
        J2, back = Zygote.forward(HMCUtilities.constrain_jacobian, c, y)
        @test J ≈ J2
    end

    if test_logdetJ_consistency
        @testset "logabsdet jacobian consistency" begin
            logdetJ = logabsdetjac(c, y)
            _, pushlogpdf = constrain_with_pushlogpdf(c, y)
            @test logdetJ ≈ pushlogpdf(zero(eltype(y)))

            ∇y_logdetJ = _format_grad(zero(∇y_logπy_exp), Zygote.gradient(logabsdetjac, c, y)[2])
            _, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, y)
            logdetJ2, ∇y_logdetJ2 = pushlogpdf_grad(zero(logπx), zero(∇x_logπx))
            @test logdetJ ≈ logdetJ2
            @test isapprox(∇y_logdetJ, ∇y_logdetJ2; atol=1e-6)
        end
    end

    if test_type_stability
        @testset "type stability" begin
            @inferred free(c, x)
            @inferred constrain(c, y)

            @inferred (y -> constrain_with_pushlogpdf(c, y)[1])(y)
            _, pushlogpdf = constrain_with_pushlogpdf(c, y)
            @inferred pushlogpdf(logπx)

            @inferred (y -> constrain_with_pushlogpdf_grad(c, y)[1])(y)
            _, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, y)
            @inferred pushlogpdf_grad(logπx, ∇x_logπx)
        end
    end
end

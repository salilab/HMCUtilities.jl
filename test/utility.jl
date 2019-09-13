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
        tx = convert(CVType, x)
        ty = convert(FVType, y)
        t∇x_logπx = convert(CVType, ∇x_logπx)

        @testset "free" begin
            y2 = free(c, tx)
            @test _size(y2) == (free_dimension(c),)
            @test y2 ≈ convert(FVType, y_exp)
            x3 = constrain(c, y2)
            @test _size(x3) == (constrain_dimension(c),)
            @test x3 ≈ tx
        end

        @testset "constrain" begin
            x2 = constrain(c, ty)
            @test _size(x2) == (constrain_dimension(c),)
            @test x2 ≈ tx
        end

        @testset "constrain_with_pushlogpdf" begin
            x2, pushlogpdf = constrain_with_pushlogpdf(c, ty)
            @test x2 ≈ constrain(c, ty)
            logπy = pushlogpdf(logπx)
            @test isreal(logπy)
            @test logπy ≈ convert(eltype(FVType), logπy_exp)
        end

        @testset "constrain_with_pushlogpdf_grad" begin
            x2, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, ty)
            @test x2 ≈ constrain(c, ty)
            logπy, ∇y_logπy = pushlogpdf_grad(logπx, ∇x_logπx)
            @test isreal(logπy)
            @test logπy ≈ convert(eltype(FVType), logπy_exp)
            @test _size(∇y_logπy) == (free_dimension(c),)
            @test ∇y_logπy ≈ convert(FVType, ∇y_logπy_exp)
        end

        @testset "jacobian consistency" begin
            J = HMCUtilities.constrain_jacobian(c, ty)
            J2, back = Zygote.forward(HMCUtilities.constrain_jacobian, c, ty)
            @test J ≈ J2
        end

        if test_logdetJ_consistency
            @testset "logabsdet jacobian consistency" begin
                logdetJ = logabsdetjac(c, ty)
                _, pushlogpdf = constrain_with_pushlogpdf(c, ty)
                @test logdetJ ≈ pushlogpdf(zero(logπx))

                ∇y_logdetJ = _format_grad(zero(∇y_logπy_exp), Zygote.gradient(logabsdetjac, c, ty)[2])
                _, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, ty)
                logdetJ2, ∇y_logdetJ2 = pushlogpdf_grad(zero(logπx), zero(∇x_logπx))
                @test logdetJ ≈ logdetJ2
                @test ∇y_logdetJ ≈ ∇y_logdetJ2
            end
        end

        if test_type_stability
            @testset "type stability" begin
                @inferred free(c, tx)
                @inferred constrain(c, ty)

                @inferred (y -> constrain_with_pushlogpdf(c, y)[1])(ty)
                _, pushlogpdf = constrain_with_pushlogpdf(c, ty)
                @inferred pushlogpdf(logπx)

                @inferred (y -> constrain_with_pushlogpdf_grad(c, y)[1])(ty)
                _, pushlogpdf_grad = constrain_with_pushlogpdf_grad(c, ty)
                @inferred pushlogpdf_grad(logπx, ∇x_logπx)
            end
        end
    end
end

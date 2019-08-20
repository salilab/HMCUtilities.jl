using HMCUtilities:
    free_dimension,
    constrain_dimension,
    free,
    constrain,
    free_jacobian,
    free_logpdf_correction,
    free_logpdf_gradient

function test_constraint(
        c,
        xs,
        ys,
        logπxs,
        ∇x_logπxs,
        logπys,
        ∇y_logπys;
        jacobians=[],
        cvtypes=[Vector],
        fvtypes=[Vector],
    )
    @test isa(c, VariableConstraint)

    @testset "$CVType, $FVType" for (CVType, FVType) in zip(cvtypes, fvtypes)
        txs = convert.(CVType, xs)
        tys = convert.(FVType, ys)
        t∇x_logπxs = convert.(CVType, ∇x_logπxs)

        @testset "constrain" begin
            for (x_exp, y) in zip(txs, tys)
                @inferred constrain(c, y)
                x = constrain(c, y)
                @test length(x) == constrain_dimension(c)
                @test x ≈ x_exp
            end
        end

        @testset "free" begin
            for (x, y_exp) in zip(txs, tys)
                @inferred free(c, x)
                y = free(c, x)
                @test length(y) == free_dimension(c)
                x2 = constrain(c, y)
                @test x2 ≈ x
            end
        end

        @testset "free_logpdf_correction" begin
            for (y, logπx_exp, logπy_exp) in zip(tys, logπxs, logπys)
                @inferred free_logpdf_correction(c, y)
                @test free_logpdf_correction(c, y) ≈ logπy_exp - logπx_exp
            end
        end

        @testset "free_logpdf_gradient" begin
            for (y, logπx, ∇x_logπx, logπy_exp, ∇y_logπy_exp) in zip(tys, logπxs, t∇x_logπxs, logπys, ∇y_logπys)
                @inferred free_logpdf_gradient(c, y, logπx, ∇x_logπx)
                logπy, ∇y_logπy = free_logpdf_gradient(c, y, logπx, ∇x_logπx)
                @test logπy ≈ logπy_exp
                @test length(∇y_logπy) == free_dimension(c)
                @test ∇y_logπy ≈ ∇y_logπy_exp
            end
        end

        if !isempty(jacobians)
            @testset "free_jacobian" begin
                for (y, J_exp) in zip(tys, jacobians)
                    @inferred free_jacobian(c, y)
                    J = free_jacobian(c, y)
                    @test J ≈ J_exp
                end
            end
        end
    end
end

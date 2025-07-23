@testset "Parametric Models Tests" begin
    
    @testset "Weibull Model Fitting" begin
        sample_data = _create_sample_loan_data()
        
        # Test Weibull model without covariates
        model = fit_parametric_model(sample_data, distribution=:weibull)
        
        @test isa(model, WeibullPrepaymentModel)
        @test model.shape > 0.0
        @test model.n_observations > 0
        @test isfinite(model.loglikelihood)
        
        # Test with covariates
        covariates = [:interest_rate, :ltv_ratio]
        model_with_covs = fit_parametric_model(
            sample_data, 
            distribution=:weibull, 
            covariates=covariates
        )
        
        @test length(model_with_covs.scale_coefficients) == length(covariates) + 1  # +1 for intercept
        @test model_with_covs.covariate_names == covariates
    end
    
    @testset "Log-Normal Model Fitting" begin
        sample_data = _create_sample_loan_data()
        covariates = [:credit_score]
        
        model = fit_parametric_model(
            sample_data, 
            distribution=:lognormal, 
            covariates=covariates
        )
        
        @test isa(model, LogNormalPrepaymentModel)
        @test model.scale > 0.0
        @test length(model.location_coefficients) == length(covariates) + 1
        @test isfinite(model.loglikelihood)
        @test isfinite(model.aic)
    end
    
    @testset "Weibull Survival Probability" begin
        sample_data = _create_sample_loan_data()
        model = fit_parametric_model(sample_data, distribution=:weibull, 
                                   covariates=[:interest_rate])
        
        test_covariates = Dict(:interest_rate => 4.0)
        
        # Test survival probability at different times
        times = [6.0, 12.0, 24.0, 36.0]
        for t in times
            surv_prob = survival_probability(model, test_covariates, t)
            @test 0.0 <= surv_prob <= 1.0
        end
        
        # Survival probability should decrease over time
        prob_6m = survival_probability(model, test_covariates, 6.0)
        prob_36m = survival_probability(model, test_covariates, 36.0)
        @test prob_36m <= prob_6m
    end
    
    @testset "Log-Normal Survival Probability" begin
        sample_data = _create_sample_loan_data()
        model = fit_parametric_model(sample_data, distribution=:lognormal,
                                   covariates=[:ltv_ratio])
        
        test_covariates = Dict(:ltv_ratio => 0.8)
        
        surv_prob = survival_probability(model, test_covariates, 24.0)
        @test 0.0 <= surv_prob <= 1.0
    end
    
    @testset "Weibull Hazard Function" begin
        sample_data = _create_sample_loan_data()
        model = fit_parametric_model(sample_data, distribution=:weibull,
                                   covariates=[:interest_rate])
        
        test_covariates = Dict(:interest_rate => 3.5)
        
        # Test hazard at different times
        times = [6.0, 12.0, 24.0]
        hazards = [hazard_function(model, test_covariates, t) for t in times]
        
        @test all(hazards .>= 0.0)
        @test all(isfinite.(hazards))
    end
    
    @testset "Median Survival Time" begin
        sample_data = _create_sample_loan_data()
        
        # Weibull model
        weibull_model = fit_parametric_model(sample_data, distribution=:weibull,
                                           covariates=[:interest_rate])
        
        test_covariates = Dict(:interest_rate => 4.0)
        median_time = median_survival_time(weibull_model, test_covariates)
        @test median_time > 0.0
        @test isfinite(median_time)
        
        # Log-normal model
        lognormal_model = fit_parametric_model(sample_data, distribution=:lognormal,
                                             covariates=[:interest_rate])
        
        median_time_ln = median_survival_time(lognormal_model, test_covariates)
        @test median_time_ln > 0.0
        @test isfinite(median_time_ln)
    end
    
    @testset "Model Comparison" begin
        sample_data = _create_sample_loan_data()
        
        # Fit multiple models
        weibull_model = fit_parametric_model(sample_data, distribution=:weibull,
                                           covariates=[:interest_rate])
        lognormal_model = fit_parametric_model(sample_data, distribution=:lognormal,
                                             covariates=[:interest_rate])
        
        models = [weibull_model, lognormal_model]
        comparison = model_comparison(models)
        
        @test nrow(comparison) == 2
        @test "Model" in names(comparison)
        @test "LogLikelihood" in names(comparison)
        @test "AIC" in names(comparison)
        @test "Distribution" in names(comparison)
        
        # Should be sorted by AIC
        @test issorted(comparison.AIC)
    end
    
    @testset "Design Matrix Construction" begin
        sample_data = _create_sample_loan_data()
        survival_df = _prepare_survival_data(sample_data, [:interest_rate, :ltv_ratio])
        
        X = _build_design_matrix(survival_df, [:interest_rate, :ltv_ratio])
        
        @test size(X, 1) == nrow(survival_df)
        @test size(X, 2) == 3  # intercept + 2 covariates
        @test all(X[:, 1] .== 1.0)  # Intercept column
    end
    
    @testset "Likelihood Functions" begin
        sample_data = _create_sample_loan_data()
        survival_df = _prepare_survival_data(sample_data, [:interest_rate])
        X = _build_design_matrix(survival_df, [:interest_rate])
        
        # Test Weibull likelihood
        θ_weibull = [1.0, -0.1, 0.8]  # [intercept, coef, sigma]
        ll_weibull = _weibull_loglikelihood(θ_weibull, survival_df, X)
        @test isfinite(ll_weibull)
        
        # Test log-normal likelihood  
        θ_lognormal = [3.0, -0.1, 0.5]  # [intercept, coef, sigma]
        ll_lognormal = _lognormal_loglikelihood(θ_lognormal, survival_df, X)
        @test isfinite(ll_lognormal)
    end
end
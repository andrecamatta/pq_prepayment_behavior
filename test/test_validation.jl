@testset "Validation Tests" begin
    
    @testset "Concordance Index Calculation" begin
        # Test with simple example
        predictions = [0.1, 0.3, 0.7, 0.9]
        times = [10.0, 8.0, 5.0, 2.0]
        events = [true, true, true, true]
        
        c_index = compute_concordance(predictions, times, events)
        
        @test 0.0 <= c_index <= 1.0
        
        # Perfect concordance case
        perfect_predictions = [0.2, 0.5, 0.7, 0.9]  # Higher risk = earlier event
        perfect_times = [10.0, 8.0, 5.0, 2.0]
        perfect_events = [true, true, true, true]
        
        perfect_c = compute_concordance(perfect_predictions, perfect_times, perfect_events)
        @test perfect_c >= 0.8  # Should be high concordance
        
        # Test with censored data
        mixed_events = [false, true, true, false]
        mixed_c = compute_concordance(predictions, times, mixed_events)
        @test 0.0 <= mixed_c <= 1.0
    end
    
    @testset "Model Validation - Cox" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data, covariates=[:interest_rate, :ltv_ratio])
        
        validation_results = validate_model(model, sample_data)
        
        @test haskey(validation_results, "model_summary")
        @test haskey(validation_results, "residuals")
        @test haskey(validation_results, "proportional_hazards_test")
        @test haskey(validation_results, "bootstrap_ci")
        
        # Check model summary
        summary = validation_results["model_summary"]
        @test summary["model_type"] == "Cox Proportional Hazards"
        @test summary["n_observations"] > 0
        @test summary["n_events"] > 0
    end
    
    @testset "Model Validation - Parametric" begin
        sample_data = _create_sample_loan_data()
        model = fit_parametric_model(sample_data, distribution=:weibull, 
                                   covariates=[:interest_rate])
        
        validation_results = validate_model(model, sample_data)
        
        @test haskey(validation_results, "model_summary")
        @test haskey(validation_results, "residuals")
        @test haskey(validation_results, "goodness_of_fit")
        
        # Check model summary
        summary = validation_results["model_summary"]
        @test summary["model_type"] == "Weibull AFT"
        @test summary["shape_parameter"] > 0.0
    end
    
    @testset "Likelihood Ratio Test" begin
        sample_data = _create_sample_loan_data()
        
        # Fit nested models
        model_reduced = fit_cox_model(sample_data, covariates=[:interest_rate])
        model_full = fit_cox_model(sample_data, covariates=[:interest_rate, :ltv_ratio])
        
        lr_test = likelihood_ratio_test(model_full, model_reduced)
        
        @test haskey(lr_test, :lr_statistic)
        @test haskey(lr_test, :degrees_of_freedom)
        @test haskey(lr_test, :p_value)
        @test haskey(lr_test, :significant)
        
        @test lr_test.lr_statistic >= 0.0
        @test lr_test.degrees_of_freedom == 1
        @test 0.0 <= lr_test.p_value <= 1.0
    end
    
    @testset "Cox Residuals" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data, covariates=[:interest_rate])
        
        residuals_df = _cox_residuals(model, sample_data)
        
        @test nrow(residuals_df) == length(sample_data.loan_id)
        @test "martingale_residual" in names(residuals_df)
        @test "deviance_residual" in names(residuals_df)
        @test all(isfinite.(residuals_df.martingale_residual))
        @test all(isfinite.(residuals_df.deviance_residual))
    end
    
    @testset "Parametric Residuals" begin
        sample_data = _create_sample_loan_data()
        model = fit_parametric_model(sample_data, distribution=:weibull,
                                   covariates=[:interest_rate])
        
        residuals_df = _parametric_residuals(model, sample_data)
        
        @test nrow(residuals_df) == length(sample_data.loan_id)
        @test "standardized_residual" in names(residuals_df)
        @test all(isfinite.(residuals_df.standardized_residual))
    end
    
    @testset "Out-of-Sample Validation" begin
        sample_data = _create_sample_loan_data()
        
        # Split data
        n_total = length(sample_data.loan_id)
        n_train = Int(floor(n_total * 0.7))
        train_indices = sample(1:n_total, n_train, replace=false)
        test_indices = setdiff(1:n_total, train_indices)
        
        train_data = _subset_loan_data(sample_data, train_indices)
        test_data = _subset_loan_data(sample_data, test_indices)
        
        # Fit model on training data
        model = fit_cox_model(train_data, covariates=[:interest_rate])
        
        # Validate on test data
        validation_results = validate_model(model, train_data, test_data=test_data)
        
        @test haskey(validation_results, "out_of_sample_metrics")
        out_of_sample = validation_results["out_of_sample_metrics"]
        
        @test haskey(out_of_sample, "concordance_index")
        @test 0.0 <= out_of_sample["concordance_index"] <= 1.0
    end
    
    @testset "Bootstrap Confidence Intervals" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data, covariates=[:interest_rate])
        
        # Use small number of bootstrap samples for testing
        ci_results = _bootstrap_confidence_intervals(model, sample_data, 
                                                   n_bootstrap=10, 
                                                   confidence_level=0.95)
        
        @test haskey(ci_results, "ci_lower")
        @test haskey(ci_results, "ci_upper")
        @test haskey(ci_results, "confidence_level")
        
        @test length(ci_results["ci_lower"]) == length(model.coefficients)
        @test length(ci_results["ci_upper"]) == length(model.coefficients)
        
        # Lower bounds should be less than upper bounds
        @test all(ci_results["ci_lower"] .< ci_results["ci_upper"])
    end
    
    @testset "Utility Functions" begin
        sample_data = _create_sample_loan_data()
        
        # Test parameter counting
        cox_model = fit_cox_model(sample_data, covariates=[:interest_rate, :ltv_ratio])
        @test _count_parameters(cox_model) == 2
        
        weibull_model = fit_parametric_model(sample_data, distribution=:weibull,
                                           covariates=[:interest_rate])
        @test _count_parameters(weibull_model) == 3  # intercept + coef + shape
        
        # Test loglikelihood extraction
        cox_ll = _extract_loglikelihood(cox_model)
        weibull_ll = _extract_loglikelihood(weibull_model)
        
        @test isfinite(cox_ll)
        @test isfinite(weibull_ll)
    end
    
    @testset "KS Test Statistic" begin
        using Distributions
        
        # Test with normal data
        sample = randn(100)
        ks_stat = _ks_test_statistic(sample, Normal(0, 1))
        
        @test 0.0 <= ks_stat <= 1.0
        
        # Perfect fit should have small KS statistic
        perfect_sample = quantile.(Normal(), (1:100) ./ 101)
        perfect_ks = _ks_test_statistic(perfect_sample, Normal())
        @test perfect_ks < 0.1
    end
end
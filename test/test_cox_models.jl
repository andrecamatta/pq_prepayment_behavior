@testset "Cox Models Tests" begin
    
    @testset "Cox Model Fitting" begin
        # Create sample data
        sample_data = _create_sample_loan_data()
        
        # Test basic Cox model without covariates
        model = fit_cox_model(sample_data)
        
        @test isa(model, CoxPrepaymentModel)
        @test model.n_observations > 0
        @test model.n_events > 0
        
        # Test Cox model with covariates
        covariates = [:interest_rate, :ltv_ratio, :credit_score]
        model_with_covs = fit_cox_model(sample_data, covariates=covariates)
        
        @test length(model_with_covs.coefficients) == length(covariates)
        @test model_with_covs.loglikelihood != 0.0
    end
    
    @testset "Hazard Ratio Calculation" begin
        sample_data = _create_sample_loan_data()
        covariates = [:interest_rate, :ltv_ratio]
        model = fit_cox_model(sample_data, covariates=covariates)
        
        # Test hazard ratio calculation
        test_covariates = Dict(:interest_rate => 4.0, :ltv_ratio => 0.8)
        hr = hazard_ratio(model, test_covariates)
        
        @test hr > 0.0
        @test isfinite(hr)
        
        # Test that different covariate values give different hazard ratios
        test_covariates2 = Dict(:interest_rate => 3.0, :ltv_ratio => 0.6)
        hr2 = hazard_ratio(model, test_covariates2)
        
        @test hr != hr2
    end
    
    @testset "Survival Curve Prediction" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data, covariates=[:interest_rate])
        
        test_covariates = Dict(:interest_rate => 4.0)
        times = collect(1.0:36.0)
        survival_probs = survival_curve(model, test_covariates, times=times)
        
        @test length(survival_probs) == length(times)
        @test all(0.0 .<= survival_probs .<= 1.0)
        @test issorted(survival_probs, rev=true)  # Should be decreasing
    end
    
    @testset "Cumulative Hazard Calculation" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data, covariates=[:ltv_ratio])
        
        test_covariates = Dict(:ltv_ratio => 0.8)
        times = collect(1.0:24.0)
        cum_hazard = cumulative_hazard(model, test_covariates, times=times)
        
        @test length(cum_hazard) == length(times)
        @test all(cum_hazard .>= 0.0)
        @test issorted(cum_hazard)  # Should be non-decreasing
    end
    
    @testset "Prepayment Prediction" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data, covariates=[:interest_rate, :ltv_ratio])
        
        # Predict on subset of data
        n_predict = min(20, length(sample_data.loan_id))
        subset_indices = 1:n_predict
        subset_data = _subset_loan_data(sample_data, collect(subset_indices))
        
        predictions = predict_prepayment(model, subset_data, 36)
        
        @test length(predictions) == n_predict
        @test all(0.0 .<= predictions .<= 1.0)
    end
    
    @testset "Survival Data Preparation" begin
        sample_data = _create_sample_loan_data()
        covariates = [:interest_rate, :ltv_ratio]
        
        survival_df = _prepare_survival_data(sample_data, covariates)
        
        @test nrow(survival_df) == length(sample_data.loan_id)
        @test "time" in names(survival_df)
        @test "event" in names(survival_df)
        @test all(survival_df.time .> 0.0)
        @test all(isa.(survival_df.event, Bool))
        
        # Check that covariates are included
        for cov in covariates
            @test string(cov) in names(survival_df)
        end
    end
    
    @testset "Baseline Hazard Extraction" begin
        sample_data = _create_sample_loan_data()
        model = fit_cox_model(sample_data)
        
        @test length(model.baseline_hazard) > 0
        @test length(model.times) == length(model.baseline_hazard)
        @test all(model.baseline_hazard .>= 0.0)
        @test issorted(model.times)
    end
end
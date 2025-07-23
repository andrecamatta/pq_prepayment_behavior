@testset "Data Loading Tests" begin
    
    @testset "LoanData Structure" begin
        # Test LoanData constructor with minimal data
        loan_ids = ["LOAN001", "LOAN002", "LOAN003"]
        orig_dates = [Date(2020, 1, 1), Date(2020, 2, 1), Date(2020, 3, 1)]
        maturity_dates = [Date(2050, 1, 1), Date(2050, 2, 1), Date(2050, 3, 1)]
        rates = [3.5, 4.0, 3.8]
        amounts = [300000.0, 250000.0, 400000.0]
        ltvs = [0.8, 0.75, 0.85]
        scores = [750, 720, 780]
        prop_types = ["SFR", "SFR", "CONDO"]
        prepay_dates = [missing, Date(2022, 1, 1), missing]
        default_dates = [missing, missing, missing]
        covariates = DataFrame()
        
        data = LoanData(
            loan_ids, orig_dates, maturity_dates, rates, amounts,
            ltvs, scores, prop_types, prepay_dates, default_dates, covariates
        )
        
        @test length(data.loan_id) == 3
        @test data.interest_rate[1] == 3.5
        @test data.ltv_ratio[2] == 0.75
        @test !ismissing(data.prepayment_date[2])
        @test ismissing(data.prepayment_date[1])
    end
    
    @testset "Data Preprocessing" begin
        # Create sample data for testing
        sample_data = _create_sample_loan_data()
        
        # Test filtering with default parameters
        filtered_data = preprocess_loan_data(sample_data)
        
        @test length(filtered_data.loan_id) <= length(sample_data.loan_id)
        @test all(filtered_data.ltv_ratio .<= 1.0)
        @test all(filtered_data.credit_score .>= 300)
        
        # Test with stricter filtering
        strict_filtered = preprocess_loan_data(
            sample_data, 
            min_seasoning_months=12,
            max_ltv=0.9,
            min_credit_score=600
        )
        
        @test length(strict_filtered.loan_id) <= length(filtered_data.loan_id)
        @test all(strict_filtered.ltv_ratio .<= 0.9)
        @test all(strict_filtered.credit_score .>= 600)
    end
    
    @testset "Freddie Mac Data Processing" begin
        # Test with mock Freddie Mac data structure
        mock_orig_data = DataFrame(
            loan_sequence_number = [1, 2, 3],
            origination_date = [202001, 202002, 202003],
            original_loan_term = [360, 360, 300],
            original_interest_rate = [3.5, 4.0, 3.8],
            original_loan_amount = [300000, 250000, 400000],
            original_ltv = [80, 75, 85],
            credit_score = [750, 720, 780],
            property_type = ["SFR", "SFR", "CO"],
            loan_purpose = ["P", "P", "R"]
        )
        
        mock_perf_data = DataFrame(
            loan_sequence_number = [1, 2, 3],
            zero_balance_code = [missing, "01", missing],
            zero_balance_effective_date = [missing, 202201, missing]
        )
        
        # This would test the actual processing function
        # processed_data = _process_freddie_mac_files(mock_orig_data, mock_perf_data)
        # @test length(processed_data.loan_id) == 3
        # @test !ismissing(processed_data.prepayment_date[2])
    end
end

function _create_sample_loan_data()::LoanData
    n = 100
    
    loan_ids = ["LOAN$(lpad(i, 3, '0'))" for i in 1:n]
    orig_dates = [Date(2020, 1, 1) + Month(rand(0:24)) for _ in 1:n]
    maturity_dates = [d + Year(30) for d in orig_dates]
    rates = 3.0 .+ 2.0 .* rand(n)
    amounts = 200000.0 .+ 300000.0 .* rand(n)
    ltvs = 0.6 .+ 0.4 .* rand(n)
    scores = 600 .+ round.(Int, 200 .* rand(n))
    prop_types = rand(["SFR", "CONDO", "TOWNHOUSE"], n)
    
    # Some loans have prepayment
    prepay_dates = Vector{Union{Date, Missing}}(undef, n)
    for i in 1:n
        if rand() < 0.3  # 30% prepayment rate
            prepay_dates[i] = orig_dates[i] + Month(rand(6:60))
        else
            prepay_dates[i] = missing
        end
    end
    
    default_dates = fill(missing, n)
    covariates = DataFrame(
        loan_purpose = rand(["Purchase", "Refinance"], n),
        property_state = rand(["CA", "TX", "FL", "NY"], n)
    )
    
    return LoanData(
        loan_ids, orig_dates, maturity_dates, rates, amounts,
        ltvs, scores, prop_types, prepay_dates, default_dates, covariates
    )
end
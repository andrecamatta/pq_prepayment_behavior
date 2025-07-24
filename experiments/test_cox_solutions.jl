#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, Random
import StatsBase
Random.seed!(42)

include("src/PrepaymentModels.jl")
using .PrepaymentModels

println("🔧 TESTANDO SOLUÇÕES PARA O PROBLEMA COX")
println(repeat("=", 50))

filepath = "data/official_based_data/brazilian_loans_2025-07-23_20-16.csv"
raw_data = CSV.read(filepath, DataFrame)

sample_size = 1000
sample_indices = StatsBase.sample(1:nrow(raw_data), sample_size, replace=false)
sample_df = raw_data[sample_indices, :]

train_data = sample_df[1:700, :]

function create_loan_data(df::DataFrame)
    return PrepaymentModels.LoanData(
        String.(df.loan_id),
        df.origination_date,
        df.maturity_date,
        df.interest_rate,
        df.loan_amount,
        df.loan_term,
        df.credit_score,
        df.borrower_income,
        df.loan_type,
        df.collateral_type,
        df.prepayment_date,
        fill(missing, nrow(df)),
        DataFrame()
    )
end

loan_train = create_loan_data(train_data)

println("📊 SOLUTION 1: Credit Score Binned (Categorical)")
# Create credit score bins instead of continuous
train_data_binned = copy(train_data)
credit_quantiles = quantile(train_data.credit_score, [0.33, 0.67])
train_data_binned.credit_score_binned = ifelse.(
    train_data.credit_score .<= credit_quantiles[1], 1,
    ifelse.(train_data.credit_score .<= credit_quantiles[2], 2, 3)
)

println("Credit score bins:")
println("Bin 1 (low): $(sum(train_data_binned.credit_score_binned .== 1)) observations")
println("Bin 2 (med): $(sum(train_data_binned.credit_score_binned .== 2)) observations") 
println("Bin 3 (high): $(sum(train_data_binned.credit_score_binned .== 3)) observations")

# Test prepayment rates by bin
for bin in 1:3
    bin_data = train_data_binned[train_data_binned.credit_score_binned .== bin, :]
    prepay_rate = mean(.!ismissing.(bin_data.prepayment_date))
    println("Bin $bin prepayment rate: $(round(prepay_rate*100, digits=1))%")
end

println("\n📊 SOLUTION 2: Regularization via Ridge")
# Test with smaller sample to reduce overfitting
small_sample = 200
small_indices = StatsBase.sample(1:nrow(train_data), small_sample, replace=false)
small_train = train_data[small_indices, :]
loan_small = create_loan_data(small_train)

println("Testing with smaller sample (n=$small_sample):")
try
    cox_small = PrepaymentModels.fit_cox_model(loan_small, covariates=[:interest_rate, :credit_score])
    println("✅ Small sample model fitted")
    println("Coefficients: $(round.(cox_small.coefficients, digits=4))")
    println("Hazard ratios: $(round.(exp.(cox_small.coefficients), digits=4))")
catch e
    println("❌ Error: $e")
end

println("\n📊 SOLUTION 3: Alternative Normalization")
# Let's check what happens with different normalization
println("Testing credit score with robust normalization (IQR):")
try
    # Test with quartile-based normalization
    q25, q75 = quantile(train_data.credit_score, [0.25, 0.75])
    iqr = q75 - q25
    median_cs = median(train_data.credit_score)
    
    println("Median: $median_cs, IQR: $iqr")
    println("Current range after z-score: $((minimum(train_data.credit_score) - mean(train_data.credit_score))/std(train_data.credit_score)), $((maximum(train_data.credit_score) - mean(train_data.credit_score))/std(train_data.credit_score)))")
    
    robust_normalized = (train_data.credit_score .- median_cs) ./ iqr
    println("After IQR normalization range: $(round(minimum(robust_normalized), digits=4)), $(round(maximum(robust_normalized), digits=4))")
catch e
    println("❌ Error in robust normalization: $e")
end

println("\n📊 SOLUTION 4: Check for Outliers")
# Check if there are extreme outliers causing issues
cs_mean = mean(train_data.credit_score)
cs_std = std(train_data.credit_score)
z_scores = abs.((train_data.credit_score .- cs_mean) ./ cs_std)
extreme_outliers = sum(z_scores .> 3)
mild_outliers = sum(z_scores .> 2)

println("Credit score outliers:")
println("Extreme outliers (|z| > 3): $extreme_outliers")
println("Mild outliers (|z| > 2): $mild_outliers")

if extreme_outliers > 0
    outlier_indices = findall(z_scores .> 3)
    println("Extreme outlier values: $(train_data.credit_score[outlier_indices])")
end
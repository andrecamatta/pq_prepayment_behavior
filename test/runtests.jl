using Test
using PrepaymentModels

@testset "PrepaymentModels.jl" begin
    include("test_data_loading.jl")
    include("test_cox_models.jl")
    include("test_parametric_models.jl")
    include("test_validation.jl")
end
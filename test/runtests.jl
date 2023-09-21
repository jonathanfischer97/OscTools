using OscTools
using Test, Random

Random.seed!(1234)

@testset "OscTools.jl" begin
    # Write your tests here.
    ga_result = test4fixedGA()
    @test length(ga_result.fitvals) == 588
end

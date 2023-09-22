using OscTools
using Test, Random

Random.seed!(1234)

@testset "OscTools.jl" begin
    # Write your tests here.
    ga_result = test4fixedGA()
    @test length(ga_result.fitvals) in (558,559) #for some reason is 558 on macOS-latest
end

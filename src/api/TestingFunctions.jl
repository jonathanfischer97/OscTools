"""
    test4fixedGA(popsize=10000, fixedsymbols = [:DF, :K, :P, :A], fixedvalues = [1000., 1.0, 1.0, 3.16])

Test tun a GA with fixed constraints and fixed values for the parameters DF, K, P, and A.
"""
function test4fixedGA(popsize=10000, fixedsymbols = [:DF, :K, :P, :A], fixedvalues = [1000., 1.0, 1.0, 3.16]; kwargs...)
    #* Set up the default GA problem
    ga_problem = GAProblem()

    #* Fixed some constraints 
    set_fixed_constraints!(ga_problem.constraints, fixedsymbols)

    #* Assign the fixed values 
    set_fixed_values!(ga_problem.constraints, fixedvalues...)

    #* Set seed 
    Random.seed!(1234)

    #* Generate the initial population
    population = generate_population(ga_problem.constraints, popsize)

    #* Run the GA
    ga_results = run_GA(ga_problem, population; kwargs...)

    return make_ga_dataframe(ga_results, ga_problem.constraints), ga_results.oscillatory_idxs
end
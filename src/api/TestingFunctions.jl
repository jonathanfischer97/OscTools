function test4fixedGA()
    #* Set up the default GA problem
    ga_problem = GAProblem()

    #* Fixed some constraints 
    set_fixed_constraints!(ga_problem.constraints, [:DF, :K, :P, :A])

    #* Assign the fixed values 
    set_fixed_values!(ga_problem.constraints, 1000., 1.0, 1.0, 3.16)

    #* Set seed 
    Random.seed!(1234)

    #* Generate the initial population
    population = generate_population(ga_problem.constraints, 10000)

    #* Run the GA
    run_GA(ga_problem, population)
end
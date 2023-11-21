#< FITNESS FUNCTION CONSTRUCTOR ##
"""
    make_fitness_function_threaded(constraints::ConstraintSet, ode_problem::OP)

Multithreaded fitness function, allocated a merged array for each thread
    """
function make_fitness_function_threaded(constraints::CT, ode_problem::OP) where {CT<:ConstraintSet, OP<:ODEProblem}
    fixed_idxs = get_fixed_indices(constraints)
    fixed_values = [constraints[i].fixed_value for i in fixed_idxs]
    n_fixed = length(fixed_idxs)
    n_total = n_fixed + activelength(constraints) 

    non_fixed_indices = setdiff(1:n_total, fixed_idxs)

    # Create a ThreadLocal array
    # merged_inputs = [zeros(Float64, n_total+12) for _ in 1:Threads.nthreads()]

    # Fill in the fixed values
    # for input in merged_inputs
    #     input[fixed_idxs] .= fixed_values 
    # end

    merged_input = zeros(Float64, n_total+12)
    merged_input[fixed_idxs] .= fixed_values  # Fill in fixed values

    function fitness_function(input::Vector{Float64})
        # Get the merged_input array for the current thread
        # merged_input = merged_inputs[Threads.threadid()]
        local_merged_input = copy(merged_input) 
        local_merged_input[non_fixed_indices] .= input  # Fill in variable values

        return eval_fitness(local_merged_input, ode_problem)
    end

    return fitness_function
end
#> END



"""
    make_split_fitness_function(constraints::ConstraintSet, ode_problem::OP)

Constructs fitness function that seperates fitness evaluation and oscillation classification.
"""
function make_split_fitness_function(constraints::CT, ode_problem::OP) where {CT<:ConstraintSet, OP<:ODEProblem}
    fixed_idxs = get_fixed_indices(constraints)
    fixed_values = [constraints[i].fixed_value for i in fixed_idxs]
    n_fixed = length(fixed_idxs)
    n_total = n_fixed + activelength(constraints) 

    non_fixed_indices = setdiff(1:n_total, fixed_idxs)

    merged_input = zeros(Float64, n_total+12)

    merged_input[fixed_idxs] .= fixed_values  # Fill in fixed values

    function fitness_function(input::Vector{Float64})
        local_merged_input = copy(merged_input) 
        local_merged_input[non_fixed_indices] .= input  # Fill in variable values

        newprob = remake_prob(local_merged_input, ode_problem)
        sol = solve_odeprob(newprob)

        if sol.retcode != ReturnCode.Success
            return [-Inf, 0.0, 0.0]
        end

        #* Calculate Amem by summing all the AP2-bound species on the membrane
        Amem_sol = map(sum, sol.u)

        #* Normalize signal to be relative to total AP2 concentration
        Amem_sol ./= local_merged_input[17]

        #* Find the extrema of the time series
        max_idxs, max_vals, min_idxs, min_vals = findextrema(Amem_sol, min_height=0.1)

        period = 0.0
        amplitude = 0.0
        fitness = 0.0

        if is_oscillatory(Amem_sol, sol.t, max_idxs, min_idxs)
            period, amplitude = getPerAmp(sol.t, max_idxs, max_vals, min_idxs, min_vals)
            fitness += log10(period)
        end

        fitness += get_fitness!(Amem_sol)

        return [fitness, period, amplitude]
    end

    return fitness_function
end

#< GA PROBLEM TYPE ##
"""
    GAProblem{T <: ConstraintSet}

Struct encapsulating a Genetic Algorithm (GA) optimization problem. It holds the constraints for the problem, the ODE problem to be solved.

# Fields
- `constraints::T`: Constraints for the problem. Either `ParameterConstraints` or `InitialConditionConstraints` or `AllConstraints`.
- `ode_problem::ODEProblem`: ODE problem to be solved.
"""
@kwdef mutable struct GAProblem{CT <: ConstraintSet, OP <: ODEProblem}
    constraints::CT = AllConstraints()
    ode_problem::OP = make_ODE_problem()
end

# show method overload for GAProblem
function show(io::IO, ::MIME"text/plain", prob::GAProblem) 
    printstyled(io, typeof(prob.constraints), ":\n"; bold = true, underline=true, color = :green)
    printstyled(io, prob.constraints, "\n")
    printstyled(io, "\nNominal parameter values:\n"; bold = true, color = :blue)
    printstyled(io, prob.ode_problem.p, "\n")
    printstyled(io, "\nNominal initial conditions:\n"; bold = true, color = :blue)
    printstyled(io, prob.ode_problem.u0, "\n")

    printstyled(io, "\nFixed values:\n"; bold = true, color = :red)
    printstyled(io, [(con.name => con.fixed_value) for con in prob.constraints if con.isfixed], "\n")
end
#> END ##

#< POPULATION GENERATION METHODS ##
"""
    generate_population(constraints::ConstraintSet, n::Int)

Generate a population of `n` individuals for the given generic `constraints <: ConstraintSet`. Each individual is sampled from a log-uniform distribution within the valid range for each parameter or initial condition.
"""
function generate_population(constraints::CT, n::Int) where CT <: ConstraintSet
    # Preallocate the population array of arrays
    population = generate_empty_population(constraints, n)
    
    generate_population!(population, constraints)
end

"""Generate an empty population array of arrays, where the length of each individual is the number of constraints minus the fixed ones"""
function generate_empty_population(constraints::CT, n::Int) where CT <: ConstraintSet
    num_params = activelength(constraints)
    
    # Preallocate the population array of arrays
    [Vector{Float64}(undef, num_params) for _ in 1:n]
end

"""In-place population generation of already allocated empty population"""
function generate_population!(population::Vector{Vector{Float64}}, constraints::CT) where CT <: ConstraintSet

    rand_vals = Vector{Float64}(undef, length(population))
    
    # Populate the array
    i = 1
    for conrange in constraints
        if !conrange.isfixed
            min_val, max_val = log10(conrange.min), log10(conrange.max)
            rand_vals .= exp10.(rand(Uniform(min_val, max_val), length(population)))
            
            for j in eachindex(population)
                population[j][i] = rand_vals[j]
            end
            i += 1
        end
    end
    return population
end
#> END ##

function generate_matrix_population(constraints::CT, n::Int) where CT <: ConstraintSet
    population = generate_empty_matrix_population(constraints, n)
    generate_matrix_population!(population, constraints)
end

function generate_empty_matrix_population(constraints::CT, n::Int) where CT <: ConstraintSet
    num_params = activelength(constraints)
    Matrix{Float64}(undef, num_params, n)
end


function generate_matrix_population!(population::Matrix{Float64}, constraints::CT) where CT <: ConstraintSet
    num_individuals = size(population, 2)
    
    i = 1
    for conrange in constraints
        if !conrange.isfixed
            min_val, max_val = log10(conrange.min), log10(conrange.max)
            rand_vals = exp10.(rand(Uniform(min_val, max_val), num_individuals))
            
            population[i, :] = rand_vals
            i += 1
        end
    end
    return population
end


#< MISCELLANEOUS FUNCTIONS ##
"""
    logrange(start, stop, length::Int)

Defines logspace function for sampling parameters and initial conditions
    """
logrange(start, stop, length::Int) = exp10.(collect(range(start=log10(start), stop=log10(stop), length=length)))
#> END ##



#< GA RESULTS TYPE ##
"Struct to hold the results of a GA optimization"
struct GAResults{T <: VecOrMat} 
    # trace::Vector{Evolutionary.OptimizationTraceRecord}
    population::T
    fitvals::Vector{Float64}
    periods::Vector{Float64}
    amplitudes::Vector{Float64}
    gen_indices::Vector{Tuple{Int,Int}}
    # lineages::Vector{Vector{Int}}
    lineages::Matrix{Int}
    fixed_names::Vector{Symbol}
end

"""Constructor for a GAResults object, also stores the lineages of each individual"""
function GAResults(result::EvolutionaryOptimizationResults, constraintset::ConstraintSet) 
    numpoints = sum(length, (gen.metadata["fitvals"] for gen in result.trace))

    indlength = activelength(constraintset)
    # population = [Vector{Float64}(undef, indlength) for _ in 1:numpoints]
    population = Matrix{Float64}(undef, indlength, numpoints)
    fitvals = Vector{Float64}(undef, numpoints)
    periods = Vector{Float64}(undef, numpoints)
    amplitudes = Vector{Float64}(undef, numpoints)
    # lineages = Vector{Vector{Int}}(undef, numpoints)
    lineages = Matrix{Int}(undef, 2, numpoints)

    gen_indices = Tuple{Int, Int}[]

    startidx = 1
    for (i, gen) in enumerate(result.trace)
        endidx = startidx + size(gen.metadata["population"], 2) - 1

        push!(gen_indices, (startidx, endidx))
        # @info size(gen.metadata["population"])
        population[:, startidx:endidx] .= gen.metadata["population"]
        fitvals[startidx:endidx] .= gen.metadata["fitvals"]
        periods[startidx:endidx] .= gen.metadata["periods"]
        amplitudes[startidx:endidx] .= gen.metadata["amplitudes"]
        lineages[:, startidx:endidx] .= gen.metadata["lineages"]

        startidx = endidx + 1
    end

    fixed_names = get_fixed_names(constraintset)
    return GAResults(population, fitvals, periods, amplitudes, gen_indices, lineages, fixed_names)
end
#> END ##


#< RUN GENETIC ALGORITHM OPTIMIZATION ##
"""
    run_GA(ga_problem::GAProblem, population::Vector{Vector{Float64}} = generate_population(ga_problem.constraints, 10000); abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true)
    
Runs the genetic algorithm, returning the `GAResult` type.
"""
function run_GA(ga_problem::GP, population = generate_matrix_population(ga_problem.constraints, 10000); 
                abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true,
                mutationScalar = 0.5, mutation_range = fill(mutationScalar, activelength(ga_problem.constraints)), mutation_scheme = BGA(mutation_range, 2), mutationRate = 1.0,
                selection_method = tournament, num_tournament_groups=20, crossover = TPX, crossoverRate = 0.75,
                n_newInds = 0.0) where GP <: GAProblem


    #* Create constraints using the min and max values from constraints if they are active for optimization.
    boxconstraints = BoxConstraints([constraint.min for constraint in ga_problem.constraints if !constraint.isfixed], [constraint.max for constraint in ga_problem.constraints if !constraint.isfixed])

    # *Create Progress bar and callback function
    # ga_progress = Progress(threshold; desc = "GA Progress")
    # callback_func = (trace) -> ga_callback(trace, ga_progress, threshold)

    #* Define options for the GA.
    opts = Options(abstol=abstol, reltol=reltol, successive_f_tol = successive_f_tol, iterations=iterations, 
                        store_trace = true, show_trace=show_trace, show_every=1, parallelization=parallelization)#, callback=callback_func)

    #* Define the range of possible values for each parameter when mutated, and the mutation scalar.

    #? BGA mutation scheme
    # mutation_scalar = 0.5
    # mutation_range = fill(mutation_scalar, activelength(ga_problem.constraints))
    # mutation_scheme = BGA(mutation_range, 2)

    #? PM mutation scheme
    # lowerbound = [constraint.min/10 for constraint in ga_problem.constraints.ranges]
    # upperbound = [constraint.max*10 for constraint in ga_problem.constraints.ranges]
    # mutation_scheme = PM(lowerbound, upperbound, 2.)

    poplength = size(population, 2)


    #* Define the GA method.
    mthd = GA(populationSize = poplength, selection = selection_method(cld(poplength,num_tournament_groups), select=argmax),
                crossover = crossover, crossoverRate = crossoverRate, # Two-point crossover event
                mutation  = mutation_scheme, mutationRate = mutationRate, ɛ = n_newInds)

    #* Make fitness function
    fitness_function = make_fitness_function_threaded(ga_problem.constraints, ga_problem.ode_problem)
    # fitness_function = make_split_fitness_function(ga_problem.constraints, ga_problem.ode_problem)

    #* Run the optimization
    result = optimize(fitness_function, zeros(3), boxconstraints, mthd, population, opts)

    return GAResults(result, ga_problem.constraints)
end
#> END




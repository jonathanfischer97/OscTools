# """Fitness function constructor called during GAProblem construction that captures the fixed indices and ODE problem"""
# function make_fitness_function(constraints::ConstraintSet, ode_problem::OT, eval_function::FT) where {OT<:ODEProblem, FT<:Function}
#     fixed_idxs = get_fixed_indices(constraints)
#     fixed_values = [constraints[i].fixed_value for i in fixed_idxs]
#     n_fixed = length(fixed_idxs)
#     n_total = n_fixed + activelength(constraints) 

#     non_fixed_indices = setdiff(1:n_total, fixed_idxs)

#     # merged_input = Vector{Float64}(undef, n_total)
#     merged_input = zeros(Float64, n_total+12)
#     # @info "Merged input length: $(length(merged_input))"

#     merged_input[fixed_idxs] .= fixed_values  # Fill in fixed values

#     function fitness_function(input::Vector{Float64})
#         merged_input[non_fixed_indices] .= input  # Fill in variable values
#         # @info "Merged input: $merged_input"
#         return eval_function(merged_input, ode_problem)
#     end

#     return fitness_function
# end

# make_fitness_function(constraints::ParameterConstraints, ode_problem::ODEProblem) = make_fitness_function(constraints, ode_problem, eval_param_fitness)
# make_fitness_function(constraints::InitialConditionConstraints, ode_problem::ODEProblem) = make_fitness_function(constraints, ode_problem, eval_ic_fitness)
# make_fitness_function(constraints::AllConstraints, ode_problem::ODEProblem) = make_fitness_function(constraints, ode_problem, eval_all_fitness)

# """Returns in-place function"""
# function make_fitness_function_inplace(constraints::ConstraintSet, ode_problem::OT, eval_function::FT) where {OT<:ODEProblem, FT<:Function}
#     fixed_idxs = get_fixed_indices(constraints)
#     fixed_values = [constraints[i].fixed_value for i in fixed_idxs]
#     n_fixed = length(fixed_idxs)
#     n_total = n_fixed + activelength(constraints)

#     non_fixed_indices = setdiff(1:n_total, fixed_idxs)


#     # Create a ThreadLocal array
#     merged_inputs = [zeros(Float64, n_total+12) for _ in 1:Threads.nthreads()]
#     Fvs = [Vector{Float64}(undef,3) for _ in 1:Threads.nthreads()]

#     # Fill in the fixed values
#     for input in merged_inputs
#         input[fixed_idxs] .= fixed_values  # Fill in fixed values
#     end

#     function fitness_function!(input::Vector{Float64})
#         # Get the merged_input array for the current thread
#         merged_input = merged_inputs[Threads.threadid()]
#         merged_input[non_fixed_indices] .= input  # Fill in variable values
#         Fvs[Threads.threadid()] .= eval_function(merged_input, ode_problem)
#     end

#     return fitness_function!
# end

# make_fitness_function_inplace(constraints::ParameterConstraints, ode_problem::ODEProblem) = make_fitness_function_inplace(constraints, ode_problem, eval_param_fitness)
# make_fitness_function_inplace(constraints::InitialConditionConstraints, ode_problem::ODEProblem) = make_fitness_function_inplace(constraints, ode_problem, eval_ic_fitness)
# make_fitness_function_inplace(constraints::AllConstraints, ode_problem::ODEProblem) = make_fitness_function_inplace(constraints, ode_problem, eval_all_fitness)

#< FITNESS FUNCTION CONSTRUCTOR ##
"""Multithreaded fitness function, allocated a merged array for each thread"""
function make_fitness_function_threaded(constraints::ConstraintSet, ode_problem::OT, eval_function::FT) where {OT<:ODEProblem, FT<:Function}
    fixed_idxs = get_fixed_indices(constraints)
    fixed_values = [constraints[i].fixed_value for i in fixed_idxs]
    n_fixed = length(fixed_idxs)
    n_total = n_fixed + activelength(constraints) 

    non_fixed_indices = setdiff(1:n_total, fixed_idxs)

    # Create a ThreadLocal array
    merged_inputs = [zeros(Float64, n_total+12) for _ in 1:Threads.nthreads()]

    # Fill in the fixed values
    for input in merged_inputs
        input[fixed_idxs] .= fixed_values  # Fill in fixed values
    end

    function fitness_function(input::Vector{Float64})
        # Get the merged_input array for the current thread
        merged_input = merged_inputs[Threads.threadid()]
        merged_input[non_fixed_indices] .= input  # Fill in variable values

        return eval_function(merged_input, ode_problem)
    end

    return fitness_function
end

make_fitness_function_threaded(constraints::ParameterConstraints, ode_problem::ODEProblem) = make_fitness_function_threaded(constraints, ode_problem, eval_param_fitness)
make_fitness_function_threaded(constraints::InitialConditionConstraints, ode_problem::ODEProblem) = make_fitness_function_threaded(constraints, ode_problem, eval_ic_fitness)
make_fitness_function_threaded(constraints::AllConstraints, ode_problem::ODEProblem) = make_fitness_function_threaded(constraints, ode_problem, eval_all_fitness)
#> END


#< GA PROBLEM TYPE ##
"""
    GAProblem{T <: ConstraintSet}

Struct encapsulating a Genetic Algorithm (GA) optimization problem. It holds the constraints for the problem, the ODE problem to be solved.

# Fields
- `constraints::T`: Constraints for the problem. Either `ParameterConstraints` or `InitialConditionConstraints` or `AllConstraints`.
- `ode_problem::ODEProblem`: ODE problem to be solved.
"""
@kwdef mutable struct GAProblem{CT <: ConstraintSet, OT <: ODEProblem}
    constraints::CT = AllConstraints()
    ode_problem::OT = make_ODE_problem()
end

# show method overload for GAProblem
function Base.show(io::IO, ::MIME"text/plain", prob::GAProblem) 
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

# Example
```julia
constraints = ParameterConstraints()
population = generate_population(constraints, 100)
```
"""
function generate_population(constraints::ConstraintSet, n::Int)
    # Preallocate the population array of arrays
    population = generate_empty_population(constraints, n)
    
    generate_population!(population, constraints)
end

"""Generate an empty population array of arrays, where the length of each individual is the number of constraints minus the fixed ones"""
function generate_empty_population(constraints::ConstraintSet, n::Int)
    num_params = activelength(constraints)
    
    # Preallocate the population array of arrays
    [Vector{Float64}(undef, num_params) for _ in 1:n]
end

"""In-place population generation of already allocated empty population"""
function generate_population!(population::Vector{Vector{Float64}}, constraints::ConstraintSet)

    rand_vals = Vector{Float64}(undef, length(population))
    
    # Populate the array
    i = 1
    for conrange in constraints
        if !conrange.isfixed
            min_val, max_val = log10(conrange.min), log10(conrange.max)
            rand_vals .= exp10.(rand(Uniform(min_val, max_val), length(population)))
            
            for j in 1:length(population)
                population[j][i] = rand_vals[j]
            end
            i += 1
        end
    end
    return population
end
#> END ##

#< MISCELLANEOUS FUNCTIONS ##
"""Defines logspace function for sampling parameters and initial conditions"""
logrange(start, stop, length::Int) = exp10.(collect(range(start=log10(start), stop=log10(stop), length=length)))
#> END ##

# function generate_population_stratified!(population::Vector{Vector{Float64}}, constraints::ConstraintSet, n_strata::Int)

#     rand_vals = Vector{Float64}(undef, length(population))
    
#     # Populate the array
#     i = 1
#     for conrange in constraints
#         if !conrange.isfixed
#             min_val, max_val = log10(conrange.min), log10(conrange.max)
#             strata_bounds = range(min_val, stop=max_val, length=n_strata+1)
            
#             for stratum in 1:n_strata
#                 lower_bound, upper_bound = strata_bounds[stratum], strata_bounds[stratum+1]
#                 n_samples_stratum = length(population) ÷ n_strata  # Integer division to get an equal number of samples from each stratum
#                 n_samples_stratum += (stratum <= length(population) % n_strata)  # Distribute any remaining samples among the first few strata
                
#                 rand_vals[1:n_samples_stratum] .= exp10.(rand(Uniform(lower_bound, upper_bound), n_samples_stratum))
                
#                 for j in 1:n_samples_stratum
#                     population[(stratum-1)*n_samples_stratum+j][i] = rand_vals[j]
#                 end
#             end
#             i += 1
#         end
#     end
#     return population
# end


# function generate_population_stratified!(population::Vector{Vector{Float64}}, constraints::ConstraintSet, n_strata::Int)
#     # Pre-allocate rand_vals based on n_strata and population size
#     rand_vals = Vector{Float64}(undef, length(population) ÷ n_strata + 1)
    
#     i = 1
#     for conrange in constraints
#         if !conrange.isfixed
#             min_val, max_val = log10(conrange.min), log10(conrange.max)
#             strata_bounds = range(min_val, stop=max_val, length=n_strata+1)
            
#             for stratum in 1:n_strata
#                 lower_bound, upper_bound = strata_bounds[stratum], strata_bounds[stratum+1]
#                 n_samples_stratum = length(population) ÷ n_strata
#                 n_samples_stratum += (stratum <= length(population) % n_strata)
                
#                 rand_vals[1:n_samples_stratum] .= exp10.(rand(Uniform(lower_bound, upper_bound), n_samples_stratum))
                
#                 for j in 1:n_samples_stratum
#                     population[(stratum-1)*n_samples_stratum+j][i] = rand_vals[j]
#                 end
#             end
#             i += 1
#         end
#     end
#     return population
# end



# """### Callback function that terminates the GA if the number of oscillations exceeds the threshold, and updates the progress bar"""
# function ga_callback(trace::Evolutionary.OptimizationTrace, progressbar::Progress, threshold::Int)
#     #? Callback function for the GA, updating the progress bar
#     num_oscillation = trace[end].metadata["num_oscillatory"]
#     if num_oscillation >= threshold 
#         finish!(progressbar)
#         return true
#     else
#         next!(progressbar, step = num_oscillation)
#         return false
#     end
# end

#< GA RESULTS TYPE ##
"Struct to hold the results of a GA optimization"
struct GAResults 
    trace::Vector{Evolutionary.OptimizationTraceRecord}
    population::Vector{Vector{Float64}}
    fitvals::Vector{Float64}
    periods::Vector{Float64}
    amplitudes::Vector{Float64}
    gen_indices::Vector{Tuple{Int,Int}}
end

"""Constructor for a GAResults object, also stores the indices of each generation"""
function GAResults(result::Evolutionary.EvolutionaryOptimizationResults, indlength::Int)
    numpoints = sum(length, (gen.metadata["fitvals"] for gen in result.trace))
    population = [Vector{Float64}(undef, indlength) for _ in 1:numpoints]
    fitvals = Vector{Float64}(undef, numpoints)
    periods = Vector{Float64}(undef, numpoints)
    amplitudes = Vector{Float64}(undef, numpoints)

    gen_indices = Tuple{Int, Int}[]
    startidx = 1
    for gen in result.trace
        endidx = startidx + length(gen.metadata["population"]) - 1

        push!(gen_indices, (startidx, endidx))

        population[startidx:endidx] .= gen.metadata["population"]
  
        fitvals[startidx:endidx] .= gen.metadata["fitvals"]
     
        periods[startidx:endidx] .= gen.metadata["periods"]
    
        amplitudes[startidx:endidx] .= gen.metadata["amplitudes"]

        startidx = endidx + 1
    end
    return GAResults(result.trace, population, fitvals, periods, amplitudes, gen_indices)
end
#> END ##


#< RUN GENETIC ALGORITHM OPTIMIZATION ##
"""
Runs the genetic algorithm, returning the `GAResult` type.
"""
function run_GA(ga_problem::GAProblem, population::Vector{Vector{Float64}} = generate_population(ga_problem.constraints, 10000); 
                abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true)#, threshold=10000)
    # blas_threads = BLAS.get_num_threads()
    # BLAS.set_num_threads(1)

    population_size = length(population)

    #* Create constraints using the min and max values from constraints if they are active for optimization.
    boxconstraints = BoxConstraints([constraint.min for constraint in ga_problem.constraints if !constraint.isfixed], [constraint.max for constraint in ga_problem.constraints if !constraint.isfixed])

    # *Create Progress bar and callback function
    # ga_progress = Progress(threshold; desc = "GA Progress")
    # callback_func = (trace) -> ga_callback(trace, ga_progress, threshold)

    #* Define options for the GA.
    opts = Evolutionary.Options(abstol=abstol, reltol=reltol, successive_f_tol = successive_f_tol, iterations=iterations, 
                        store_trace = true, show_trace=show_trace, show_every=1, parallelization=parallelization)#, callback=callback_func)

    #* Define the range of possible values for each parameter when mutated, and the mutation scalar.

    #? BGA mutation scheme
    mutation_scalar = 0.5
    mutation_range = fill(mutation_scalar, activelength(ga_problem.constraints))
    mutation_scheme = BGA(mutation_range, 2)

    #? PM mutation scheme
    # lowerbound = [constraint.min/10 for constraint in ga_problem.constraints.ranges]
    # upperbound = [constraint.max*10 for constraint in ga_problem.constraints.ranges]
    # mutation_scheme = PM(lowerbound, upperbound, 2.)


    #* Define the GA method.
    mthd = GA(populationSize = population_size, selection = tournament(Int(population_size/10), select=argmax),
                crossover = TPX, crossoverRate = 1.0, # Two-point crossover event
                mutation  = mutation_scheme, mutationRate = 1.0)

    #* Make fitness function
    fitness_function = make_fitness_function_threaded(ga_problem.constraints, ga_problem.ode_problem)

    #* Run the optimization.
    result = Evolutionary.optimize(fitness_function, zeros(3,population_size), boxconstraints, mthd, population, opts)

    # BLAS.set_num_threads(blas_threads)
    # return result
    return GAResults(result, activelength(ga_problem.constraints))
end
#> END






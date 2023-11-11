"""
Non-dominated Sorting Genetic Algorithm (NSGA-II) for Multi-objective Optimization

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `crossoverRate`: The fraction of the population at the next generation, that is created by the crossover function
- `mutationRate`: Probability of chromosome to be mutated
- `selection`: [Selection](@ref) function (default: `tournament`)
- `crossover`: [Crossover](@ref) function (default: `SBX`)
- `mutation`: [Mutation](@ref) function (default: `PLM`)
- `metrics` is a collection of convergence metrics.
"""
struct NSGA2{T1,T2,T3} <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    selection::T1
    crossover::T2
    mutation::T3
    metrics::ConvergenceMetrics

    NSGA2(; populationSize::Int=50, crossoverRate::Float64=0.9, mutationRate::Float64=0.1,
        selection::T1 = tournament(2, select=twowaycomp),
        crossover::T2 = SBX(),
        mutation::T3 = PLM(),
        metrics = ConvergenceMetric[GD(), GD(true)]
       ) where {T1,T2,T3} =
            new{T1,T2,T3}(populationSize, crossoverRate, mutationRate, selection,
                          crossover, mutation, metrics)
end
population_size(method::NSGA2) = method.populationSize
default_options(method::NSGA2) = (iterations=1000,)
summary(m::NSGA2) = "NSGA-II[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate)]"
show(io::IO,m::NSGA2) = print(io, summary(m))

mutable struct NSGAState{T,IT} <: AbstractOptimizerState
    N::Int                      # population size
    fitness::AbstractMatrix{T}  # fitness of the fittest individuals
    fitpop::AbstractMatrix{T}   # fitness of the whole population (including offspring)
    fittest::AbstractVector{IT} # fittest individuals
    offspring::AbstractArray    # offspring cache
    population::AbstractArray   # combined population (parents + offspring)
    ranks::Vector{Int}          # individual ranks
    crowding::Vector{T}         # individual crowding distance
end
value(s::NSGAState) = s.fitness
minimizer(s::NSGAState) = s.fittest

"""Initialization of NSGA2 algorithm state"""
function initial_state(method::NSGA2, options, objfun, parents)

    v = value(objfun) # objective function value
    T = eltype(v)     # objective function value type
    d = length(v)     # objective function value dimension
    N = length(first(parents)) # parents dimension
    IT = eltype(parents)       # individual type
    offspring = similar(parents) # offspring cache

    # construct fitness array that covers total population,
    # i.e. parents + offspring
    fitpop = fill(typemax(T), d, method.populationSize*2)

    # Evaluate parents fitness
    value!(objfun, fitpop, parents)

    # setup initial state
    allpop = StackView(parents, offspring)
    ranks = vcat(fill(1, method.populationSize), fill(2, method.populationSize))
    crowding = vcat(fill(zero(T), method.populationSize), fill(typemax(T), method.populationSize))
    return NSGAState(N, zeros(T,d,0), fitpop, IT[], offspring, allpop, ranks, crowding)
end

function update_state!(objfun, constraints, state, parents::AbstractVector{IT}, method::NSGA2, options, itr) where {IT}
    populationSize = method.populationSize
    rng = options.rng

    # select offspring
    specFit = StackView(state.ranks, state.crowding, dims=1)
    selected = method.selection(view(specFit,:,1:populationSize), populationSize; rng=rng)

    # perform mating
    recombine!(state.offspring, parents, selected, method)

    # perform mutation
    mutate!(state.offspring, method, constraints, rng=rng)

    # calculate fitness of the offspring
    offfit = @view state.fitpop[:, populationSize+1:end]
    evaluate!(objfun, offfit, state.offspring, constraints)

    # calculate ranks & crowding for population
    F = nondominatedsort!(state.ranks, state.fitpop)
    crowding_distance!(state.crowding, state.fitpop, F)

    # select best individuals
    fitidx = Int[]
    for f in F
        if length(fitidx) + length(f) > populationSize
            idxs = sortperm(view(state.crowding,f))
            append!(fitidx, idxs[1:(populationSize-length(fitidx))])
            break
        else
            append!(fitidx, f)
        end
    end
    # designate the first Pareto front individuals as the fittest
    fidx = length(F[1]) > populationSize ? fitidx : F[1]
    state.fittest = state.population[fidx]
    # and keep their fitness
    state.fitness = state.fitpop[:,fidx]

    # construct new parent population
    parents .= state.population[fitidx]

    return false
end


"""Trace override function"""
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population::Vector{Vector{Float64}}, method::NSGA2, options) 
    # oscillatory_population_idxs = findall(fit -> fit > 0.0, state.fitvals) #find the indices of the oscillatory individuals
    oscillatory_population_idxs = findall(period -> period > 0.0, state.periods) #find the indices of the oscillatory individuals

    record["population"] = deepcopy(population[oscillatory_population_idxs])
    # valarray = copy(view(state.valarray,:,oscillatory_population_idxs))
    # record["fitvals"] = valarray[1,:]
    # record["periods"] = valarray[2,:]
    # record["amplitudes"] = valarray[3,:]
    record["fitvals"] = state.fitvals[oscillatory_population_idxs, 1]
    record["periods"] = state.periods[oscillatory_population_idxs, 2]
    record["amplitudes"] = state.amplitudes[oscillatory_population_idxs, 3]
end

"""
    NSGAProblem{T <: ConstraintSet}

Struct encapsulating a NSGA optimization problem. It holds the constraints for the problem, the ODE problem to be solved.

# Fields
- `constraints::T`: Constraints for the problem. Either `ParameterConstraints` or `InitialConditionConstraints` or `AllConstraints`.
- `ode_problem::ODEProblem`: ODE problem to be solved.
"""
@kwdef mutable struct NSGAProblem{CT <: ConstraintSet, OP <: ODEProblem}
    constraints::CT = AllConstraints()
    ode_problem::OP = make_ODE_problem()
end

# show method overload for DEProblem
function Base.show(io::IO, ::MIME"text/plain", prob::NSGAProblem) 
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

#< NSGA RESULTS TYPE ##
"Struct to hold the results of a NSGA optimization"
struct NSGAResults 
    population::Vector{Vector{Float64}}
    fitvals::Vector{Float64}
    periods::Vector{Float64}
    amplitudes::Vector{Float64}
    gen_indices::Vector{Tuple{Int,Int}}
    fixed_names::Vector{Symbol}
end

"""Constructor for a NPResults object, also stores the indices of each generation"""
function NSGAResults(result::Evolutionary.EvolutionaryOptimizationResults, constraintset::ConstraintSet) 
    numpoints = sum(length, (gen.metadata["fitvals"] for gen in result.trace))

    indlength = activelength(constraintset)
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

    fixed_names = get_fixed_names(constraintset)
    return NSGAResults(population, fitvals, periods, amplitudes, gen_indices, fixed_names)
end
#> END ##



#< RUN NSGA OPTIMIZATION ##
"""
    run_NSGA(nsga_problem::NP, population::Vector{Vector{Float64}} = generate_population(de_problem.constraints, 10000); abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true)
    
Runs the NSGA algorithm, returning the `NSGAResult` type.
"""
function run_NSGA(nsga_problem::NP, population::Vector{Vector{Float64}} = generate_population(nsga_problem.constraints, 10000); 
                abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true,
                F = 0.9,recombination=BINX(0.5), K = 0.5) where NP <: NSGAProblem

    #* Create constraints using the min and max values from constraints if they are active for optimization.
    boxconstraints = BoxConstraints([constraint.min for constraint in de_problem.constraints if !constraint.isfixed], [constraint.max for constraint in de_problem.constraints if !constraint.isfixed])

    #* Define options for the DE.
    opts = Evolutionary.Options(abstol=abstol, reltol=reltol, successive_f_tol = successive_f_tol, iterations=iterations, 
                                store_trace = true, show_trace=show_trace, show_every=1, parallelization=parallelization)

    #* Define the DE method.
    mthd = DE(populationSize = length(population), F = F, recombination=recombination, K = K)

    #* Make fitness function.
    fitness_function = make_fitness_function_threaded(de_problem.constraints, de_problem.ode_problem)

    #* Run the optimization.
    result = Evolutionary.optimize(fitness_function, zeros(3), boxconstraints, mthd, population, opts)

    return DifferentialEvolutionResults(result, de_problem.constraints)
end
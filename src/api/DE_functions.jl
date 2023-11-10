#< DE PROBLEM TYPE ##
"""
    DEProblem{T <: ConstraintSet}

Struct encapsulating a Differential Evolution (DE) optimization problem. It holds the constraints for the problem, the ODE problem to be solved.

# Fields
- `constraints::T`: Constraints for the problem. Either `ParameterConstraints` or `InitialConditionConstraints` or `AllConstraints`.
- `ode_problem::ODEProblem`: ODE problem to be solved.
"""
@kwdef mutable struct DifferentialEvolutionProblem{CT <: ConstraintSet, OP <: ODEProblem}
    constraints::CT = AllConstraints()
    ode_problem::OP = make_ODE_problem()
end

"""Trace override function"""
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population::Vector{Vector{Float64}}, method::DE, options) 
    # oscillatory_population_idxs = findall(fit -> fit > 0.0, state.fitvals) #find the indices of the oscillatory individuals
    oscillatory_population_idxs = findall(period -> period > 0.0, state.periods) #find the indices of the oscillatory individuals

    record["population"] = deepcopy(population[oscillatory_population_idxs])
    # valarray = copy(view(state.valarray,:,oscillatory_population_idxs))
    # record["fitvals"] = valarray[1,:]
    # record["periods"] = valarray[2,:]
    # record["amplitudes"] = valarray[3,:]
    record["fitvals"] = state.fitvals[oscillatory_population_idxs]
    record["periods"] = state.periods[oscillatory_population_idxs]
    record["amplitudes"] = state.amplitudes[oscillatory_population_idxs]
end

# """Testing trace override function. Saves all solutions"""
# function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population::Vector{Vector{Float64}}, method::DE, options) 
#     record["oscillatory_idxs"] = findall(period -> period > 0.0, state.periods) #find the indices of the oscillatory individuals

#     record["population"] = deepcopy(population)

#     record["fitvals"] = state.fitvals
#     record["periods"] = state.periods
#     record["amplitudes"] = state.amplitudes
# end

# show method overload for DEProblem
function Base.show(io::IO, ::MIME"text/plain", prob::DifferentialEvolutionProblem) 
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



#< Differential Evolution RESULTS TYPE ##
"Struct to hold the results of a Differential Evolution optimization"
struct DifferentialEvolutionResults 
    population::Vector{Vector{Float64}}
    fitvals::Vector{Float64}
    periods::Vector{Float64}
    amplitudes::Vector{Float64}
    gen_indices::Vector{Tuple{Int,Int}}
    fixed_names::Vector{Symbol}
end

"""Constructor for a DifferentialEvolutionResults object, also stores the indices of each generation"""
function DifferentialEvolutionResults(result::Evolutionary.EvolutionaryOptimizationResults, constraintset::ConstraintSet) 
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
    return DifferentialEvolutionResults(population, fitvals, periods, amplitudes, gen_indices, fixed_names)
end
#> END ##



#< RUN DIFFERENTIAL EVOLUTION OPTIMIZATION ##
"""
    run_DE(de_problem::DifferentialEvolutionProblem, population::Vector{Vector{Float64}} = generate_population(de_problem.constraints, 10000); abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true)
    
Runs the differential evolution algorithm, returning the `DifferentialEvolutionResult` type.
"""
function run_DE(de_problem::DP, population::Vector{Vector{Float64}} = generate_population(de_problem.constraints, 10000); 
                abstol=1e-4, reltol=1e-2, successive_f_tol = 4, iterations=5, parallelization = :thread, show_trace=true,
                F = 0.9,recombination=BINX(0.5), K = 0.5) where DP <: DifferentialEvolutionProblem

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

#> END


#< DE Overloads
mutable struct CustomDEState{T,IT} <: Evolutionary.AbstractOptimizerState
    N::Int
    fitvals::Vector{T}
    periods::Vector{T} #* periods of the individuals
    amplitudes::Vector{T} #* amplitudes of the individuals
    fittest::IT
end
Evolutionary.value(s::CustomDEState) = minimum(s.fitvals)
Evolutionary.minimizer(s::CustomDEState) = s.fittest


"""Initialization of DE algorithm state"""
function Evolutionary.initial_state(method::DE, options, objfun, population::Vector{Vector{Float64}})
    # T = typeof(value(objfun))
    T = Float64
    individual = first(population)
    N = length(individual)
    fitvals = fill(maxintfloat(T), method.populationSize)
    periods = fill(maxintfloat(T), method.populationSize)
    amplitudes = fill(maxintfloat(T), method.populationSize)

    # setup initial state
    return CustomDEState(N, fitvals, periods, amplitudes, copy(individual))
end

function Evolutionary.update_state!(objfun, constraints, state, population::Vector{Vector{Float64}}, method::DE, options, itr)

    # setup
    Np = method.populationSize
    n = method.n
    F = method.F
    rng = options.rng

    offspring = similar(population)

    # select base vectors
    bases = method.selection(state.fitvals, Np)

    # select target vectors
    for (i,b) in enumerate(bases)
        # mutation
        base = population[b]
        offspring[i] = copy(base)
        # println("$i => base:", offspring[i])

        targets = Evolutionary.randexcl(rng, 1:Np, [i], 2*n)
        offspring[i] = Evolutionary.differentiation(offspring[i], @view population[targets]; F=F)
        # println("$i => mutated:", offspring[i], ", targets:", targets)

        # recombination
        offspring[i], _ = method.recombination(offspring[i], base, rng=rng)
        # println("$i => recombined:", offspring[i])
    end

    # Create new generation
    fitidx = 0
    minfit = Inf
    for i in 1:Np
        o = Evolutionary.apply!(constraints, offspring[i])
        # @info "Objective function: $objfun"
        # @info "Offspring: $o"
        v, p, a = Evolutionary.value(objfun, o) #+ Evolutionary.penalty(constraints, o)
        if (v <= state.fitvals[i])
            population[i] = o
            state.fitvals[i] = v
            state.periods[i] = p
            state.amplitudes[i] = a
            if v < minfit
                minfit = v
                fitidx = i
            end
        end
    end

    # set best individual
    if fitidx > 0
        state.fittest = population[fitidx]
    end

    return false
end
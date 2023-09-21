#< CUSTOM GA STATE TYPE AND BASE OVERLOADS ##
"""Custom GA state type that captures additional data from the objective function in `fitvals`\n
    - `N` is the number of elements in an individual\n
    - `eliteSize` is the number of individuals that are copied to the next generation\n
    - `fittestValue` is the fitness of the fittest individual\n
    - `fitvals` is a Matrix of the fitness, period, and amplitude of the population\n
    - `fittestInd` is the fittest individual\n"""
mutable struct CustomGAState <: Evolutionary.AbstractOptimizerState  
    N::Int  #* number of elements in an individual
    eliteSize::Int  #* number of individuals that are copied to the next generation
    fittestValue::Float64  #* fitness of the fittest individual
    # fitvals::Vector{Float64}  #* fitness values of the population
    fitvals::Matrix{Float64}
    fittestInd::Vector{Float64}  #* fittest individual
    # periods::Vector{Float64} #* periods of the individuals
    # amplitudes::Vector{Float64} #* amplitudes of the individuals
end  
Evolutionary.value(s::CustomGAState) = s.fittestValue #return the fitness of the fittest individual
Evolutionary.minimizer(s::CustomGAState) = s.fittestInd #return the fittest individual


"""Trace override function"""
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population::Vector{Vector{Float64}}, method::GA, options) 
    oscillatory_population_idxs = findall(fit -> fit > 0.0, view(state.fitvals, 1, :)) #find the indices of the oscillatory individuals

    record["population"] = deepcopy(population[oscillatory_population_idxs])
    fitvals = copy(view(state.fitvals,:,oscillatory_population_idxs))
    record["fitvals"] = fitvals[1,:]
    record["periods"] = fitvals[2,:]
    record["amplitudes"] = fitvals[3,:]
end

"""Show override function to prevent printing large arrays"""
function Evolutionary.show(io::IO, t::Evolutionary.OptimizationTraceRecord{Float64, O}) where O <: Evolutionary.AbstractOptimizer
    print(io, lpad("$(t.iteration)",6))
    print(io, "   ")
    print(io, lpad("$(t.value)",14))
    for (key, value) in t.metadata
        if !isa(value, AbstractArray)
            print(io, "\n * $key: $value")
        end
    end
    print(io, "\n * num_oscillatory: $(length(t.metadata["fitvals"]))")
    return
end
#> END OF CUSTOM GA STATE TYPE AND OVERLOADS ##



#< CUSTOM GA STATE CONSTRUCTOR AND UPDATE STATE FUNCTION ##
"""Initialization of my custom GA algorithm state that captures additional data from the objective function\n
    - `method` is the GA method\n
    - `options` is the options dictionary\n
    - `objfun` is the objective function\n
    - `population` is the initial population, specifically a Vector for dispatch\n
    - `extradata` is the additional data from the objective function\n
    - `fittest` is the fittest individual\n"""
function Evolutionary.initial_state(method::GA, options, objfun, population::Vector{Vector{Float64}})

    N = length(first(population))
    # fitvals = zeros(Float64, method.populationSize)
    
    # periods = zeros(Float64, method.populationSize)
    # amplitudes = zeros(Float64, method.populationSize)
    # @info "Initializing GA state"
    output_array = zeros(Float64, 3, method.populationSize)

    #* setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)

    #* Evaluate population fitness, period and amplitude
    # Evolutionary.value!(objfun, fitvals, population, periods, amplitudes)
    Evolutionary.value!(objfun, output_array, population)


    # maxfit, fitidx = findmax(fitvals)
    maxfit, fitidx = findmax(view(output_array,1,:))

    #* setup initial state
    # return CustomGAState(N, eliteSize, maxfit, fitvals, copy(population[fitidx]), periods, amplitudes)
    return CustomGAState(N, eliteSize, maxfit, output_array, copy(population[fitidx]))
end

function Evolutionary.evaluate!(objfun, fitvals, population::Vector{Vector{Float64}}, constraints)

    #* calculate fitness of the population
    Evolutionary.value!(objfun, fitvals, population)

    #* apply penalty to fitness
    Evolutionary.penalty!(fitvals, constraints, population)
end

"""Update state function that captures additional data from the objective function"""
function Evolutionary.update_state!(objfun, constraints, state::CustomGAState, parents::Vector{Vector{Float64}}, method::GA, options, itr)
    populationSize = method.populationSize
    # evaltype = options.parallelization
    rng = options.rng
    offspring = similar(parents)

    fitness_vals = view(state.fitvals, 1, :)

    #* select offspring
    selected = method.selection(fitness_vals, populationSize, rng=rng)

    #* perform mating
    offspringSize = populationSize - state.eliteSize
    Evolutionary.recombine!(offspring, parents, selected, method, offspringSize, rng=rng)

    #* Elitism (copy population individuals before they pass to the offspring & get mutated)
    fitidxs = sortperm(fitness_vals)
    for i in 1:state.eliteSize
        subs = offspringSize+i
        offspring[subs] = copy(parents[fitidxs[i]])
    end

    #* perform mutation
    Evolutionary.mutate!(offspring, method, constraints, rng=rng)

    #* calculate fitness and extradata of the population
    # Evolutionary.evaluate!(objfun, state.fitvals, offspring, state.periods, state.amplitudes, constraints)
    Evolutionary.evaluate!(objfun, state.fitvals, offspring, constraints)


    #* select the best individual
    _, fitidx = findmax(fitness_vals)
    state.fittestInd = offspring[fitidx]
    state.fittestValue = fitness_vals[fitidx]
    
    #* replace population
    parents .= offspring

    return false
end
#> END OF CUSTOM GA STATE CONSTRUCTOR AND UPDATE STATE FUNCTION ##




#< OBJECTIVE FUNCTION OVERRIDES ##
"""
    EvolutionaryObjective(f, x[, F])

Constructor for an objective function object around the function `f` with initial parameter `x`, and objective value `F`.
"""
function Evolutionary.EvolutionaryObjective(f::TC, x::Vector{Float64}, F::AbstractMatrix;
                               eval::Symbol = :serial) where {TC}
    # @info "Using custom EvolutionaryObjective constructor"
    defval = Evolutionary.default_values(x)

    #* convert function into the in-place one
    TF = typeof(F)

    fn = (Fv,xv) -> (Fv .= f(xv))
    TN = typeof(fn)
    EvolutionaryObjective{TN,TF,typeof(x),Val{eval}}(fn, F, defval, 0)
end

"""Override of the multiobjective check"""
Evolutionary.ismultiobjective(obj::EvolutionaryObjective{Function,Matrix{Float64},Vector{Float64},Val{:thread}}) = false

"""Modified value! function from Evolutionary.jl to allow for multiple outputs from the objective function to be stored"""
function Evolutionary.value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:thread}},
                                F::AbstractMatrix, xs::Vector{TX}) where {TC,TF,TX <: AbstractVector}
    n = length(xs)
    # @info "Evaluating $(n) individuals in parallel"
    Threads.@threads for i in 1:n
        # @info length(xs[i])
        # F[:,i] .= Evolutionary.value(obj, xs[i])  #* evaluate the fitness, period, and amplitude for each individual
        fv = view(F, :, i)
        # @info length(fv)
        value(obj, fv, xs[i])
    end
    F
end

"""Same value! function but with serial eval"""
function Evolutionary.value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:serial}},
                                F::AbstractMatrix, xs::Vector{TX}) where {TC,TF, TX <: AbstractVector}
    n = length(xs)
    for i in 1:n
        F[:,i] .= Evolutionary.value(obj, xs[i])  #* evaluate the fitness, period, and amplitude for each individual
        # println("Ind: $(xs[i]) fit: $(F[i]) per: $(P[i]) amp: $(A[i])")
    end
    F
end
#> END OF OVERRIDES ##



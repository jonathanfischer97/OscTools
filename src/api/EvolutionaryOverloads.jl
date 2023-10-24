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
    fittestInd::Vector{Float64}  #* fittest individual
    fitvals::Vector{Float64}  #* fitness values of the population
    periods::Vector{Float64} #* periods of the individuals
    amplitudes::Vector{Float64} #* amplitudes of the individuals
    # valarray::Matrix{Float64}
end  
Evolutionary.value(s::CustomGAState) = s.fittestValue #return the fitness of the fittest individual
Evolutionary.minimizer(s::CustomGAState) = s.fittestInd #return the fittest individual


"""Trace override function"""
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population::Vector{Vector{Float64}}, method::GA, options) 
    oscillatory_population_idxs = findall(fit -> fit > 0.0, state.fitvals) #find the indices of the oscillatory individuals

    record["population"] = deepcopy(population[oscillatory_population_idxs])
    # valarray = copy(view(state.valarray,:,oscillatory_population_idxs))
    # record["fitvals"] = valarray[1,:]
    # record["periods"] = valarray[2,:]
    # record["amplitudes"] = valarray[3,:]
    record["fitvals"] = state.fitvals[oscillatory_population_idxs]
    record["periods"] = state.periods[oscillatory_population_idxs]
    record["amplitudes"] = state.amplitudes[oscillatory_population_idxs]
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
    # output_array = zeros(Float64, 3, method.populationSize)


    fitvals = zeros(Float64, method.populationSize)
    periods = similar(fitvals)
    amplitudes = similar(fitvals)
    

    # @info "Initializing GA state"

    #* setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)

    #* Evaluate population fitness, period and amplitude
    Evolutionary.value!(objfun, fitvals, periods, amplitudes, population)
    # Evolutionary.value!(objfun, output_array, population)


    maxfit, fitidx = findmax(fitvals)
    # maxfit, fitidx = findmax(output_array[1,:])

    #* setup initial state
    return CustomGAState(N, eliteSize, maxfit, copy(population[fitidx]), fitvals, periods, amplitudes)
    # return CustomGAState(N, eliteSize, maxfit, copy(population[fitidx]), output_array)
end

function Evolutionary.evaluate!(objfun, population::Vector{Vector{Float64}}, fitvals, periods, amplitudes)

    #* calculate fitness of the population
    Evolutionary.value!(objfun, fitvals, periods, amplitudes, population)
    # Evolutionary.value!(objfun, output_array, population)

    #* apply penalty to fitness
    # Evolutionary.penalty!(fitvals, constraints, population)
end

"""Update state function that captures additional data from the objective function"""
function Evolutionary.update_state!(objfun, constraints, state::CustomGAState, parents::Vector{Vector{Float64}}, method::GA, options, itr)
    populationSize = method.populationSize
    # evaltype = options.parallelization
    rng = options.rng
    offspring = similar(parents)

    # fitness_vals = view(state.fitvals, 1, :)

    #* select offspring
    selected = method.selection(state.fitvals, populationSize, rng=rng)
    # selected = method.selection(state.valarray[1,:], populationSize, rng=rng)

    #* perform mating
    offspringSize = populationSize - state.eliteSize
    Evolutionary.recombine!(offspring, parents, selected, method, offspringSize, rng=rng)

    #* Elitism (copy population individuals before they pass to the offspring & get mutated)
    fitidxs = sortperm(state.fitvals)
    # fitidxs = sortperm(state.valarray[1,:])
    for i in 1:state.eliteSize
        subs = offspringSize+i
        offspring[subs] = copy(parents[fitidxs[i]])
    end

    #* perform mutation
    Evolutionary.mutate!(offspring, method, constraints, rng=rng)

    #* calculate fitness and extradata of the population
    Evolutionary.evaluate!(objfun, offspring, state.fitvals, state.periods, state.amplitudes)
    # Evolutionary.evaluate!(objfun, offspring, state.valarray)


    #* select the best individual
    _, fitidx = findmax(state.fitvals)
    # _, fitidx = findmax(state.valarray[1,:])
    state.fittestInd = offspring[fitidx]
    state.fittestValue = state.fitvals[fitidx]
    # state.fittestValue = state.valarray[1,fitidx]
    
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
function Evolutionary.EvolutionaryObjective(f::TC, x::Vector{Float64}, F::Vector{Float64};
                               eval::Symbol = :serial) where {TC}
    # @info "Using custom EvolutionaryObjective constructor"
    defval = Evolutionary.default_values(x)

    #* convert function into the in-place one
    TF = typeof(F)

    fn = (Fv,xv) -> (Fv .= f(xv))
    TN = typeof(fn)

    # fn, TN = if F isa AbstractMatrix
    #     ff = (Fv,xv) -> (Fv .= f(xv))
    #     @info "in-place conversion"
    #     ff, typeof(ff)
    # else
    #     @info "no in-place conversion"
    #     f, TC
    # end
    EvolutionaryObjective{TN,TF,typeof(x),Val{eval}}(fn, F, defval, 0)
end

"""Override of the multiobjective check"""
Evolutionary.ismultiobjective(obj::EvolutionaryObjective{Function, AbstractArray, Vector{Float64}, Val{:thread}}) = false

"""Modified value! function from Evolutionary.jl to allow for multiple outputs from the objective function to be stored"""
function Evolutionary.value!(obj::EvolutionaryObjective{TC, Vector{Float64}, Vector{Float64},Val{:thread}},
                                F::Vector{Float64}, P::Vector{Float64}, A::Vector{Float64}, xs::Vector{Vector{Float64}}) where {TC}
    n = length(xs)
    # @info "Evaluating F, P, A separately"
    # @info "F type: $(typeof(F))"
    Threads.@threads for i in 1:n
        # @info length(xs[i])
        F[i], P[i], A[i] = Evolutionary.value(obj, xs[i])  #* evaluate the fitness, period, and amplitude for each individual
    end
end

function Evolutionary.value!(obj::EvolutionaryObjective{TC, Vector{Float64}, Vector{Float64}, Val{:thread}},
                                F::Matrix{Float64}, xs::Vector{Vector{Float64}}) where {TC}
    n = length(xs)
    # @info "Evaluating $(n) individuals in parallel"
    # @info size(F)
    @info "Evaluating F as single matrix"
    @info "F type: $(typeof(F))"
    Threads.@threads for i in 1:n
        fv = view(F, :, i)
        value(obj, fv, xs[i])
    end
end

"""Same value! function but with SERIAL eval"""
function Evolutionary.value!(obj::EvolutionaryObjective{TC,TF,Vector{Float64},Val{:serial}},
                                F::Matrix{Float64}, xs::Vector{Vector{Float64}}) where {TC,TF}
    n = length(xs)
    for i in 1:n
        # F[:,i] .= Evolutionary.value(obj, xs[i])  #* evaluate the fitness, period, and amplitude for each individual
        # println("Ind: $(xs[i]) fit: $(F[i]) per: $(P[i]) amp: $(A[i])")
        fv = view(F, :, i)
        value(obj, fv, xs[i])
    end
    F
end
#> END OF OVERRIDES ##



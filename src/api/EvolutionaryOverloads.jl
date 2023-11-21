#< CUSTOM GA STATE TYPE AND BASE OVERLOADS ##
"""Custom GA state type that captures additional data from the objective function in `fitvals`\n
    - `N` is the number of elements in an individual\n
    - `eliteSize` is the number of individuals that are copied to the next generation\n
    - `fittestValue` is the fitness of the fittest individual\n
    - `fitvals` is a Matrix of the fitness, period, and amplitude of the population\n
    - `fittestInd` is the fittest individual\n"""
mutable struct CustomGAState{VT} <: AbstractOptimizerState where VT <: AbstractArray 
    N::Int  #* number of elements in an individual
    n_newInds::Int  #* number of newly generated individuals per generation
    fittestValue::Float64  #* fitness of the fittest individual
    fittestInd::Vector{Float64}  #* fittest individual
    valarray::VT
    # fitvals::VT  #* fitness values of the population
    # periods::VT #* periods of the individuals
    # amplitudes::VT #* amplitudes of the individuals

    # new field for lineage tracking
    lineages::Matrix{Int} # Matrix, 2 x populationSize, where each row is a parent lineage of the individual
    previous_saved_inds::BitVector # BitVector to track which individuals have been saved last generation
end  
value(s::CustomGAState) = s.fittestValue #return the fitness of the fittest individual
minimizer(s::CustomGAState) = s.fittestInd #return the fittest individual


"""Trace override function"""
function trace!(record::Dict{String,Any}, objfun, state::CustomGAState, population::Vector{Vector{Float64}}, method::GA, options) 
    # @info "Previous saved inds: $(findall(state.previous_saved_inds))"

    oscillatory_population_idxs = findall(period -> period > 0.0, state.periods) #find the indices of the oscillatory individuals

    record["oscillatory_idxs"] = oscillatory_population_idxs

    record["population"] = deepcopy(population[oscillatory_population_idxs])
    record["fitvals"] = state.fitvals[oscillatory_population_idxs]
    record["periods"] = state.periods[oscillatory_population_idxs]
    record["amplitudes"] = state.amplitudes[oscillatory_population_idxs]

    # Adjust lineage for oscillatory individuals using previous generation's saved indices
    # adjusted_lineage = Vector{Vector{Int}}(undef, length(oscillatory_population_idxs))
    # for (i, idx) in enumerate(oscillatory_population_idxs)
    #     original_lineage = state.lineages[idx]
    #     adjusted_lineage[i] = [adjust_parent_index(parent, state.previous_saved_inds) for parent in original_lineage]
    # end
    adjusted_lineage = Matrix{Int}(undef, 2, length(oscillatory_population_idxs))
    for (i, idx) in enumerate(oscillatory_population_idxs)
        original_lineage = state.lineages[:,idx]
        adjusted_lineage[:,i] .= [adjust_parent_index(parent, state.previous_saved_inds) for parent in original_lineage]
    end

    record["lineages"] = adjusted_lineage
    # @info size(state.lineages[:, oscillatory_population_idxs])
    # @info "Lineage: $(state.lineages[:, oscillatory_population_idxs])"
    # @info "Adjusted lineage: $(adjusted_lineage)"

    # @info "Previous saved inds: $(findall(state.previous_saved_inds))"

    # Update previous saved indices
    state.previous_saved_inds .= falses(length(state.previous_saved_inds))
    state.previous_saved_inds[oscillatory_population_idxs] .= true
end

"""Trace override function"""
function trace!(record::Dict{String,Any}, objfun, state::CustomGAState, population::Matrix{Float64}, method::GA, options) 

    oscillatory_population_idxs = findall(period -> period > 0.0, view(state.valarray, 2, :)) #find the indices of the oscillatory individuals

    oscillatory_valarray = state.valarray[:, oscillatory_population_idxs]

    record["oscillatory_idxs"] = oscillatory_population_idxs

    record["population"] = deepcopy(population[:,oscillatory_population_idxs])
    record["fitvals"] = oscillatory_valarray[1,:]
    record["periods"] = oscillatory_valarray[2,:]
    record["amplitudes"] = oscillatory_valarray[3,:]

    # oscillatory_population_idxs = findall(period -> period > 0.0, state.periods) #find the indices of the oscillatory individuals

    # record["oscillatory_idxs"] = oscillatory_population_idxs

    # record["population"] = deepcopy(population[oscillatory_population_idxs])
    # record["fitvals"] = state.fitvals[oscillatory_population_idxs]
    # record["periods"] = state.periods[oscillatory_population_idxs]
    # record["amplitudes"] = state.amplitudes[oscillatory_population_idxs]

    adjusted_lineage = Matrix{Int}(undef, 2, length(oscillatory_population_idxs))
    for (i, idx) in enumerate(oscillatory_population_idxs)
        original_lineage = state.lineages[:,idx]
        adjusted_lineage[:,i] .= [adjust_parent_index(parent, state.previous_saved_inds) for parent in original_lineage]
    end

    record["lineages"] = adjusted_lineage


    # Update previous saved indices
    state.previous_saved_inds .= falses(length(state.previous_saved_inds))
    state.previous_saved_inds[oscillatory_population_idxs] .= true
end

function adjust_parent_index(parent_idx::Int, saved_inds::BitVector)
    # Sum over a view of the BitVector up to the parent index
    saved_count = sum(view(saved_inds, 1:parent_idx))
    return saved_count == 0 ? -1 : saved_count  # Return -1 if parent was not saved
end



# """Trace function for saving all individuals"""
# """Testing trace override function. Saves all solutions"""
# function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population::Vector{Vector{Float64}}, method::GA, options) 
#     record["oscillatory_idxs"] = findall(period -> period > 0.0, state.periods) #find the indices of the oscillatory individuals

#     record["population"] = deepcopy(population)

#     record["fitvals"] = state.fitvals
#     record["periods"] = state.periods
#     record["amplitudes"] = state.amplitudes
# end

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
function initial_state(method::GA, options, objfun, population::Matrix{Float64})

    # N = length(first(population))
    N = size(population, 1)

    valarray = zeros(Float64, (3, method.populationSize))
    fitvals = @view valarray[1,:]
    # periods = @view valarray[2,:]
    # amplitudes = @view valarray[3,:]
    # fitvals = zeros(Float64, method.populationSize)
    # periods = similar(fitvals)
    # amplitudes = similar(fitvals)
    

    #* setup state values
    n_newInds = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize) #determine number of individuals to newly generate each generation

    #* Evaluate population fitness, period and amplitude
    value!(objfun, valarray, population)
    # value!(objfun, fitvals, periods, amplitudes, population)


    maxfit, fitidx = vfindmax(fitvals)
    # @info "Max fitness: $(maxfit)"
    # @info "Min fitness: $(findmin(fitvals)[1])"

    #* Initialize lineage array
    # lineages = fill([1,1], method.populationSize)
    lineages = fill(1, (2, method.populationSize))

    #* Initialize saved_inds BitVector
    previous_saved_inds = falses(method.populationSize)

    #* setup initial state
    return CustomGAState(N, n_newInds, maxfit, copy(population[:,fitidx]), valarray, lineages, previous_saved_inds)
end






"""Update state function that captures additional data from the objective function"""
function update_state!(objfun, constraints, state::CustomGAState, parents::Matrix{Float64}, method::GA, options, itr)
    populationSize = method.populationSize
    rng = options.rng
    # offspring = similar(parents)
    offspring = copy(parents) #copy so that array is fully initialized

    offspringSize = populationSize - state.n_newInds #! recombination and mutation will only be performed on the offspring of the selected, new indidivudals will be generated randomly anyways

    fitvals = @view state.valarray[1,:]


    #* select offspring
    selected = method.selection(fitvals, offspringSize, rng=rng)


    #* perform mating
    recombine!(offspring, parents, selected, method, state, rng=rng)


    #* perform mutation
    # mutate!(view(offspring,1:offspringSize), method, constraints, rng=rng) #! only mutate descendants of the selected
    mutate!([ind for ind in eachcol(offspring)], method, constraints, rng=rng) #! only mutate descendants of the selected
    #* Generate new individuals for niching
    if state.n_newInds > 0
        new_inds = @view offspring[offspringSize+1:end] #! writes to offspring directly
        generate_new_individuals!(new_inds, constraints)
    end

    #* calculate fitness, period, and amplitude of the population
    # evaluate!(objfun, offspring, state.fitvals, state.periods, state.amplitudes)
    evaluate!(objfun, offspring, state.valarray)


    #* select the best individual
    _, fitidx = vfindmax(fitvals)
    # @info "Max fitness: $(fitmax)"
    state.fittestInd .= offspring[:, fitidx]

    
    #* replace population
    parents .= offspring

    return false
end
"""
    recombine!(offspring, parents, selected, method, state::CustomGAState; rng::AbstractRNG=Random.default_rng())

Recombine the selected individuals from the parents population into the offspring population using the `method` recombination method. Tracks lineage.
"""
function recombine!(offspring, parents::Vector{Vector{Float64}}, selected, method, state::CustomGAState, n=length(selected);
                    rng::AbstractRNG=default_rng())
    mates = ((i,i == n ? i-1 : i+1) for i in 1:2:n)
    for (i,j) in mates
        p1, p2 = parents[selected[i]], parents[selected[j]]
        if rand(rng) < method.crossoverRate
            offspring[i], offspring[j] = method.crossover(p1, p2, rng=rng)
        else
            offspring[i], offspring[j] = p1, p2
        end
        # Update lineage for offspring
        # state.lineages[i] = state.lineages[j] = [selected[i], selected[j]]
        state.lineages[:,i] .= state.lineages[:,j] .= [selected[i], selected[j]]
    end
end

function recombine!(offspring, parents::Matrix{Float64}, selected, method, state::CustomGAState, n=length(selected);
                    rng::AbstractRNG=default_rng())
    mates = ((i,i == n ? i-1 : i+1) for i in 1:2:n)
    for (i,j) in mates
        p1, p2 = parents[:,selected[i]], parents[:,selected[j]]
        if rand(rng) < method.crossoverRate
            offspring[:,i], offspring[:,j] = method.crossover(p1, p2, rng=rng)
        else
            offspring[:,i], offspring[:,j] = p1, p2
        end
        # Update lineage for offspring
        # state.lineages[i] = state.lineages[j] = [selected[i], selected[j]]
        state.lineages[:,i] .= state.lineages[:,j] .= [selected[i], selected[j]]
    end
end

#> END OF CUSTOM GA STATE CONSTRUCTOR AND UPDATE STATE FUNCTION ##



"""
    generate_new_individuals!(new_inds::IT, constraints::CT) where {IT <: AbstractArray, CT <: BoxConstraints}

Generates `n_newInds` individuals to fill out the `offspring` array through log-uniform sampling.
"""
function generate_new_individuals!(new_inds::IT, constraints::CT) where {IT <: AbstractArray, CT <: BoxConstraints}

    rand_vals = Vector{Float64}(undef, length(new_inds))
    
    # Populate the array
    i = 1
    for minidx in 1:2:length(constraints.bounds.bx)
        min_val, max_val = log10(constraints.bounds.bx[minidx]), log10(constraints.bounds.bx[minidx+1])
        rand_vals .= exp10.(rand(Uniform(min_val, max_val), length(new_inds)))
        
        for j in eachindex(new_inds)
            new_inds[j][i] = rand_vals[j]
        end
        i += 1
    end
    return new_inds
end

# """
#     unique_tournament_bitarray(groupSize::Int; select=argmax)

# Returns a function that performs a unique tournament selection of `groupSize` individuals from a population. 

# - WARNING: number of selected will be < N if N is `populationsize`, so offspring array won't be filled completely
# """
# function unique_tournament_bitarray(groupSize::Int; select=argmax)
#     @assert groupSize > 0 "Group size must be positive"
#     function tournamentN(fitness::AbstractVecOrMat{<:Real}, N_selected::Int;
#                          rng::AbstractRNG=Random.GLOBAL_RNG)
#         sFitness = size(fitness)
#         d, nFitness = length(sFitness) == 1 ? (1, sFitness[1]) : sFitness
#         selected_flags = falses(nFitness)  # BitArray
#         selection = Vector{Int}(undef, N_selected)
        
#         count = 1
#         while count <= N_selected
#             tour = randperm(rng, nFitness)
#             j = 1
#             while (j+groupSize) <= nFitness && count <= N_selected
#                 idxs = tour[j:j+groupSize-1]
#                 idxs = filter(x -> !selected_flags[x], idxs)  # Remove already selected
                
#                 if isempty(idxs)
#                     j += groupSize
#                     continue
#                 end
                
#                 selected = d == 1 ? view(fitness, idxs) : view(fitness, :, idxs)
#                 winner = select(selected)
#                 winner_idx = idxs[winner]
                
#                 if !selected_flags[winner_idx]
#                     selection[count] = winner_idx
#                     selected_flags[winner_idx] = true
#                     count += 1
#                 end
                
#                 j += groupSize
#             end
#         end
        
#         return selection
#     end
#     return tournamentN
# end


#< OBJECTIVE FUNCTION OVERRIDES ##
"""
    EvolutionaryObjective(f, x[, F])

Constructor for an objective function object around the function `f` with initial parameter `x`, and objective value `F`.
"""
function EvolutionaryObjective(f::TC, x::Vector{Float64}, F::Vector{Float64};
                               eval::Symbol = :serial) where {TC}
    # @info "Using custom EvolutionaryObjective constructor"
    defval = default_values(x)

    #* convert function into the in-place one
    TF = typeof(F)

    fn = (Fv,xv) -> (Fv .= f(xv))
    TN = typeof(fn)

    EvolutionaryObjective{TN,TF,typeof(x),Val{eval}}(fn, F, defval, 0)
end

"""Override of the multiobjective check"""
ismultiobjective(obj::EvolutionaryObjective{Function, AbstractArray, Vector{Float64}, Val{:thread}}) = false



"""Override of evaluate! to take multiple outputs from the objective function"""
function evaluate!(objfun, population, fitvals, periods, amplitudes)

    #* calculate fitness of the population
    value!(objfun, fitvals, periods, amplitudes, population)

end

"""Override of evaluate! to take multiple outputs from the objective function and write to single array"""
function evaluate!(objfun, population, valarray)

    #* calculate fitness of the population
    value!(objfun, valarray, population)

end

"""Modified value! function from Evolutionary.jl to allow for multiple outputs from the objective function to be stored"""
function value!(obj::EvolutionaryObjective{TC, Vector{Float64}, Vector{Float64},Val{:thread}},
                                F::Vector{Float64}, P::Vector{Float64}, A::Vector{Float64}, xs::Vector{Vector{Float64}}) where {TC}
    n = length(xs)
    Threads.@threads for i in 1:n
        F[i], P[i], A[i] = value(obj, xs[i])  #* evaluate the fitness, period, and amplitude for each individual
    end
end

function value!(obj::EvolutionaryObjective{TC, Vector{Float64}, Vector{Float64},Val{:thread}},
                                F::VT, P::VT, A::VT, xs::Matrix{Float64}) where {TC, VT <: AbstractArray}
    n = size(xs, 2)
    Threads.@threads for i in 1:n
        F[i], P[i], A[i] = value(obj, xs[:, i])  #* evaluate the fitness, period, and amplitude for each individual
    end
end

function value!(obj::EvolutionaryObjective{TC, Vector{Float64}, Vector{Float64},Val{:thread}},
                                V::VT, xs::Matrix{Float64}) where {TC, VT <: AbstractArray}
    n = size(xs, 2)
    Threads.@threads for i in 1:n
        # @inbounds V[:, i] .= value(obj, xs[:, i])  #* evaluate the fitness, period, and amplitude for each individual

        vview = view(V, :, i)
        value(obj, vview, xs[:, i])
    end
end

function optimize(f::TC, F::TF, constraints::BoxConstraints, method::M, population::Matrix{Float64},
                  opts::Options) where {TC, TF, M<:AbstractOptimizer}

    @assert size(population, 2) > 0 "Population is empty"
    objfun = EvolutionaryObjective(f, population[:,1], F; eval=opts.parallelization)
    optimize(objfun, constraints, method, population, opts)
end

# function Evolutionary.value!(obj::EvolutionaryObjective{TC, Vector{Float64}, Vector{Float64}, Val{:thread}},
#                                 F::Matrix{Float64}, xs::Vector{Vector{Float64}}) where {TC}
#     n = length(xs)
#     # @info "Evaluating $(n) individuals in parallel"
#     # @info size(F)
#     # @info "Evaluating F as single matrix"
#     # @info "F type: $(typeof(F))"
#     Threads.@threads for i in 1:n
#         fv = view(F, :, i)
#         value(obj, fv, xs[i])
#     end
# end

# """Same value! function but with SERIAL eval"""
# function Evolutionary.value!(obj::EvolutionaryObjective{TC,TF,Vector{Float64},Val{:serial}},
#                                 F::Matrix{Float64}, xs::Vector{Vector{Float64}}) where {TC,TF}
#     n = length(xs)
#     for i in 1:n
#         # F[:,i] .= Evolutionary.value(obj, xs[i])  #* evaluate the fitness, period, and amplitude for each individual
#         # println("Ind: $(xs[i]) fit: $(F[i]) per: $(P[i]) amp: $(A[i])")
#         fv = view(F, :, i)
#         value(obj, fv, xs[i])
#     end
#     F
# end
#> END OF OVERRIDES ##



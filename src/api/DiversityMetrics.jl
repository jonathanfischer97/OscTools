"""
    getmax_pairwise_diversity(population::AbstractMatrix{Float64})

Computes the maximum pairwise distance as a diversity metric for a population of individuals.
"""
function getmax_pairwise_diversity(population::AbstractMatrix{Float64})
    # Step 1: Log Transformation 
    log_population = log.(population)
    
    # Step 2: Normalization
    # min_vals = minimum(log_population, dims=2)
    # max_vals = maximum(log_population, dims=2)
    # normalized_population = (log_population .- min_vals) ./ (max_vals - min_vals)
    
    # Step 3 & 4: Compute Average Pairwise Distances
    # distances = [norm(normalized_population[:, i] - normalized_population[:, j]) for i in 1:n for j in (i+1):n]
    distances = pairwise(Euclidean(), log_population, dims=2)
    
    return maximum(distances)
end

function getmax_pairwise_diversity(population::Vector{Vector{Float64}})
    pop_matrix = stack(population)

    getmax_pairwise_diversity(pop_matrix)
end

function getmax_pairwise_diversity(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp])
    dfmat = df_to_matrix(df, exclude_cols)
    return getmax_pairwise_diversity(dfmat)
end

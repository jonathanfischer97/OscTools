"""
    getmax_pairwise_diversity(population::AbstractMatrix{Float64})

Computes the maximum pairwise distance as a diversity metric for a population of individuals.
"""
function getmax_pairwise_diversity(population::AbstractMatrix{Float64}, method=Euclidean())
    
    distances = pairwise(method, population, dims=2)
    
    return maximum(triu(distances))
end

function getmax_pairwise_diversity(population::Vector{Vector{Float64}}, method=Euclidean())
    pop_matrix = stack(population)

    getmax_pairwise_diversity(pop_matrix, method)
end

function getmax_pairwise_diversity(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp, :DF], method=Euclidean())
    dfmat = df_to_matrix(df, exclude_cols)
    return getmax_pairwise_diversity(dfmat, method)
end





function get_spread(S::AbstractMatrix)
    n = size(S,2) # number of individuals
    n == 1 && return NaN # if there is only one individual, return NaN

    # Compute pairwise Euclidean distances
    dists = pairwise(Euclidean(), S, dims=2)

    # Replace diagonal of distance matrix with Inf to exclude self-distances
    dists[diagind(dists)] .= Inf

    # Compute minimum distance for each individual
    Δₖ = [minimum(dists[:,i]) for i in 1:n]

    # Compute mean minimum distance
    Δ = mean(Δₖ)

    # Compute spread metric by summing the absolute difference between each minimum distance and the mean, scaled by the mean
    sum(abs.(Δₖ.-Δ))/n*Δ
end

"""
    get_spread(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp, :DF])

Computes the spread metric for a population of individuals, which is the average deviation of all minimum distances to the nearest neighbor from the mean distance, scaled by the mean.
"""
function get_spread(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp, :DF])
    dfmat = df_to_matrix(df, exclude_cols)
    return get_spread(dfmat)
end



"""
    calculate_shannon_index(clustering_result::ClusteringResult)

Calculate the Shannon diversity index for a given clustering result.

# Arguments
- `clustering_result::ClusteringResult`: The result of a clustering operation.

# Returns
- `Float64`: The Shannon diversity index.
"""
function get_shannon_index(cr::ClusteringResult)
    n = nclusters(cr) # get the number of clusters
    p = wcounts(cr) / sum(wcounts(cr)) # get the proportions of each cluster
    H = -sum(p .* log.(p)) # compute the Shannon Index
    return H
end

function get_shannon_index(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp, :DF])
    best_k = get_optimal_clusters(df, 20, exclude_cols)
    cr = get_kmeans(df, best_k, exclude_cols)
    return shannon_index(cr)
end



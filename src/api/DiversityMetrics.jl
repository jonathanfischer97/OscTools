"""
    getmax_pairwise_diversity(population::AbstractMatrix{Float64}, method=Euclidean())

Computes the maximum pairwise distance as a diversity metric for a population of individuals. Non-allocating.
"""
function getmax_pairwise_diversity(population::AbstractMatrix{Float64}, method=Euclidean())::Float64
    max_dist = 0.0
    n = size(population, 2) # assuming each row is an individual

    for i in 1:n
        for j in i+1:n
            # Calculate distance between the i-th and j-th individual
            dist = evaluate(method, view(population, :, i), view(population, :, j))
            max_dist = max(max_dist, dist)
        end
    end

    return max_dist
end

function getmax_pairwise_diversity(population::Vector{Vector{Float64}}, method=Euclidean())
    pop_matrix = stack(population)

    getmax_pairwise_diversity(pop_matrix, method)
end

function getmax_pairwise_diversity(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :relamp, :DF], method=Euclidean())
    dfmat = df_to_matrix(df, exclude_cols)
    return getmax_pairwise_diversity(dfmat, method)
end


"""
    get_nearest_neighbor(S::AbstractMatrix{Float64})

Uses `KDTree` to search the nearest neighbor distance for each individual in a population of individuals.
"""
function get_nearest_neighbor(S::Matrix{Float64})::Vector{Float64}

    # Create a KDTree with Euclidean metric
    kdtree = KDTree(S)

    # Find the nearest neighbor for each point
    _, dists = knn(kdtree, S, 2, true)

    # Compute minimum distance for each individual
    getindex.(dists, 2)
end


# """
#     get_nearest_neighbor2(population::AbstractMatrix{Float64})

# Non-allocating version of `get_nearest_neighbor`.
# """
# function get_nearest_neighbor2(population::Matrix{Float64}, method=Euclidean())
#     n = size(population, 2) # assuming each col is an individual

#     min_dists = Vector{Float64}(undef, n)

#     for i in 1:n
#         i_min_dist = Inf
#         for j in i+1:n
#             # Calculate distance between the i-th and j-th individual
#             dist = Distances.evaluate(method, view(population, :, i), view(population, :, j))
#             i_min_dist = min(i_min_dist, dist)
#         end
#         min_dists[i] = i_min_dist
#     end

#     return min_dists
# end



function get_spread(S::Matrix{Float64}) 
    n = size(S,2) # number of individuals
    n == 1 && return NaN # if there is only one individual, return NaN

    # Compute minimum distance for each individual
    Δₖ = get_nearest_neighbor(S)

    # Compute mean minimum distance
    Δ = mean(Δₖ)

    # Compute spread metric by summing the absolute difference between each minimum distance and the mean, scaled by the mean
    sum(abs.(Δₖ.-Δ))/n*Δ
end

"""
    get_spread(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :DF])

Computes the spread metric for a population of individuals, which is the average deviation of all minimum distances to the nearest neighbor from the mean distance, scaled by the mean.
"""
function get_spread(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :relamp , :DF])
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
    # n = nclusters(cr) # get the number of clusters
    weighted_cluster_sizes = wcounts(cr) # get the weighted cluster sizes
    p = weighted_cluster_sizes / sum(weighted_cluster_sizes) # get the proportions of each cluster
    H = -sum(p .* log.(p)) # compute the Shannon Index
    return H |> abs
end

function get_shannon_index(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :relamp, :DF])
    best_k = get_optimal_clusters(df, 20, exclude_cols)
    cr = get_kmeans(df, best_k, exclude_cols)
    return shannon_index(cr)
end




"""
    calculate_simpson_index(clustering_result::ClusteringResult)

Calculate the Simpson diversity index for a given clustering result.

# Arguments
- `clustering_result::ClusteringResult`: The result of a clustering operation.

# Returns
- `Float64`: The Simpson diversity index.
"""
function calculate_simpson_index(cr::ClusteringResult)
    cluster_sizes = wcounts(cr) # get the cluster sizes
    total_count = sum(cluster_sizes) # total number of elements
    proportions = cluster_sizes / total_count # compute the proportions of each cluster
    simpson_index = 1.0 - sum(proportions.^2) # compute the Simpson Index
    return simpson_index
end

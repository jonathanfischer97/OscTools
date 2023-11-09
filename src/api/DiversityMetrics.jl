# """
#     getmax_pairwise_diversity(population::AbstractMatrix{Float64})

# Computes the maximum pairwise distance as a diversity metric for a population of individuals.
# """
# function getmax_pairwise_diversity(population::AbstractMatrix{Float64}, method=Euclidean())

#     n = size(population, 2) # number of individuals
    
#     distance_matrix = Matrix{Float64}(undef, n, n)
    
#     return getmax_pairwise_diversity!(distance_matrix, population, method)
# end

# function getmax_pairwise_diversity!(distance_matrix, population::AbstractMatrix{Float64}, method=Euclidean())

#     pairwise!(method, distance_matrix, population, dims=2)
    
#     return maximum(triu(distance_matrix))
# end

"""
    getmax_pairwise_diversity(population::AbstractMatrix{Float64}, method=Euclidean())

Computes the maximum pairwise distance as a diversity metric for a population of individuals. Non-allocating.
"""
function getmax_pairwise_diversity(population::AbstractMatrix{Float64}, method=Euclidean())
    max_dist = 0.0
    n = size(population, 2) # assuming each row is an individual

    for i in 1:n
        for j in i+1:n
            # Calculate distance between the i-th and j-th individual
            dist = Distances.evaluate(method, view(population, :, i), view(population, :, j))
            max_dist = max(max_dist, dist)
        end
    end

    return max_dist
end

function getmax_pairwise_diversity(population::Vector{Vector{Float64}}, method=Euclidean())
    pop_matrix = stack(population)

    getmax_pairwise_diversity(pop_matrix, method)
end

function getmax_pairwise_diversity(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp, :DF], method=Euclidean())
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


"""
    get_nearest_neighbor2(population::AbstractMatrix{Float64})

Non-allocating version of `get_nearest_neighbor`.
"""
function get_nearest_neighbor2(population::Matrix{Float64}, method=Euclidean())
    n = size(population, 2) # assuming each col is an individual

    min_dists = Vector{Float64}(undef, n)

    for i in 1:n
        i_min_dist = Inf
        for j in i+1:n
            # Calculate distance between the i-th and j-th individual
            dist = Distances.evaluate(method, view(population, :, i), view(population, :, j))
            i_min_dist = min(i_min_dist, dist)
        end
        min_dists[i] = i_min_dist
    end

    return min_dists
end





# function get_spread(S::AbstractMatrix{Float64}) 
#     n = size(S,2) # number of individuals
#     n == 1 && return NaN # if there is only one individual, return NaN

#     # Compute pairwise Euclidean distances
#     dists = Matrix{Float64}(undef, n, n)
#     return get_spread!(dists, S)
# end

# function get_spread!(dists::Matrix{Float64}, S::AbstractMatrix{Float64}) 
#     pairwise!(Euclidean(), dists, S, dims=2)

#     # Replace diagonal of distance matrix with Inf to exclude self-distances
#     dists[diagind(dists)] .= Inf

#     n = size(S,2) # number of individuals

#     # Compute minimum distance for each individual
#     Δₖ = [minimum(col) for col in eachcol(dists)]

#     # Compute mean minimum distance
#     Δ = mean(Δₖ)

#     # Compute spread metric by summing the absolute difference between each minimum distance and the mean, scaled by the mean
#     sum(abs.(Δₖ.-Δ))/n*Δ
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
    # n = nclusters(cr) # get the number of clusters
    p = wcounts(cr) / sum(wcounts(cr)) # get the proportions of each cluster
    H = -sum(p .* log2.(p)) # compute the Shannon Index
    return H
end

function get_shannon_index(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp, :DF])
    best_k = get_optimal_clusters(df, 20, exclude_cols)
    cr = get_kmeans(df, best_k, exclude_cols)
    return shannon_index(cr)
end



"""
    population_to_matrix(results::GAResults)
Converts the population of a GAResults object to a matrix.
"""
function population_to_matrix(results::GAResults)
    stack(results.population)
end

"""
    kmeans(results::GAResults, clusters::Int)
Performs k-means clustering on the population of a GAResults object, returns a ClusteringResult object.
"""
function get_kmeans(results::GAResults, k::Int)
    data_matrix = population_to_matrix(results)
    return kmeans(data_matrix, k)
end

"""
    df_to_matrix(df::AbstractDataFrame, exclude_cols::Vector{Symbol})
Converts a DataFrame to a transposed matrix while excluding the fixed columns.
"""
function df_to_matrix(df::AbstractDataFrame, exclude_cols::Vector{Symbol} = Symbol[])
    Matrix{Float64}(df[:, Not(exclude_cols)]) |> permutedims
end

"""
    kmeans(df::AbstractDataFrame, clusters::Int, exclude_cols::Vector{Symbol})
Performs k-means clustering on a DataFrame while excluding the fixed columns, returns a ClusteringResult object.
"""
function get_kmeans(df::AbstractDataFrame, k::Int, exclude_cols::Vector{Symbol} = Symbol[])
    data_matrix = df_to_matrix(df, exclude_cols)
    return kmeans(data_matrix, k)
end


"""
    identify_fixed_columns(df::AbstractDataFrame)
Identifies columns in a dataframe that have only one unique value, returns a vector of symbols of the independent variables (excludes gen, fit, per, relamp).
"""
function identify_fixed_columns(df::AbstractDataFrame)
    fixed_cols = Symbol[]
    for col in propertynames(df)
        if length(unique(df[!, col])) == 1
            push!(fixed_cols, col)
        end
    end
    push!(fixed_cols, :gen, :fit, :per, :relamp, :parent1, :parent2)
    return fixed_cols
end


"""
    silhouette_score(X::AbstractMatrix{Float64}, labels::Vector{Int}, sample_size::Int=1000)::Float64

Compute the silhouette score for a sample of data and labels.

# Arguments
- `X::AbstractMatrix{Float64}`: The data matrix where each column is a data point.
- `labels::Vector{Int}`: The cluster assignments of each data point.
- `sample_size::Int`: The number of samples to use for the calculation (default is 1000).

# Returns
- `Float64`: The average silhouette score for the sample.

# Note
- The function samples data points proportionally from each cluster.
"""
function silhouette_score(X::AbstractMatrix{Float64}, labels::Vector{Int}, sample_size::Int=1000)::Float64
    unique_labels = unique(labels)
    n_labels = length(unique_labels)
    n_samples_per_label = sample_size ÷ n_labels
    n = size(X, 2)

    sampled_indices = Int[]

    # Proportionally sample from each cluster without exceeding the total sample size
    for label in unique_labels
        label_indices = findall(x -> x == label, labels)
        n_label_samples = min(length(label_indices), n_samples_per_label, sample_size - length(sampled_indices))
        sampled_indices = [sampled_indices; sample(label_indices, n_label_samples, replace = false)]
    end

    sampled_X = X[:, sampled_indices]
    sampled_labels = labels[sampled_indices]

    # Compute the distance matrix once
    dist_matrix = pairwise(Euclidean(), sampled_X, dims=2)

    sil_scores = Float64[]
    for i in eachindex(sampled_indices)
        own_cluster_indices = findall(x -> x == sampled_labels[i], sampled_labels)

        # Intra-cluster distance (a)
        a = mean(dist_matrix[i, own_cluster_indices])

        # Nearest-cluster distance (b)
        b = Inf
        for other_label in setdiff(unique_labels, [sampled_labels[i]])
            other_cluster_indices = findall(x -> x == other_label, sampled_labels)
            b = min(b, mean(dist_matrix[i, other_cluster_indices]))
        end

        # Silhouette score for this sample
        score = (b - a) / max(a, b)
        push!(sil_scores, score)
    end

    # Return the average silhouette score
    return mean(sil_scores)
end

# """
#     silhouette_score(X::AbstractMatrix{Float64}, labels::Vector{Int}, sample_size::Int=100)
# Computes the silhouette score for a sample of data and labels.
# """
# function silhouette_score(X::AbstractMatrix{Float64}, labels::Vector{Int}, sample_size::Int=1000)
#     # Sample data and corresponding labels
#     idx = rand(1:size(X, 2), sample_size)
#     sampled_X = X[:, idx]
#     sampled_labels = labels[idx]
    
#     dist_matrix = pairwise(Euclidean(), sampled_X, sampled_X)
#     sils = silhouettes(sampled_labels, dist_matrix)
    
#     return mean(sils)
# end

# """
#     optimal_kmeans_clusters(data_matrix::AbstractMatrix{Float64}, max_k::Int)

# Determine optimal cluster count using silhouette method.
# """
# function get_optimal_clusters(data_matrix::AbstractMatrix{Float64}, max_k::Int)

#     # Handle edge cases: empty data matrix or max_k < 2
#     if isempty(data_matrix) || max_k < 2
#         return 1
#     end

#     best_k = 2
#     best_score = -Inf

#     # Create an array to store the scores for each k
#     scores = zeros(max_k - 1)

#     # Iterate through k values and compute silhouette score
#     for k in 2:max_k
#         result = kmeans(data_matrix, k)
#         # @info "Number of clusters: $(nclusters(result))"
#         try 
#             score = silhouette_score(data_matrix, assignments(result))
#             scores[k - 1] = score # Store the score in the array
#             # println("Try: Unique labels: ", unique(assignments(result)))
#         catch
#             # println("Caught: Unique labels: ", unique(assignments(result)))
#             scores[k - 1] = -Inf
#         end
#     end
    
#     # Find the maximum score and the corresponding k value
#     best_score, best_k = findmax(scores)
#     best_k += 1 # Adjust the index to match the k value
    
#     return best_k
# end

"""
    get_optimal_clusters(data_matrix::AbstractMatrix{Float64}, max_k::Int)::Int

Find the optimal number of clusters for a given data matrix using silhouette score.

# Arguments
- `data_matrix::AbstractMatrix{Float64}`: The data matrix where each column is a data point.
- `max_k::Int`: The maximum number of clusters to consider.

# Returns
- `Int`: The optimal number of clusters according to the silhouette score.

# Note
- Returns 1 if the data matrix is empty or if max_k is less than 2.
"""
function get_optimal_clusters(data_matrix::AbstractMatrix{Float64}, max_k::Int)::Int
    n = size(data_matrix, 2)
    
    if max_k > n
        max_k = n
    elseif n < 2
        return 1
    end

    # if isempty(data_matrix) || max_k < 2
    #     return 1
    # end

    #* Initialize variables to store the best silhouette score and corresponding k
    best_score = -Inf
    best_k = 1

    #* Compute silhouette scores for k from 2 to max_k
    for k in 2:max_k
        result = kmeans(data_matrix, k) 
        dist_matrix = pairwise(Euclidean(), data_matrix, dims=2)
        score = mean(silhouettes(assignments(result), dist_matrix))
        # score = silhouette_score(data_matrix, assignments(result))


        #* Update best score and best k if a better score is found
        if score > best_score
            best_score = score
            best_k = k
        end
    end

    return best_k
end

"""
    get_optimal_clusters(df::AbstractDataFrame, max_k::Int, exclude_cols::Vector{Symbol} = [])
Wrapper function for optimal_kmeans_clusters that converts a DataFrame to a Matrix, and returns the optimal cluster count.
"""
function get_optimal_clusters(df::AbstractDataFrame, max_k::Int, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :relamp, :DF])
    # if max_k > nrow(df)
    #     max_k = nrow(df)
    # elseif nrow(df) < 2
    #     return 1
    # end
    data_matrix = df_to_matrix(df, exclude_cols)
    return get_optimal_clusters(data_matrix, max_k)
end

"""
    get_cluster_distances(result::ClusteringResult)
Computes the pairwise distances matrix between cluster centers.
"""
function get_cluster_distances(result::ClusteringResult)
    dist_matrix = pairwise(Euclidean(), result.centers, result.centers)
    return dist_matrix
end



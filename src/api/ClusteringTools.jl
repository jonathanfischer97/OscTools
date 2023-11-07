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
    df_to_matrix(df::DataFrame, exclude_cols::Vector{Symbol})
Converts a DataFrame to a transposed matrix while excluding the fixed columns.
"""
function df_to_matrix(df::DataFrame, exclude_cols::Vector{Symbol})
    return Matrix(df[:, Not(exclude_cols)]) |> transpose
end

"""
    kmeans(df::DataFrame, clusters::Int, exclude_cols::Vector{Symbol})
Performs k-means clustering on a DataFrame while excluding the fixed columns, returns a ClusteringResult object.
"""
function get_kmeans(df::DataFrame, k::Int, exclude_cols::Vector{Symbol} = Symbol[])
    data_matrix = df_to_matrix(df, exclude_cols)
    return kmeans(data_matrix, k)
end




"""
    identify_fixed_columns(df::DataFrame)
Identifies columns in a dataframe that have only one unique value, returns a vector of symbols of the independent variables (excludes gen, fit, per, amp).
"""
function identify_fixed_columns(df::DataFrame)
    fixed_cols = Symbol[]
    for col in propertynames(df)
        if length(unique(df[!, col])) == 1
            push!(fixed_cols, col)
        end
    end
    push!(fixed_cols, :gen, :fit, :per, :amp, :relamp)
    return fixed_cols
end




"""
    silhouette_score(X::AbstractMatrix{Float64}, labels::Vector{Int}, sample_size::Int=100)
Computes the silhouette score for a sample of data and labels.
"""
function silhouette_score(X::AbstractMatrix{Float64}, labels::Vector{Int}, sample_size::Int=100)
    # Get the unique labels and total number of data points
    unique_labels = unique(labels)
    total_points = size(X, 2)

    # Limit sample_size to the number of available data points
    sample_size = min(sample_size, total_points)

    # Preallocate array for sampled indices
    sampled_idx = Int[]

    # Loop through each unique label to sample data points
    for lbl in unique_labels
        idx = findall(x -> x == lbl, labels)

        # Determine how many samples to take from this label, considering available data points
        n_samples = min(sample_size, length(idx))

        # Sample indices without replacement, but only up to the number available
        sampled_indices = sample(idx, n_samples, replace=false)

        append!(sampled_idx, sampled_indices)
    end

    # Extract sampled data and labels
    sampled_X = X[:, sampled_idx]
    sampled_labels = labels[sampled_idx]

    # Calculate pairwise distances and compute silhouette values
    dist_matrix = pairwise(Euclidean(), sampled_X, sampled_X)
    sils = silhouettes(sampled_labels, dist_matrix)

    return mean(sils)
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
    #* Handle edge cases: empty data matrix or max_k < 2
    if isempty(data_matrix) || max_k < 2
        return 1
    end

    #* Initialize variables to store the best silhouette score and corresponding k
    best_score = -Inf
    best_k = 1

    #* Compute silhouette scores for k from 2 to max_k
    for k in 2:max_k
        result = kmeans(data_matrix, k) 
        score = silhouette_score(data_matrix, assignments(result))

        #* Update best score and best k if a better score is found
        if score > best_score
            best_score = score
            best_k = k
        end
    end

    return best_k
end

"""
    get_optimal_clusters(df::DataFrame, max_k::Int, exclude_cols::Vector{Symbol} = [])
Wrapper function for optimal_kmeans_clusters that converts a DataFrame to a Matrix, and returns the optimal cluster count.
"""
function get_optimal_clusters(df::DataFrame, max_k::Int, exclude_cols::Vector{Symbol} = [])
    if max_k > nrow(df)
        max_k = nrow(df)
    elseif nrow(df) < 2
        return 1
    end
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



function get_pca_model(df::DataFrame, exclude_cols::Vector{Symbol} = [:gen, :fit, :per, :amp, :relamp]; maxoutdim=3)
    data_matrix = df_to_matrix(df, exclude_cols)
    return fit(PCA, data_matrix, maxoutdim=maxoutdim)
end

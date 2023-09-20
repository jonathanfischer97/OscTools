"Converts the population of a GAResults object to a matrix."
function population_to_matrix(results::GAResults)
    return hcat(results.population...)'
end

"Performs k-means clustering on a GAResults object."
function kmeans(results::GAResults, k::Int)
    data_matrix = population_to_matrix(results)
    return kmeans(data_matrix, k)
end

"Performs the elbow method on a GAResults object to find the optimal number of clusters."
function elbow_method(results::GAResults, kmax::Int)
    data_matrix = population_to_matrix(results)
    return elbow_method(data_matrix, kmax)
end

function kmeans(df::DataFrame, k::Int; exclude_cols::Vector{Symbol} = Symbol[])
    included_cols = setdiff(propertynames(df), exclude_cols)
    data_matrix = Matrix(df[:, included_cols])
    return kmeans(data_matrix, k)
end

function elbow_method(df::DataFrame, max_k::Int; exclude_cols::Vector{Symbol} = Symbol[])
    included_cols = setdiff(propertynames(df), exclude_cols)
    data_matrix = Matrix(df[:, included_cols])
    return elbow_method(data_matrix, max_k)
end

function elbow_method(X::Matrix{Float64}, max_k::Int)
    distortions = Vector{Float64}(undef, max_k)
    for k in 1:max_k
        result = kmeans(X, k)
        distortions[k] = result.totalcost
    end
    return distortions
end
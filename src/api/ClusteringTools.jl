"""
    population_to_matrix(results::GAResults)
Converts the population of a GAResults object to a matrix.
"""
function population_to_matrix(results::GAResults)
    # return hcat(results.population...)'
    stack(results.population)
end

"""
    kmeans(results::GAResults, clusters::Int)
Performs k-means clustering on the population of a GAResults object, returns a ClusteringResult object.
"""
function kmeans(results::GAResults, k::Int)
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
function kmeans(df::DataFrame, k::Int; exclude_cols::Vector{Symbol} = Symbol[])
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
    push!(fixed_cols, :gen, :fit, :per, :amp)
    return fixed_cols
end
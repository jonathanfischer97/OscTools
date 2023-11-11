#< CSV TOOLS ##
"""
    read_csvs_in_directory(directory_path::String)

Read all CSV files in a given directory into an array of DataFrames.

# Arguments
- `directory_path::String`: The path to the directory containing the CSV files.

# Returns
- `Array{DataFrame, 1}`: An array of DataFrames, each corresponding to a CSV file.
"""
function read_csvs_in_directory(directory_path::String)
    # Initialize an empty array to store DataFrames
    dfs = DataFrame[]
    
    # Loop over each file in the directory
    for file_name in readdir(directory_path)
        # Check if the file is a CSV file
        if occursin(".csv", file_name)
            # Full path to the CSV file
            full_file_path = joinpath(directory_path, file_name)
            
            # Read the CSV file into a DataFrame
            df = DataFrame(CSV.File(full_file_path))
            
            # Append the DataFrame to the array
            push!(dfs, df)
        end
    end
    return dfs
end


"""
Saves GA results to a CSV file without needing to load the results into a DataFrame first
"""
function save_to_csv(results::GAResults, constraints::ConstraintSet, filename::String)
    open(filename, "w") do io
        # Write the header with an additional "Generation" column
        write(io, "gen,fit,per,amp")
        for conrange in constraints
            write(io, ",$(conrange.name)")
        end
        write(io, "\n")
        
        # Loop over each generation based on gen_indices
        for (gen, (start_idx, end_idx)) in enumerate(results.gen_indices)
            for i in start_idx:end_idx
                # Write the generation, fitness, period, and amplitude values
                write(io, "$gen,$(results.fitvals[i]),$(results.periods[i]),$(results.amplitudes[i])")
                
                # Write the population and fixed values
                j = 1
                for conrange in constraints
                    if !conrange.isfixed
                        write(io, ",$(results.population[i][j])")
                        j += 1
                    else
                        write(io, ",$(conrange.fixed_value)")
                    end
                end
                
                write(io, "\n")
            end
        end
    end
end
#> END ##



#< DATAFRAME UTILITIES ##
"""
    make_ga_dataframe(results::GAResults, constraints::ConstraintSet)

Makes a DataFrame from a GAResults object.
"""
function make_ga_dataframe(results::GAResults, constraints::ConstraintSet)
    df = DataFrame(gen = Vector{Int}(undef, length(results.fitvals)), fit = results.fitvals, per = results.periods, amp = results.amplitudes, relamp = Vector{Float64}(undef, length(results.fitvals)))

    #* Loop over each generation based on gen_indices
    for (gen, (start_idx, end_idx)) in enumerate(results.gen_indices)
        df.gen[start_idx:end_idx] .= gen
    end

    i = 1
    for conrange in constraints
        if !conrange.isfixed
            df[!, conrange.name] .= [x[i] for x in results.population]
            i+=1
        else
            df[!, conrange.name] .= conrange.fixed_value
        end
    end
    #* Calculate the relative amplitude by dividing the amp column by the initial concentration of A
    if !isempty(df.A)
        df.relamp .= df.amp ./ df.A
    end
    df.gen = categorical(df.gen; ordered=true)
    return df
end

"""
Makes a DataFrame from a raw population, nested vectors
"""
function make_pop_dataframe(pop::Vector{Vector{Float64}}, constraints::AllConstraints)
    df = DataFrame()
    i = 1
    for conrange in constraints
        if !conrange.isfixed
            df[!, conrange.name] = [x[i] for x in pop]
            i+=1
        else
            df[!, conrange.name] = conrange.fixed_value
        end
    end
    return df
end
#> END ##


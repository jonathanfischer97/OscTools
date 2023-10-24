#< CUSTOM PEAK FINDER
function findmaxpeaks(x; 
                    height::Union{Nothing,<:Real,NTuple{2,<:Real}}=nothing,
                    distance::Union{Nothing,Int}=nothing,
                    find_maxima::Bool=true) #where {T<:Real}
    midpts = Vector{Int}(undef, 0)
    i = 2
    imax = length(x)

    while i < imax
        if (find_maxima && x[i-1] < x[i]) #|| (!find_maxima && x[i-1] > x[i])
            iahead = i + 1
            while (iahead < imax) && (x[iahead] == x[i])
                iahead += 1
            end

            if (find_maxima && x[iahead] < x[i]) #|| (!find_maxima && x[iahead] > x[i])
                push!(midpts, (i + iahead - 1) ÷ 2)
                i = iahead
            end
        end
        i += 1
    end 

    #* Filter by height if needed
    if !isnothing(height)
        hmin, hmax = height isa Number ? (height, nothing) : height
        keepheight = (hmin === nothing || x[midpts] .>= hmin) .& (hmax === nothing || x[midpts] .<= hmax)
        midpts = midpts[keepheight]
    end

    #* Filter by distance if needed
    if !isnothing(distance)
        priority = find_maxima ? x[midpts] : -x[midpts] # Use negative values for minima
        keep = selectbypeakdistance(midpts, priority, distance)
        midpts = midpts[keep]
    end

    extrema_indices = midpts
    extrema_heights = x[extrema_indices]

    extrema_indices, extrema_heights
end

"""
    findextrema(x::Vector{T}; height::Union{Nothing,<:Real,NTuple{2,<:Real}}=nothing, distance::Union{Nothing,Int}=nothing)

Finds the maxima and minima values of a vector, along with their indices. The height and distance filters can be applied to the maxima and minima.
"""
function findextrema(x::Vector{T};
                     height::Union{Nothing,<:Real,NTuple{2,<:Real}}=nothing,
                     distance::Union{Nothing,Int}=nothing) where {T<:Real}
    maxima_indices = Vector{Int}(undef, 0)
    minima_indices = Vector{Int}(undef, 0)
    i = 2
    imax = length(x)

    while i < imax
        iahead = i + 1
        while (iahead < imax) && (x[iahead] == x[i])
            iahead += 1
        end

        # Check for maxima
        if x[i-1] < x[i] && x[iahead] < x[i]
            push!(maxima_indices, (i + iahead - 1) ÷ 2)
        end

        # Check for minima
        if x[i-1] > x[i] && x[iahead] > x[i]
            push!(minima_indices, (i + iahead - 1) ÷ 2)
        end

        i = iahead
    end

    # Apply height and distance filters to both maxima and minima
    filter_extrema!(x, maxima_indices, height, distance)
    filter_extrema!(x, minima_indices, height, distance)

    maxima_heights = x[maxima_indices]
    minima_heights = x[minima_indices]

    return maxima_indices, maxima_heights, minima_indices, minima_heights
end


function filter_extrema!(x::Vector{T}, extrema::Vector{Int}, height, distance) where {T<:Real}
    # Filter by height
    if !isnothing(height)
        hmin, hmax = height isa Number ? (height, nothing) : height
        keepheight = (hmin === nothing || x[extrema] .>= hmin) .& (hmax === nothing || x[extrema] .<= hmax)
        extrema = extrema[keepheight]
    end

    # Filter by distance
    if !isnothing(distance)
        priority = x[extrema]  # Priority based on height
        keep = selectbypeakdistance(extrema, priority, distance)
        extrema = extrema[keep]
    end
end



function selectbypeakdistance(pkindices, priority, distance)
    npkindices = length(pkindices)
    keep = trues(npkindices)

    prioritytoposition = sortperm(priority, rev=true)
    for i ∈ npkindices:-1:1
        j = prioritytoposition[i]
        iszero(keep[j]) && continue

        k = j-1
        while (1 <= k) && ((pkindices[j]-pkindices[k]) < distance)
            keep[k] = false
            k -= 1
        end

        k = j+1
        while (k <= npkindices) && ((pkindices[k]-pkindices[j]) < distance)
            keep[k] = false
            k += 1
        end
    end
    keep
end

# function findextrema(x::AbstractVector{T}; 
#                     height::Union{Nothing,<:Real,NTuple{2,<:Real}}=nothing,
#                     distance::Union{Nothing,Int}=nothing) where {T<:Real}
#     midpts = Vector{Int}(undef, 0)
#     i = 2
#     imax = length(x)

#     while i < imax
#         if x[i-1] < x[i]
#             iahead = i + 1
#             while (iahead < imax) && (x[iahead] == x[i])
#                 iahead += 1
#             end
#             if x[iahead] < x[i]
#                 push!(midpts, (i + iahead - 1) ÷ 2)
#                 i = iahead
#             end
#         end
#         i += 1
#     end 

#     #* Filter by height if needed
#     if !isnothing(height)
#         hmin, hmax = height isa Number ? (height, nothing) : height
#         keepheight = (hmin === nothing || x[midpts] .>= hmin) .& (hmax === nothing || x[midpts] .<= hmax)
#         midpts = midpts[keepheight]
#     end

#     #* Filter by distance if needed
#     if !isnothing(distance)
#         priority = x[midpts]
#         keep = selectbypeakdistance(midpts, priority, distance)
#         midpts = midpts[keep]
#     end

#     extrema_indices = midpts
#     extrema_heights = x[extrema_indices]

#     extrema_indices, extrema_heights
# end

# function find_minima_between_maxima(x::AbstractVector{T}, maxima_indices::Vector{Int}) where {T<:Real}
#     minima_indices = Vector{Int}(undef, length(maxima_indices) - 1)
#     minima_values = Vector{T}(undef, length(maxima_indices) - 1)

#     for i in 1:(length(maxima_indices) - 1)
#         range = maxima_indices[i]:maxima_indices[i+1]
#         minima_indices[i] = argmin(x[range]) + range.start - 1
#         minima_values[i] = x[minima_indices[i]]
#     end

#     return minima_indices, minima_values
# end



# """
# Struct to hold the properties of the peaks found by the peak finder
# """
# struct PeakProperties
#     peak_heights::Union{Nothing, Vector{Float64}}
#     prominences::Union{Nothing, Vector{Float64}}
#     leftbases::Union{Nothing, Vector{Int}}
#     rightbases::Union{Nothing, Vector{Int}}
#     widths::Union{Nothing, Vector{Float64}}
#     widthheights::Union{Nothing, Vector{Float64}}
#     leftips::Union{Nothing, Vector{Float64}}
#     rightips::Union{Nothing, Vector{Float64}}
# end


# function filterproperties!(properties::PeakProperties, keep::BitVector)
#     properties.peak_heights = properties.peak_heights[keep]
#     properties.prominences = properties.prominences[keep]
#     properties.leftbases = properties.leftbases[keep]
#     properties.rightbases = properties.rightbases[keep]
#     properties.widths = properties.widths[keep]
#     properties.widthheights = properties.widthheights[keep]
#     properties.leftips = properties.leftips[keep]
#     properties.rightips = properties.rightips[keep]
# end

# function findpeaks1d(x::AbstractVector{T};
#                     height::Union{Nothing,<:Real,NTuple{2,<:Real}}=nothing,
#                     distance::Union{Nothing,I}=nothing,
#                     prominence::Union{Nothing,Real,NTuple{2,Real}}=nothing,
#                     width::Union{Nothing,Real,NTuple{2,Real}}=nothing,
#                     wlen::Union{Nothing,I}=nothing,
#                     relheight::Real=0.5,
#                     calc_peak_heights::Bool=false,
#                     calc_prominences::Bool=false,
#                     calc_widths::Bool=false) where {T<:Real,I<:Integer}

#     pkindices, leftedges, rightedges = localmaxima1d(x)

#     # Initialize variables for optional calculations
#     peak_heights = nothing
#     prominences = nothing
#     leftbases = nothing
#     rightbases = nothing
#     widths = nothing
#     widthheights = nothing
#     leftips = nothing
#     rightips = nothing

#     if calc_peak_heights && !isnothing(height)
#         pkheights = x[pkindices]
#         hmin, hmax = height isa Number ? (height, nothing) : height
#         keepheight = selectbyproperty(pkheights, hmin, hmax)
#         pkindices = pkindices[keepheight]
#         peak_heights = pkheights[keepheight]
#     end

#     if !isnothing(distance)
#         keepdist = selectbypeakdistance(pkindices, x[pkindices], distance)
#         pkindices = pkindices[keepdist]
#     end

#     if calc_prominences && (!isnothing(prominence) || !isnothing(width))
#         prominences, leftbases, rightbases = peakprominences1d(x, pkindices, wlen)
#     end

#     if !isnothing(prominence)
#         pmin, pmax = prominence isa Number ? (prominence, nothing) : prominence
#         keepprom = selectbyproperty(prominences, pmin, pmax)
#         pkindices = pkindices[keepprom]
#     end

#     if calc_widths && !isnothing(width)
#         widths, widthheights, leftips, rightips = peakwidths1d(x, pkindices, relheight, prominences, leftbases, rightbases)
#         wmin, wmax = width isa Number ? (width, nothing) : width
#         keepwidth = selectbyproperty(widths, wmin, wmax)
#         pkindices = pkindices[keepwidth]
#     end

#     # Construct the properties struct with the calculated values
#     properties = PeakProperties(peak_heights, prominences, leftbases, rightbases, widths, widthheights, leftips, rightips)

#     pkindices, properties
# end
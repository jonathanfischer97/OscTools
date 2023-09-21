abstract type ConstraintSet end



#< CONSTRAINT RANGE OVERLOADS ##
# Define the length method
Base.length(constraint::ConstraintSet) = length(fieldnames(typeof(constraint)))

# Define the getindex method for index-based access
function Base.getindex(constraint::ConstraintSet, idx::Int)
    field_name = fieldnames(typeof(constraint))[idx]
    return getfield(constraint, field_name)
end

# To make it iterable, define start, next and done methods
Base.iterate(constraint::ConstraintSet, state=1) = state > length(constraint) ? nothing : (getfield(constraint, fieldnames(typeof(constraint))[state]), state + 1)

# Required for the `in` keyword
Base.eltype(::Type{ConstraintSet}) = ConstraintRange

#> END ##


#< CONSTRAINT RANGE RETRIEVERS, TOOLS ##
"""Gives the active length of a ConstraintSet, i.e. the number of elements that are not fixed. Doesn't care whether value is assigned or not!"""
activelength(constraints::ConstraintSet) = count(x -> !x.isfixed, constraints)

"""Gives the fixed length of a ConstraintSet, i.e. the number of elements that are fixed. Doesn't care whether value is assigned or not!"""
fixedlength(constraints::ConstraintSet) = count(x -> x.isfixed, constraints)


"""Returns the numerical index of a field in a ConstraintSet"""
function find_field_index(field_name::Union{Symbol, String}, constraint::ConstraintSet)
    fields = fieldnames(typeof(constraint))
    idx = findfirst(x -> x == Symbol(field_name), fields)
    
    if idx === nothing
        throw(ArgumentError("Field name '$field_name' not found in ConstraintSet."))
    end
    
    return idx
end

"""Returns a vector of the numerical indices of the fixed elements in a ConstraintSet"""
function get_fixed_indices(constraints::ConstraintSet)::Vector{Int}
    inactive_indices = Int[]  # Initialize an empty array to store the indices of inactive elements
    for (idx, constraint) in enumerate(constraints)  # Loop through each element along with its index
        if constraint.isfixed  # Check if the element is fixed
            push!(inactive_indices, idx)  # Add the index to the array
        end
    end
    return inactive_indices  # Return the array of indices
end


"""Returns a vector of the constraintranges that are marked as FIXED but have NOT BEEN ASSIGNED fixed values"""
function get_fixed_constraintranges(constraints::ConstraintSet)::Vector{ConstraintRange}
    fixed_constraintranges = ConstraintRange[]
    for constraintrange in constraints  # Loop through each element along with its index
        if constraintrange.isfixed && isnan(constraintrange.fixed_value)  # Check if the element is fixed but not assigned a value
            push!(fixed_constraintranges, constraintrange)  # Add the index to the array
        end
    end
    return fixed_constraintranges  # Return the array of indices
end


"""Returns a vector of the names (symbols) of the fixed input"""
get_fixed_names(constraints::ConstraintSet) = [constraintrange.name for constraintrange in constraints if constraintrange.isfixed]

#> END ##


#< CONCRETE TYPES FOR CONSTRAINTS ##
"""
    ConstraintRange

Struct for defining parameter or initial condition ranges. Each instance contains a name, and a range defined by a minimum and maximum value.

# Fields
- `name::String`: Name of the parameter or initial condtion.
- `min::Float64`: Minimum allowed value.
- `max::Float64`: Maximum allowed value.
- `isfixed::Bool`: Whether the parameter or initial condition is fixed. Defaults to false.
- `fixed_value::Float64`: Fixed value is to be used if fixed. Defaults to NaN.
"""
@kwdef mutable struct ConstraintRange
    const name::Symbol
    const min::Float64
    const max::Float64
    isfixed::Bool = false
    fixed_value::Float64 = NaN
end


"""
    ParameterConstraints

Struct encapsulating parameter constraints. Each field represents a different parameter, holding a `ConstraintRange` object that defines the valid range for that parameter.
"""
mutable struct ParameterConstraints <: ConstraintSet
    ka1::ConstraintRange
    kb1::ConstraintRange
    kcat1::ConstraintRange
    ka2::ConstraintRange
    kb2::ConstraintRange
    ka3::ConstraintRange
    kb3::ConstraintRange
    ka4::ConstraintRange
    kb4::ConstraintRange
    ka7::ConstraintRange
    kb7::ConstraintRange
    kcat7::ConstraintRange
    DF::ConstraintRange
end

"""
    InitialConditionConstraints

Struct encapsulating initial condition constraints. Each field represents a different initial condition, holding a `ConstraintRange` object that defines the valid range for that initial condition.
"""
mutable struct InitialConditionConstraints <: ConstraintSet 
    L::ConstraintRange
    K::ConstraintRange
    P::ConstraintRange
    A::ConstraintRange
end

"""
    AllConstraints

Struct encapsulating all constraints. Each field represents a different parameter or initial condition, holding a `ConstraintRange` object that defines the valid range for that parameter or initial condition.
"""
mutable struct AllConstraints <: ConstraintSet
    ka1::ConstraintRange
    kb1::ConstraintRange
    kcat1::ConstraintRange
    ka2::ConstraintRange
    kb2::ConstraintRange
    ka3::ConstraintRange
    kb3::ConstraintRange
    ka4::ConstraintRange
    kb4::ConstraintRange
    ka7::ConstraintRange
    kb7::ConstraintRange
    kcat7::ConstraintRange
    DF::ConstraintRange

    L::ConstraintRange
    K::ConstraintRange
    P::ConstraintRange
    A::ConstraintRange
end
#> END ##



#< CONSTRAINT CONSTRUCTORS ##
"""
    ParameterConstraints(; kwargs...)

Define parameter constraints. Each keyword argument represents a different parameter, where the value is a tuple defining the valid range for that parameter.

# Example
```julia
constraints = ParameterConstraints(
    karange = (-3.0, 1.0), 
    kbrange = (-3.0, 3.0), 
    kcatrange = (-3.0, 3.0), 
    dfrange = (1.0, 5.0)
)
```
"""
function ParameterConstraints(; karange = (1e-3, 1e2), kbrange = (1e-3, 1e3), kcatrange = (1e-3, 1e3), dfrange = (1e2, 2e4))#, nominalvals = repeat([Nothing],13))
    #* Define parameter constraint ranges
    ka_min, ka_max = karange  # uM^-1s^-1, log scale
    kb_min, kb_max = kbrange  # s^-1, log scale
    kcat_min, kcat_max = kcatrange # s^-1, log scale
    df_min, df_max = dfrange # for DF, log scale

    return ParameterConstraints(
        ConstraintRange(name = :ka1, min = ka_min, max = ka_max),
        ConstraintRange(name = :kb1, min = kb_min, max = kb_max),
        ConstraintRange(name = :kcat1, min = kcat_min, max = kcat_max),
        ConstraintRange(name = :ka2, min = ka_min, max = ka_max),
        ConstraintRange(name = :kb2, min = kb_min, max = kb_max),
        ConstraintRange(name = :ka3, min = ka_min, max = ka_max),
        ConstraintRange(name = :kb3, min = kb_min, max = kb_max),
        ConstraintRange(name = :ka4, min = ka_min, max = ka_max),
        ConstraintRange(name = :kb4, min = kb_min, max = kb_max),
        ConstraintRange(name = :ka7, min = ka_min, max = ka_max),
        ConstraintRange(name = :kb7, min = kb_min, max = kb_max),
        ConstraintRange(name = :kcat7, min = kcat_min, max = kcat_max),
        ConstraintRange(name = :DF, min = df_min, max = df_max)
    )
end


"""
    InitialConditionConstraints(; kwargs...)

Define initial condition constraints. Each keyword argument represents a different initial condition, where the value is a tuple defining the valid range for that initial condition.

# Example
```julia
constraints = InitialConditionConstraints(
    lipidrange = (0.1, 10.0), 
    kinaserange = (0.1, 10.0), 
    phosphataserange = (0.1, 10.0), 
    ap2range = (0.1, 10.0)
)
```
"""
function InitialConditionConstraints(; Lrange = (1e-1, 1e2), Krange = (1e-2, 1e2), Prange = (1e-2, 1e2), Arange = (1e-1, 1e2))#, nominalvals = repeat([Nothing],4))
    # Define initial condition constraint ranges
    lipid_min, lipid_max = Lrange  # uM
    kinase_min, kinase_max = Krange  # uM
    phosphatase_min, phosphatase_max = Prange # uM
    ap2_min, ap2_max = Arange # uM

    return InitialConditionConstraints(
        ConstraintRange(name = :L, min = lipid_min, max = lipid_max),
        ConstraintRange(name = :K, min = kinase_min, max = kinase_max),
        ConstraintRange(name = :P, min = phosphatase_min, max = phosphatase_max),
        ConstraintRange(name = :A, min = ap2_min, max = ap2_max)
    )
end


function AllConstraints(paramconstraints::ParameterConstraints=ParameterConstraints(), icconstraints::InitialConditionConstraints=InitialConditionConstraints()) 
    return AllConstraints(
        paramconstraints.ka1,
        paramconstraints.kb1,
        paramconstraints.kcat1,
        paramconstraints.ka2,
        paramconstraints.kb2,
        paramconstraints.ka3,
        paramconstraints.kb3,
        paramconstraints.ka4,
        paramconstraints.kb4,
        paramconstraints.ka7,
        paramconstraints.kb7,
        paramconstraints.kcat7,
        paramconstraints.DF,

        icconstraints.L,
        icconstraints.K,
        icconstraints.P,
        icconstraints.A
    )
end
#> END ##


#< TOOLS FOR FIXING INPUTS/CONSTRAINTS ##
"""Simply marks the constraints as fixed, without assigning a value"""
function set_fixed_constraints!(constraints::ConstraintSet, fixednames::Vector{Symbol})
    for name in fixednames
        if name in fieldnames(typeof(constraints))
            conrange = getfield(constraints, name)
            conrange.isfixed = true
            conrange.fixed_value = NaN
        end
    end
    return constraints
end

"""Sets the vector of unpacked fixed constraints according to symbol, assigning the given values"""
function set_fixed_values!(fixed_constraintranges::Vector{ConstraintRange}, values...)
    for (conrange, value) in zip(fixed_constraintranges, values)
        conrange.fixed_value = value
    end
    return fixed_constraintranges
end

function set_fixed_values!(constraints::ConstraintSet, values...)
    fixed_constraintranges = get_fixed_constraintranges(constraints)
    return set_fixed_values!(fixed_constraintranges, values...)
end

"""Unsets the fixed constraints according to symbol, resetting both the isfixed and fixed_value fields to default"""
function unset_fixed_constraints!(constraints::ConstraintSet, fixednames::Vector{Symbol})
    for name in fixednames
        if name in fieldnames(typeof(constraints))
            conrange = getfield(constraints, name)
            conrange.isfixed = false
            conrange.fixed_value = NaN
        end
    end
    return constraints
end
#> END
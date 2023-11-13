module OscTools 

    using Catalyst, OrdinaryDiffEq
    using ModelingToolkit: modelingtoolkitize
    using Evolutionary, FFTW
    using Statistics, Random, Distributions
    using DataFrames, CSV
    using LinearAlgebra
    using ProgressMeter
    using Plots#, RecipesBase
    using ColorSchemes, Plots.PlotMeasures
    using Clustering
    using Distances
    using NearestNeighbors
    using CategoricalArrays
    using Printf
    using StatsBase



    # import the overloads for Evolutionary.jl
    include("api/EvolutionaryOverloads.jl")

    # import the Catalyst model "fullrn"
    include("api/ReactionNetwork.jl")
    export make_fullrn, make_ODE_problem

    # import custom peak finder
    include("api/CustomPeakFinder.jl")

    # import the cost function and other evaluation functions
    include("api/EvaluationFunctions.jl")
    export FitnessFunction, getFrequencies, getSTD, getDif, 
            getPerAmp, solve_odeprob, solve_for_fitness_peramp, eval_fitness

    # import the constraint types and tools
    include("api/ConstraintSetTools.jl")
    export ConstraintSet, ConstraintRange, ParameterConstraints, InitialConditionConstraints, AllConstraints, 
            activelength, set_fixed_constraints!, set_fixed_values!, unset_fixed_constraints!, get_fixed_constraintranges, minima, maxima

    # import the genetic algorithm and associated functions
    include("api/GA_functions.jl")
    export make_fitness_function_threaded, GAProblem, 
            generate_population, generate_empty_population, generate_population!, logrange,
            GAResults, run_GA

        # # import the differential evolution algorithm and associated functions
        # include("api/DE_functions.jl")
        # export DifferentialEvolutionProblem, DifferentialEvolutionResults, run_DE

    #include the file utilities
    include("api/FileUtils.jl")
    export read_csvs_in_directory, save_to_csv, make_ga_dataframe, make_pop_dataframe

    #include plotting utilities
    include("api/PlotUtils.jl")
    export plotsol, plotboth, plot_everything, plot_everything_from_csv_indir, get_row_prob

    #include clustering utilities
    include("api/ClusteringTools.jl")
    export population_to_matrix, get_kmeans, df_to_matrix, identify_fixed_columns, get_optimal_clusters, get_cluster_distances

    #include diversity metrics
    include("api/DiversityMetrics.jl")
    export getmax_pairwise_diversity, get_spread, get_shannon_index

    #include testing functions
    include("api/TestingFunctions.jl")
    export test4fixedGA

end
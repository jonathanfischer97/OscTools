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
    import Clustering: kmeans



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
            getPerAmp, solve_odeprob, solve_for_fitness_peramp, eval_param_fitness, eval_ic_fitness, eval_all_fitness

    # import the constraint types and tools
    include("api/ConstraintSetTools.jl")
    export ConstraintSet, ConstraintRange, ParameterConstraints, InitialConditionConstraints, AllConstraints, 
            activelength, set_fixed_constraints!, set_fixed_values!, unset_fixed_constraints!, get_fixed_constraintranges

    # import the genetic algorithm and associated functions
    include("api/GA_functions.jl")
    export make_fitness_function_threaded, GAProblem, 
            generate_population, generate_empty_population, generate_population!, logrange,
            GAResults, run_GA

    #include the file utilities
    include("api/FileUtils.jl")
    export read_csvs_in_directory, save_to_csv, make_ga_dataframe, make_pop_dataframe

    #include plotting utilities
    include("api/PlotUtils.jl")
    export plotsol, plotboth, plot_everything, plot_everything_from_csv_indir

    #include clustering utilities
    include("api/ClusteringTools.jl")
    export population_to_matrix, kmeans, df_to_matrix, identify_fixed_columns, get_optimal_clusters, optimal_kmeans_clusters

    #include testing functions
    include("api/TestingFunctions.jl")
    export test4fixedGA

end
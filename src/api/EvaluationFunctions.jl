#< COST FUNCTION HELPER FUNCTIONS ##
"""Get summed difference of the peak values from the FFT of the solution"""
function getDif(peakvals::Vector{Float64})
    # length(peakvals) == 1 ? peakvals[1] : peakvals[1] - peakvals[end]
    (peakvals[begin] - peakvals[end])/length(peakvals)
end

"""Get summed average standard deviation of peaks values from the FFT of the solution"""
function getSTD(fft_peakindxs::Vector{Int}, fft_arrayData; window::Int =1) #get average standard deviation of fft peak indexes
    arrLen = length(fft_arrayData)

    #window = max(1,cld(arrLen,window_ratio)) #* window size is 1% of array length, or 1 if array length is less than 100
    sum_std = sum(std(@view fft_arrayData[max(1, ind - window):min(arrLen, ind + window)]) for ind in fft_peakindxs; init=0.0) #* sum rolling window of standard deviations

    return sum_std / length(fft_peakindxs) #* divide by number of peaks to get average std
end 

"""
    getFrequencies(timeseries)
Return the real-valued FFT of a timeseries, will be half the length of the timeseries
"""
function getFrequencies(timeseries::Vector{Float64}; jump::Int = 2) 
    # rfft_result = rfft(@view timeseries[1:2:end])
    sampled_timeseries = @view timeseries[1:jump:end]
    rfft_result = rfft(sampled_timeseries)
    norm_val = cld(length(timeseries), 2) #* normalize by length of timeseries
    abs.(rfft_result) ./ norm_val
end

"""
    getFrequencies!(fft_array, timeseries)
Computes the real-valued FFT and returns it in-place to the preallocated fft_array, which is half the length of `timeseries`.
"""
function getFrequencies!(fft_array, timeseries::Vector{Float64}; jump::Int = 2) 
    rfft_result = rfft(@view timeseries[1:jump:end])
    norm_val = cld(length(timeseries), 2) #* normalize by length of timeseries
    fft_array .= abs.(rfft_result) ./ norm_val
end

"""Normalize FFT array in-place to have mean 0 and amplitude 1"""
function normalize_time_series!(fftarray)
    mu = mean(fftarray)
    amplitude = maximum(fftarray) - minimum(fftarray)
    fftarray .= (fftarray .- mu) ./ amplitude
end
#> END OF COST FUNCTION HELPER FUNCTIONS ##


#<< PERIOD AND AMPLITUDE FUNCTIONS ##
"""Calculates the period and amplitude of each individual in the population"""
function getPerAmp(sol::OS) where OS <: ODESolution

    Amem_sol = sol[6,:] .+ sol[9,:] .+ sol[10,:] .+ sol[11,:] .+ sol[12,:] .+ sol[15,:] .+ sol[16,:]

    indx_max, vals_max, indx_min, vals_min = findextrema(Amem_sol; min_height=0.1)
    # indx_min, vals_min = findextrema(Amem_sol; height = 0.0, distance = 5, find_maxima=false)
    return getPerAmp(sol.t, indx_max, vals_max, indx_min, vals_min)
end

function getPerAmp(Amem_sol, solt) 

    indx_max, vals_max, indx_min, vals_min = findextrema(Amem_sol; min_height=0.1)
    # indx_min, vals_min = findextrema(Amem_sol; height = 0.0, distance = 5, find_maxima=false)
    return getPerAmp(solt, indx_max, vals_max, indx_min, vals_min)
end

"""Calculates the period and amplitude of each individual in the population"""
function getPerAmp(solt, indx_max::Vector{Int}, vals_max::Vector{Float64}, indx_min::Vector{Int}, vals_min::Vector{Float64})

    #* Calculate amplitudes and periods
    pers = (solt[indx_max[i+1]] - solt[indx_max[i]] for i in 1:(length(indx_max)-1))
    amps = (vals_max[i] - vals_min[i] for i in 1:min(length(indx_max), length(indx_min)))

    return mean(pers), mean(amps) #.|> abs 
end
#> END OF PERIOD AND AMPLITUDE FUNCTIONS ##

#< HEURISTICS ##
"""
    is_steadystate(solu::Vector{Float64}, solt::Vector{Float64})

Checks if the last tenth of the solution array is steady state
"""
function is_steadystate(solu::Vector{Float64}, solt::Vector{Float64})
    tstart = cld(length(solt),10) 

    #* Check if last tenth of the solution array is steady state
    testwindow = solu[end-tstart:end]
    if std(testwindow; mean=mean(testwindow)) < 0.01  
        return true
    else
        return false
    end 
end



#<< FITNESS FUNCTION ##
"""Core fitness function logic to be plugged into eval_fitness wrapper, sums AP2 membrane species before calling FitnessFunction"""
function FitnessFunction(sol::OS) where {OS <: ODESolution}
    Amem_sol = map(sum, sol.u) #* sum all AP2 species on the membrane to get the amplitude of the solution
    FitnessFunction(Amem_sol, sol.t)
end

"""
    FitnessFunction(solu::Vector{Float64}, solt::Vector{Float64})
    
Core fitness function logic that takes in the solution and time array and returns the [fitness, period, amplitude]

# Arguments
-`solu::Vector{Float64}`: The concentration output from the ODESolution (sol.u)
-`solt::Vector{Float64}`: The time data from the ODESolution (sol.t)

# Returns
-[fitness, period, amplitude] Note that a nonzero fitness indicates an oscillatory solution
"""
function FitnessFunction(solu::Vector{Float64}, solt::Vector{Float64})

    #* Check if the solution is steady state
    # if is_steadystate(solu, solt)
    #     return [0.0, 0.0, 0.0]
    # end

    #* Get the indexes of the peaks in the time domain
    indx_max, vals_max, indx_min, vals_min = findextrema(solu; min_height = 0.1)

    #* if there is no signal in the time domain, return 0.0s
    if length(indx_max) < 2 || length(indx_min) < 2 
        return [0.0, 0.0, 0.0]
    end
    
    #* Get the rfft of the solution and normalize it
    fftData = @view solu[1:cld(length(solu),4)] 
    fftData = getFrequencies!(fftData, solu) |> normalize_time_series!

    #* get the indexes of the peaks in the fft
    fft_peakindexes, fft_peakvals = findmaxpeaks(fftData; height = 0.0, distance = 1) 

    #* if there is no signal in the frequency domain, return 0.0s
    if length(fft_peakindexes) < 2 
        return [0.0, 0.0, 0.0]
    else
        #* get the summed standard deviation of the peaks in frequency domain
        standard_deviation = getSTD(fft_peakindexes, fftData) 

        #* get the summed difference between the first and last peaks in frequency domain
        sum_diff = getDif(fft_peakvals) 
    
        #* Compute the period and amplitude
        period, amplitude = getPerAmp(solt, indx_max, vals_max, indx_min, vals_min)
    
        #* add the log of the period to the standard deviation and summed difference to calculate fitness and privelage longer periods
        return [standard_deviation + sum_diff + log(10,period), period, amplitude]
    end
end


function get_fitness!(solu::Vector{Float64})
    #* Get the rfft of the solution and normalize it
    fftData = @view solu[1:cld(length(solu),4)] 
    fftData = getFrequencies!(fftData, solu) |> normalize_time_series!

    #* get the indexes of the peaks in the fft
    fft_peakindexes, fft_peakvals = findmaxpeaks(fftData; height = 0.0, distance = 1) 

    #* get the summed standard deviation of the peaks in frequency domain
    standard_deviation = getSTD(fft_peakindexes, fftData) 

    #* get the summed difference between the first and last peaks in frequency domain
    sum_diff = getDif(fft_peakvals) 

    #* add the log of the period to the standard deviation and summed difference to calculate fitness and privelage longer periods
    return standard_deviation + sum_diff #+ log10(period)
end

function get_std_last10th(solu::Vector{Float64}, solt::Vector{Float64})
    tstart = cld(length(solt),10) 

    #* Test window of last 10% of solution
    testwindow = solu[end-tstart:end]
    
    return std(testwindow; mean=mean(testwindow))
end


function is_oscillatory(solu::Vector{Float64}, solt::Vector{Float64}, max_idxs::Vector{Int}, min_idxs::Vector{Int})
    if !is_steadystate(solu, solt) && length(max_idxs) > 2 && length(min_idxs) > 2
        return true
    else
        return false
    end
end





#< FITNESS FUNCTION CALLERS AND WRAPPERS ## 
# """Evaluate the fitness of an individual with new parameters"""
# function eval_param_fitness(params::Vector{Float64},  prob::OP; idx::Vector{Int} = [6, 9, 10, 11, 12, 15, 16]) where {OP <: ODEProblem}
#     #* remake with new parameters
#     new_prob = remake(prob, p=params)
#     return solve_for_fitness_peramp(new_prob, idx)
# end

# """Evaluate the fitness of an individual with new initial conditions"""
# function eval_ic_fitness(initial_conditions::Vector{Float64}, prob::OP; idx::Vector{Int} = [6, 9, 10, 11, 12, 15, 16]) where {OP <: ODEProblem}
#     #* remake with new initial conditions
#     new_prob = remake(prob, u0=initial_conditions)
#     return solve_for_fitness_peramp(new_prob, idx)
# end

"""Evaluate the fitness of an individual with new initial conditions and new parameters"""
function eval_all_fitness(inputs::Vector{Float64}, prob::OP; idx::Vector{Int} = [6, 9, 10, 11, 12, 15, 16]) where {OP <: ODEProblem}
    newp = @view inputs[1:13]
    newu = @view inputs[14:end]

    # #* holds the non-complexed initial species concentrations L, K, P, A
    # first4u = @view inputs[14:17]

    # #* calculate tspan based on the slowest reaction rate
    # tend = calculate_tspan(newp, first4u)

    #* remake with new initial conditions and new parameters
    new_prob = remake(prob; p = newp, u0= newu)
    return solve_for_fitness_peramp(new_prob, idx)
end

function remake_prob(inputs::Vector{Float64}, prob::OP) where {OP <: ODEProblem}
    newp = @view inputs[1:13]
    newu = @view inputs[14:end]

    #* remake with new initial conditions and new parameters
    return remake(prob; p = newp, u0= newu)
end

"""Takes in an ODEProblem and returns solution"""
function solve_odeprob(prob::OP, idx=[6, 9, 10, 11, 12, 15, 16]) where OP <: ODEProblem
    #* calculate first 10% of the tspan
    tstart = prob.tspan[2] / 10

    #* solve the ODE and only save the last 90% of the solution
    savepoints = tstart:0.1:prob.tspan[2]
    solve(prob, Rosenbrock23(), saveat=savepoints, save_idxs=idx, verbose=false, maxiters=1e6)
end

"""Utility function to call ODE solver and return the fitness and period/amplitude"""
function solve_for_fitness_peramp(prob::OP, idx) where {OP <: ODEProblem}

    sol = solve_odeprob(prob, idx)

    if sol.retcode == ReturnCode.Success
        return FitnessFunction(sol)
    else
        return [0.0, 0.0, 0.0]
    end
end



"""
## Calculate tspan based on the slowest reaction rate.\n
- Simply the reciprocal of the minimum first order rate constants, or the reciprocal of the minimum second order rate constants multiplied by the minimum concentration of the reactants
"""
function calculate_tspan(params, initial_conditions; max_t = 1e4)
    #* Get the minimum rate constant
    min_k, min_k_idx = findmin(params)

    if min_k_idx in (1,4,6,8,10) #* If the minimum rate constant is a second order rate constant, multiply by the minimum concentration of the reactants
        #* Get the minimum concentration of the reactants
        min_conc = minimum(initial_conditions)

        #* Calculate the tspan
        return min(max(10.0, (min_k * min_conc)^-1), max_t)
    else #* If the minimum rate constant is a first order rate constant, simply take the reciprocal
        return min(max(10.0, min_k^-1), max_t)
    end
end
#> END OF FITNESS FUNCTION CALLERS AND WRAPPERS ##


#* change rates associated with which complexes in calculate_tspan 















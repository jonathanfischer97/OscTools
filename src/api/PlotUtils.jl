#< Plotting utilities for testing
"""
Plot the solution from a row of the DataFrame
L(t) K(t) P(t) A(t) Lp(t) LpA(t) LK(t) LpP(t) LpAK(t) LpAP(t) LpAKL(t) LpAPLp(t) AK(t) AP(t) AKL(t) APLp(t)
"""
function plotsol(sol::ODESolution; title = "")

        #* Sum up all A in solution 
        Asol = sol[4,:]  + sol[13,:] + sol[14, :]

        Amem = sol[6,:] + sol[9,:] + sol[10,:]+ sol[11,:] + sol[12,:]+ sol[15,:] + sol[16,:]
        

        # cost, per, amp = CostFunction(sol)
        p = plot(sol, idxs = [1,5,2,3], title = title, xlabel = "Time (s)", ylabel = "Concentration (µM)", lw = 2, size = (1000, 600),
                color = [:blue :orange :purple :gold], label = ["PIP" "PIP2" "PIP5K" "Synaptojanin"], alpha = 0.5)
        # annotate!(p, (0, 0), text("Period = $per\nAmplitude = $amp", :left, 10))
        # p = plot(sol, title=title, alpha = 0.4)
        plot!(p, sol.t, Asol, label="AP2 in solution", ls = :dash, alpha=1.0, color=:gray)
        plot!(p, sol.t, Amem, lw = 2, size = (1000, 600), label = "AP2 on membrane", ls = :dash, alpha=1.0, color=:black)

        # display(p)
        return p    
end



"""
Calculates the frequency per minute of the FFT vector in-place 
"""
function frequencies_per_minute!(t::Vector{Float64}, freq_values::Vector{Float64})
        # Calculate the time step between samples
        dt = t[2] - t[1]
        
        # Calculate the length of the original solution array
        N = 2 * length(freq_values)
        
        # Calculate the frequency step in Hz (per second)
        freq_step_per_second = 1 / (N * dt)
        
        # Convert the frequency step to per minute
        freq_step_per_minute = freq_step_per_second * 60
        
        # Update the frequency values in-place to frequencies per minute
        freq_values .= freq_values .* freq_step_per_minute
end
    
    


"""
Plot the FFT of a solution from a row of the DataFrame
"""
function plotfft(sol::ODESolution)
        #* Trim first 10% of the solution array to avoid initial spikes
        tstart = cld(length(sol.t),10) 
        trimsol = sol[tstart:end] 

        # normsol = normalize_time_series!(trimsol[fitidx,:])
        # normsol = sol[1,:]
        Amem = trimsol[6,:] + trimsol[9,:] + trimsol[10,:]+ trimsol[11,:] + trimsol[12,:]+ trimsol[15,:] + trimsol[16,:]
        solfft = getFrequencies(Amem)

        #* Get the frequencies per minute for x axis
        frequencies_per_minute!(sol.t, solfft)

        #* Normalize the FFT to have mean 0 and amplitude 1
        normalize_time_series!(solfft)
        # fft_peakindexes, fft_peakvals = findmaxima(solfft,1) #* get the indexes of the peaks in the fft
        fft_peakindexes, fft_peakvals = findextrema(solfft; height = 1e-2, distance = 2) #* get the indexes of the peaks in the fft
        
        #* If there are no peaks, return a plot with no peaks
        if isempty(fft_peakindexes)
                p1 = plot(solfft, title = "getDif: 0.0", xlabel = "Frequency (min⁻¹)", ylabel = "Amplitude", lw = 2, 
                                xlims = (0, 100), label="", titlefontsize = 18, titlefontcolor = :green)
                return p1
        else
                window = 1

                diffs = round(getDif(fft_peakvals); digits=4)
                standevs = round(getSTD(fft_peakindexes, solfft; window = window);digits=4)


                p1 = plot(solfft, title = "getDif: $(diffs)", xlabel = "Frequency (min⁻¹)", ylabel = "Amplitude", lw = 2, 
                                xlims = (0, min(length(solfft),fft_peakindexes[end]+50)), ylims=(0.0,min(1.0, maximum(fft_peakvals)+0.25)), label="", titlefontsize = 18, titlefontcolor = :green)
                peaklabels = [text("$(round.(val; digits=4))", :bottom, 10) for val in fft_peakvals]
                scatter!(p1, fft_peakindexes, fft_peakvals, text = peaklabels, label = "", color = :red, markersize = 5)

                maxpeak_idx = fft_peakindexes[argmax(fft_peakvals)]
                stdlines = [maxpeak_idx - window, maxpeak_idx + window]

                
                p2 = plot(solfft, title = "getSTD: $(standevs)", xlabel = "Frequency (min⁻¹)", lw = 2, xlims = (max(0,maxpeak_idx-50), min(length(solfft),maxpeak_idx+50)), 
                                                ylims=(0.0,min(1.0, maximum(fft_peakvals)+0.25)),label="", titlefontsize = 18, titlefontcolor = :red)
                scatter!(p2, fft_peakindexes, fft_peakvals, text = peaklabels, color = :red, markersize = 5, label="")
                vline!(p2, stdlines, color = :blue, label = "")
                
                return plot(p1, p2, layout = (1,2), size = (1000, 600))
        end
end


"""
Plot both the solution and the FFT of a solution from a row of the DataFrame
"""
function plotboth(dfrow::DataFrameRow, prob::ODEProblem; vars::Vector{Int} = collect(1:length(prob.u0)))
        newp = [param for param in dfrow[Between(:ka1, :DF)]]
        newu0 = [ic for ic in dfrow[Between(:L, :A)]]

        reprob = remake(prob, p = newp, u0 = [newu0; zeros(length(prob.u0) - length(newu0))])

        plotboth(reprob; vars = vars)
end

function plotboth(prob::ODEProblem; vars::Vector{Int} = collect(1:length(prob.u0)))
        sol = solve(prob, AutoTsit5(Rodas5P()), saveat=0.1, save_idxs=vars)

        plotboth(sol)
end

function plotboth(sol::ODESolution)

        tstart = cld(length(sol.t),10) 
        trimsol = sol[tstart:end] 

        Amem = trimsol[6,:] + trimsol[9,:] + trimsol[10,:]+ trimsol[11,:] + trimsol[12,:]+ trimsol[15,:] + trimsol[16,:]

        cost, per, amp = CostFunction(Amem, trimsol.t)
        amp_percentage = amp/sol[4,1]

        solplot = plotsol(sol)
        fftplot = plotfft(sol)

        bothplot = plot(solplot, fftplot, plot_title ="Fit: $(round(cost;digits=4))\nPeriod: $(round(per;digits=4)) s\nAmplitude: $(round(amp_percentage;digits=4)) %" , 
                        plot_titlefontsize = 20, layout = (2,1), size = (1000, 800))
        display(bothplot)
        return bothplot
end



"""
Plot the solution and FFT of every row in the DataFrame
"""
function plot_everything(df::DataFrame, prob::ODEProblem; jump=10, path)
        progbar = Progress(cld(nrow(df),jump); desc = "Plotting:")
        # plotpath = mkpath(path*"/Plots")
        # CSV.write(path*"/Set$(setnum)-$(label).csv", df)
    
        for i in 1:jump:nrow(df)
            p = plotboth(df[i,:], prob)
            savefig(p, path*"/plot_$(i).png")
            next!(progbar)
        end
end

"""
Plot everything in the directory
"""
function plot_everything_from_csv_indir(dirpath::String, prob::ODEProblem=make_ODE_problem(); numplots = 100, filenum = 1)
        files = readdir(dirpath; join=true) |> filter(x -> !isdir(x))
        filepath = files[filenum]
        df = CSV.read(filepath, DataFrame)
        plotpath = mkpath(dirpath*"/File$(filenum)_Plots")

        jump = cld(nrow(df), numplots)
        plot_everything(df, prob; jump = jump, path = plotpath)
end

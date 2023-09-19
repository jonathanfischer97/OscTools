"""Full oscillator model"""
function make_fullrn(;defparams = [:ka1 => 0.009433439939827041, :kb1 => 2.3550169939427845, :kcat1 => 832.7213093872278, :ka2 => 12.993995997539924, :kb2 => 6.150972501791291,
                                    :ka3 => 1.3481451097940793, :kb3 => 0.006201726090609513, :ka4 => 0.006277294665474662, :kb4 => 0.9250191811994848, :ka7 => 57.36471615394549, 
                                        :kb7 => 0.04411989797898752, :kcat7 => 42.288085868394326, :DF => 3631.050539219606],
                    defvars = [:L => 3.0, :K => 0.5, :P => 0.3, :A => 2.0, :Lp => 0.0, :LpA => 0.0, :LK => 0.0, 
                        :LpP => 0.0, :LpAK => 0.0, :LpAP => 0.0, :LpAKL => 0.0, :LpAPLp => 0.0, :AK => 0.0, :AP => 0.0, :AKL => 0.0, :APLp => 0.0])

    fullrn = @reaction_network fullrn begin
        @parameters ka1 kb1 kcat1 ka2 kb2 ka3 kb3 ka4 kb4 ka7 kb7 kcat7 DF
        @species L(t) K(t) P(t) A(t) Lp(t) LpA(t) LK(t) LpP(t) LpAK(t) LpAP(t) LpAKL(t) LpAPLp(t) AK(t) AP(t) AKL(t) APLp(t)
        # ALIASES: L = PIP, Lp = PIP2, K = Kinase, P = Phosphatase, A = AP2 
        # reactions between the same binding interfaces will have the same rate constant no matter the dimensionality or complex
        (ka1,kb1), L + K <--> LK # L binding to kinase
        kcat1, LK --> Lp + K # L phosphorylation by kinase into Lp
        (ka2,kb2), Lp + A <--> LpA # Lp binding to AP2 adaptor 
        (ka3,kb3), LpA + K <--> LpAK # Membrane-bound adaptor binding to kinase
        (ka1*DF,kb1), LpAK + L <--> LpAKL # 2D reaction: Membrane-bound kinase binds to L with greater affinity as determined by y (V/A)
        kcat1, LpAKL --> Lp + LpAK # L phosphorylation by kinase into Lp, same as 3D: first order reactions aren't dependent on dimensionality 
        (ka7,kb7), Lp + P <--> LpP # Lp binding to phosphatase 
        kcat7, LpP --> L + P # L dephosphorylation by phosphatase
        (ka4,kb4), LpA + P <--> LpAP # Membrane-bound adaptor binding to phosphatase 
        (ka7*DF,kb7), Lp + LpAP <--> LpAPLp # 2D reaction: Membrane-bound phosphatase binds to Lp with greater affinity as determined by y (V/A)
        kcat7, LpAPLp --> L + LpAP # L dephosphorylation by phosphatase, same as 3D: first order reactions aren't dependent on dimensionality

        #previously excluded reactions, all possible combinations possible in vitro
        (ka2,kb2), Lp + AK <--> LpAK
        (ka2*DF,kb2), Lp + AKL <--> LpAKL
        (ka2,kb2), Lp + AP <--> LpAP
        (ka2*DF,kb2), Lp + APLp <--> LpAPLp
        (ka3,kb3), A + K <--> AK
        (ka4,kb4), A + P <--> AP
        (ka3,kb3), A + LK <--> AKL
        (ka4,kb4), A + LpP <--> APLp
        (ka3*DF,kb3), LpA + LK <--> LpAKL
        (ka4*DF,kb4), LpA + LpP <--> LpAPLp
        (ka1,kb1), AK + L <--> AKL #binding of kinase to lipid
        kcat1, AKL --> Lp + AK #phosphorylation of lipid
        (ka7,kb7), AP + Lp <--> APLp #binding of phosphatase to lipid
        kcat7, APLp --> L + AP #dephosphorylation of lipid
    end  
    setdefaults!(fullrn, [defparams; defvars])
    return fullrn
end

"""Convenience constructor for my normal ODEProblem"""
function make_ODE_problem(tend::Float64=2000.)
    tspan = (0., tend)

    fullrn = make_fullrn()

    ogprob = ODEProblem(fullrn, [], tspan, [])

    # @info typeof(ogprob)

    de = modelingtoolkitize(ogprob)
    # @info typeof(de)

    ODEProblem{true,SciMLBase.FullSpecialize}(de, [], tspan, jac=true)
end


# """Original oscillator model, without all possible pairs of reactions"""
# originalrn = @reaction_network originalrn begin
#     @parameters ka1 kb1 kcat1 ka2 kb2 ka3 kb3 ka4 kb4 ka7 kb7 kcat7 DF
#     @species L(t) K(t) P(t) A(t) Lp(t) LpA(t) LK(t) LpP(t) LpAK(t) LpAP(t) LpAKL(t) LpAPLp(t) 
#     # ALIASES: L = PIP, Lp = PIP2, K = Kinase, P = Phosphatase, A = AP2 
#     # reactions between the same binding interfaces will have the same rate constant no matter the dimensionality or complex
#     (ka1,kb1), L + K <--> LK # L binding to kinase
#     kcat1, LK --> Lp + K # L phosphorylation by kinase into Lp
#     (ka2,kb2), Lp + A <--> LpA # Lp binding to AP2 adaptor
#     (ka3,kb3), LpA + K <--> LpAK # Membrane-bound adaptor binding to kinase
#     (ka1*DF,kb1), LpAK + L <--> LpAKL # 2D reaction: Membrane-bound kinase binds to L with greater affinity as determined by y (V/A)
#     kcat1, LpAKL --> Lp + LpAK # L phosphorylation by kinase into Lp, same as 3D: first order reactions aren't dependent on dimensionality 
#     (ka7,kb7), Lp + P <--> LpP # Lp binding to phosphatase
#     kcat7, LpP --> L + P # L dephosphorylation by phosphatase
#     (ka4,kb4), LpA + P <--> LpAP # Membrane-bound adaptor binding to phosphatase 
#     (ka7*DF,kb7), Lp + LpAP <--> LpAPLp # 2D reaction: Membrane-bound phosphatase binds to Lp with greater affinity as determined by y (V/A)
#     kcat7, LpAPLp --> L + LpAP # L dephosphorylation by phosphatase, same as 3D: first order reactions aren't dependent on dimensionality
# end  



"""Function to print out differential equations of a reaction network in Python format"""
function format_equations(reactionnetwork::ReactionSystem)
    osys = convert(ODESystem, reactionnetwork)
    for ode in osys.eqs
        ode_str = string(ode)
        # ode_str = replace(ode_str, "~" => "=")
        ode_str = replace(ode_str, "(t)" => "")
        ode_str = replace(ode_str, "Differential(" => "d")
        ode_str = replace(ode_str, ") ~" => " =")
        println(ode_str)
    end
end


"""Automatically render the ODEs of a reaction network in LaTeX"""
function getlatex_equations(rn)
    rxnEqs = convert(ODESystem, rn)
    txt = latexify(rxnEqs)
    render(txt)
end


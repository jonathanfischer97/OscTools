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

# """Michaelis-Menten approximation of the original oscillator model, without all possible pairs of reactions"""
# mm_rn = @reaction_network mm_rn begin
#     @parameters Vmax1 Km1 ka2 kb2 ka3 kb3 ka4 kb4 Vmax7 Km7 DF
#     @species L(t) K(t) P(t) A(t) Lp(t) LpA(t) LpAK(t) LpAP(t) 

#     Vmax1*L/(Km1+L), L + K --> Lp + K # L phosphorylation by kinase into Lp using Michaelis-Menten
#     (ka2,kb2), Lp + A <--> LpA # Lp binding to AP2 adaptor
#     (ka3,kb3), LpA + K <--> LpAK # Membrane-bound adaptor binding to kinase
#     Vmax1*DF*L/(Km1+L), LpAK + L --> Lp + LpAK # 2D reaction: Membrane-bound kinase binds to L with greater affinity as determined by y (V/A) using Michaelis-Menten
#     Vmax7*Lp/(Km7+Lp), Lp + P --> L + P # Lp dephosphorylation by phosphatase using Michaelis-Menten
#     (ka4,kb4), LpA + P <--> LpAP # Membrane-bound adaptor binding to phosphatase 
#     Vmax7*DF*Lp/(Km7+Lp), Lp + LpAP --> L + LpAP # 2D reaction: Membrane-bound phosphatase binds to Lp with greater affinity as determined by y (V/A) using Michaelis-Menten
# end

"""Michaelis-Menten approximation of the original oscillator model, without all possible pairs of reactions"""
mm_rn = @reaction_network mm_rn begin
    @parameters kcat1 Km1 ka2 kb2 ka3 kb3 ka4 kb4 kcat7 Km7 DF
    @species L(t) K(t) P(t) A(t) Lp(t) LpA(t) LpAK(t) LpAP(t) 

    (kcat1*K)*L/(Km1+L), L + K --> Lp + K # L phosphorylation by kinase into Lp using Michaelis-Menten
    (ka2,kb2), Lp + A <--> LpA # Lp binding to AP2 adaptor
    (ka3,kb3), LpA + K <--> LpAK # Membrane-bound adaptor binding to kinase
    (kcat1*K*DF)*L/(Km1+L), LpAK + L --> Lp + LpAK # 2D reaction: Membrane-bound kinase binds to L with greater affinity as determined by y (V/A) using Michaelis-Menten
    (kcat7*P)*Lp/(Km7+Lp), Lp + P --> L + P # Lp dephosphorylation by phosphatase using Michaelis-Menten
    (ka4,kb4), LpA + P <--> LpAP # Membrane-bound adaptor binding to phosphatase 
    (kcat7*P*DF)*Lp/(Km7+Lp), Lp + LpAP --> L + LpAP # 2D reaction: Membrane-bound phosphatase binds to Lp with greater affinity as determined by y (V/A) using Michaelis-Menten
end

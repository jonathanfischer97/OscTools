using DifferentialEquations
using StaticArrays
using BenchmarkTools, Profile
using ModelingToolkit

function fullmodel_ode!(du, u, p, t)
    L, K, P, A, Lp, LpA, LK, LpP, LpAK, LpAP, LpAKL, LpAPLp, AK, AP, AKL, APLp = u #initial conditions
    ka1, kb1, kcat1, ka2, kb2, ka3, kb3, ka4, kb4, ka7, kb7, kcat7, y = p #parameters

    du[1] = @fastmath kb1*AKL + kcat7*APLp + kb1*LK + kb1*LpAKL + kcat7*LpAPLp + kcat7*LpP - ka1*AK*L - ka1*K*L - ka1*y*L*LpAK #* L
    du[2] = @fastmath kb1*LK + kb3*AK + kcat1*LK + kb3*LpAK - ka3*A*K - ka1*K*L - ka3*K*LpA #* K
    du[3] = @fastmath kb4*AP + kb4*LpAP + kb7*LpP + kcat7*LpP - ka4*A*P - ka7*Lp*P - ka4*LpA*P #* P
    du[4] = @fastmath kb2*LpA + kb3*AK + kb3*AKL + kb4*AP + kb4*APLp - ka2*A*Lp - ka3*A*K - ka3*A*LK - ka4*A*LpP - ka4*A*P #* A
    du[5] = @fastmath kb7*APLp + kcat1*AKL + kcat1*LK + kb2*LpA + kb2*LpAK + kb2*LpAKL + kb2*LpAP + kcat1*LpAKL + kb2*LpAPLp + kb7*LpAPLp + kb7*LpP - ka2*A*Lp - ka2*AK*Lp - ka2*AP*Lp - ka7*AP*Lp - ka7*Lp*P - ka2*y*AKL*Lp - ka2*y*APLp*Lp - ka7*y*Lp*LpAP #* Lp
    du[6] = @fastmath kb3*LpAK + kb3*LpAKL + kb4*LpAP + kb4*LpAPLp + ka2*A*Lp - kb2*LpA - ka3*K*LpA - ka4*LpA*P - ka3*y*LK*LpA - ka4*y*LpA*LpP #* LpA
    du[7] = @fastmath kb3*AKL + kb3*LpAKL + ka1*K*L - kb1*LK - kcat1*LK - ka3*A*LK - ka3*y*LK*LpA #* LK
    du[8] = @fastmath kb4*APLp + kb4*LpAPLp + ka7*Lp*P - kb7*LpP - kcat7*LpP - ka4*A*LpP - ka4*y*LpA*LpP #* LpP
    du[9] = @fastmath kb1*LpAKL + kcat1*LpAKL + ka2*AK*Lp + ka3*K*LpA - kb2*LpAK - kb3*LpAK - ka1*y*L*LpAK #* LpAK
    du[10] = @fastmath kb7*LpAPLp + kcat7*LpAPLp + ka2*AP*Lp + ka4*LpA*P - kb2*LpAP - kb4*LpAP - ka7*y*Lp*LpAP #* LpAP
    du[11] = @fastmath ka1*y*L*LpAK + ka2*y*AKL*Lp + ka3*y*LK*LpA - kb1*LpAKL - kb2*LpAKL - kb3*LpAKL - kcat1*LpAKL #* LpAKL
    du[12] = @fastmath ka2*y*APLp*Lp + ka7*y*Lp*LpAP + ka4*y*LpA*LpP - kb2*LpAPLp - kb4*LpAPLp - kb7*LpAPLp - kcat7*LpAPLp #* LpAPLp
    du[13] = @fastmath kb1*AKL + kb2*LpAK + kcat1*AKL + ka3*A*K - kb3*AK - ka1*AK*L - ka2*AK*Lp #* AK
    du[14] = @fastmath kb2*LpAP + kb7*APLp + kcat7*APLp + ka4*A*P - kb4*AP - ka2*AP*Lp - ka7*AP*Lp #* AP
    du[15] = @fastmath kb2*LpAKL + ka3*A*LK + ka1*AK*L - kb1*AKL - kb3*AKL - kcat1*AKL - ka2*y*AKL*Lp #* AKL
    du[16] = @fastmath kb2*LpAPLp + ka7*AP*Lp + ka4*A*LpP - kb4*APLp - kb7*APLp - kcat7*APLp - ka2*y*APLp*Lp #* APLp
    nothing
end



#* Out of place, with StaticArrays
function fullmodel(u, p, t)
        L, K, P, A, Lp, LpA, LK, LpP, LpAK, LpAP, LpAKL, LpAPLp, AK, AP, AKL, APLp = u #initial conditions
        ka1, kb1, kcat1, ka2, kb2, ka3, kb3, ka4, kb4, ka7, kb7, kcat7, y = p #parameters
        dL = kb1*AKL + kcat7*APLp + kb1*LK + kb1*LpAKL + kcat7*LpAPLp + kcat7*LpP - ka1*AK*L - ka1*K*L - ka1*y*L*LpAK #* L
        dK = kb1*LK + kb3*AK + kcat1*LK + kb3*LpAK - ka3*A*K - ka1*K*L - ka3*K*LpA #* K
        dP = kb4*AP + kb4*LpAP + kb7*LpP + kcat7*LpP - ka4*A*P - ka7*Lp*P - ka4*LpA*P #* P
        dA = kb2*LpA + kb3*AK + kb3*AKL + kb4*AP + kb4*APLp - ka2*A*Lp - ka3*A*K - ka3*A*LK - ka4*A*LpP - ka4*A*P #* A
        dLp = kb7*APLp + kcat1*AKL + kcat1*LK + kb2*LpA + kb2*LpAK + kb2*LpAKL + kb2*LpAP + kcat1*LpAKL + kb2*LpAPLp + kb7*LpAPLp + kb7*LpP - ka2*A*Lp - ka2*AK*Lp - ka2*AP*Lp - ka7*AP*Lp - ka7*Lp*P - ka2*y*AKL*Lp - ka2*y*APLp*Lp - ka7*y*Lp*LpAP #* Lp
        dLpA = kb3*LpAK + kb3*LpAKL + kb4*LpAP + kb4*LpAPLp + ka2*A*Lp - kb2*LpA - ka3*K*LpA - ka4*LpA*P - ka3*y*LK*LpA - ka4*y*LpA*LpP #* LpA
        dLK = kb3*AKL + kb3*LpAKL + ka1*K*L - kb1*LK - kcat1*LK - ka3*A*LK - ka3*y*LK*LpA #* LK
        dLpP = kb4*APLp + kb4*LpAPLp + ka7*Lp*P - kb7*LpP - kcat7*LpP - ka4*A*LpP - ka4*y*LpA*LpP #* LpP
        dLpAK = kb1*LpAKL + kcat1*LpAKL + ka2*AK*Lp + ka3*K*LpA - kb2*LpAK - kb3*LpAK - ka1*y*L*LpAK #* LpAK
        dLpAP = kb7*LpAPLp + kcat7*LpAPLp + ka2*AP*Lp + ka4*LpA*P - kb2*LpAP - kb4*LpAP - ka7*y*Lp*LpAP #* LpAP
        dLpAKL = ka1*y*L*LpAK + ka2*y*AKL*Lp + ka3*y*LK*LpA - kb1*LpAKL - kb2*LpAKL - kb3*LpAKL - kcat1*LpAKL #* LpAKL
        dLpAPLp = ka2*y*APLp*Lp + ka7*y*Lp*LpAP + ka4*y*LpA*LpP - kb2*LpAPLp - kb4*LpAPLp - kb7*LpAPLp - kcat7*LpAPLp #* LpAPLp
        dAK = kb1*AKL + kb2*LpAK + kcat1*AKL + ka3*A*K - kb3*AK - ka1*AK*L - ka2*AK*Lp #* AK
        dAP = kb2*LpAP + kb7*APLp + kcat7*APLp + ka4*A*P - kb4*AP - ka2*AP*Lp - ka7*AP*Lp #* AP
        dAKL = kb2*LpAKL + ka3*A*LK + ka1*AK*L - kb1*AKL - kb3*AKL - kcat1*AKL - ka2*y*AKL*Lp #* AKL
        dAPLp = kb2*LpAPLp + ka7*AP*Lp + ka4*A*LpP - kb4*APLp - kb7*APLp - kcat7*APLp - ka2*y*APLp*Lp #* APLp
        SA[dL, dK, dP, dA, dLp, dLpA, dLK, dLpP, dLpAK, dLpAP, dLpAKL, dLpAPLp, dAK, dAP, dAKL, dAPLp]
end



#? Parameter list
psym = [:ka1 => 0.009433439939827041, :kb1 => 2.3550169939427845, :kcat1 => 832.7213093872278, :ka2 => 12.993995997539924, :kb2 => 6.150972501791291,
        :ka3 => 1.3481451097940793, :kb3 => 0.006201726090609513, :ka4 => 0.006277294665474662, :kb4 => 0.9250191811994848, :ka7 => 57.36471615394549, 
        :kb7 => 0.04411989797898752, :kcat7 => 42.288085868394326, :DF => 3631.050539219606]
p = [x[2] for x in psym]
    
#? Initial condition list
usym = [:L => 3.0, :K => 0.5, :P => 0.3, :A => 2.0, :Lp => 0.0, :LpA => 0.0, :LK => 0.0, 
        :LpP => 0.0, :LpAK => 0.0, :LpAP => 0.0, :LpAKL => 0.0, :LpAPLp => 0.0, :AK => 0.0, :AP => 0.0, 
        :AKL => 0.0, :APLp => 0.0]
u0 = [x[2] for x in usym]

pstatic = SA[0.009433439939827041, 2.3550169939427845, 832.7213093872278, 12.993995997539924, 6.150972501791291, 
                1.3481451097940793, 0.006201726090609513, 0.006277294665474662, 0.9250191811994848, 57.36471615394549, 0.04411989797898752, 42.288085868394326, 3631.050539219606]
u0static = SA[0.0, 0.5, 0.3, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#? Timespan for integration
tspan = (0., 1000.)

inplaceprob = ODEProblem(fullmodel_ode!, u0, tspan, p)
fullrn = make_fullrn()
ogprob = ODEProblem(fullrn, u0, tspan, p)
oopprob = ODEProblem(fullmodel, u0static, tspan, pstatic)

@btime solve($inplaceprob, Rosenbrock23(), saveat=0.1)
@btime solve($ogprob, Rosenbrock23(), saveat=0.1)
@btime solve($oopprob, Rosenbrock23(),saveat=0.1)




@code_warntype fullmodel_static(u0static, pstatic, tspan)

osys = convert(ODESystem, fullrn)
for eq in osys.eqs
        @show eq
end


#* Testing auto jacobian 

inplacede = modelingtoolkitize(inplaceprob)
ogde = modelingtoolkitize(ogprob)
oopde = modelingtoolkitize(oopprob)

inplaceprobjac = ODEProblem(inplacede, [], tspan, jac=true)
ogprobjac = ODEProblem(ogde, [], tspan, jac=true)
oopprobjac = ODEProblem(oopde, [], tspan, jac=true)

@btime solve($inplaceprobjac, Rosenbrock23(), saveat=0.1)
@btime solve($ogprobjac, Rosenbrock23(), saveat=0.1)            
@btime solve($oopprobjac, Rosenbrock23(), saveat=0.1)



function lorenz_static(u, p, t)
        dx = 10.0 * (u[2] - u[1])
        dy = u[1] * (28.0 - u[3]) - u[2]
        dz = u[1] * u[2] - (8 / 3) * u[3]
        SA[dx, dy, dz]
    end
    
u0 = SA[1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz_static, u0, tspan)
@btime solve(prob, Tsit5());
@btime solve(prob, Tsit5(), save_everystep = false);

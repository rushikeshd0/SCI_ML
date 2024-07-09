using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots


# ρ = 1410 Kg/m^3 
# r_sun = 695700 Km
# Stellar structure equation (mass continuity) : dM/dr = 4 π r^2 ρ

ρ = 1
u0 = Float32[0.0 ] # R centre  = 0
rspan = (0.0f0, 1.0f0) # 
# datasize = 160
datasize = 30
rsteps = range(rspan[1], rspan[2]; length = datasize)


# Ground truth data
M_true = 4/3 * π * ρ * rsteps .^ 3


rng = Random.default_rng()
# function stellar!(dm, m, p, r)
#     dm[1] =  4*pi*(m[1]^2)*ρ
    
# end

# prob_trueode = ODEProblem(stellar!, u0, rspan)

# ## Generating the ground truth data
# ode_data = Array(solve(prob_trueode, Tsit5(); saveat = tsteps))

plot(M_true)

# dudt2 = Lux.Chain( Lux.Dense(1, 30, tanh), Lux.Dense(30, 1))
dudt2 = Lux.Chain(Lux.Dense(1, 30, tanh),Lux.Dense(30, 30, relu),Lux.Dense(30, 1))


p, st = Lux.setup(rng, dudt2)

prob_neuralode = NeuralODE(dudt2, rspan, Tsit5(); saveat = rsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

### Define loss function as the difference between actual ground truth data and Neural ODE prediction
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, M_true .- pred)
    return loss, pred
end


callback1 = function (p, l, pred; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(rsteps, M_true[1, :]; label = "data")
        scatter!(plt, rsteps, pred[1, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end


pinit = ComponentArray(p)
callback1(pinit, loss_neuralode(pinit)...; doplot = true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = solve(optprob, OptimizationOptimisers.Adam(0.01); callback = callback1,
    maxiters = 300)


optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.02);
callback=callback1, allow_f_increases = false)

callback1(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)
data_pred = predict_neuralode(result_neuralode2.u)
plot(legend=:topright)


# data_pred = predict_neuralode(result_neuralode2.u)
# plot()

bar!(M_true[1,:], alpha = 0.3, label=" mass_true",title = "stellar structure in  NN ODE",
xlabel = "radius", ylabel = "mass")

plot!(data_pred[1,:], lw=3, label=" mass_pred",title = "stellar structure in  NN ODE",
xlabel = "radius", ylabel = "mass")





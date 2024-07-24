# using DifferentialEquations, Plots

using DifferentialEquations,ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots,Sundials


# Define the ODE function
function stellar!(dm, m, p, r)
    dm[1] = 4 * π * r^2 * ρ
end

# Initial conditions and problem setup
u0 = [0.0]  # R center = 0
ρ = 1.0
rspan = (0.0, 1.0)
datasize = 10

rsteps = range(rspan[1], rspan[2]; length = datasize)

prob_trueode = ODEProblem(stellar!, u0, rspan)

ode_data = Array(solve(prob_trueode, Tsit5(); saveat = rsteps))

plot(ode_data')



# # Solving the ODE
# sol = solve(prob_trueode, Tsit5(); saveat=rsteps)

# # Plotting the results
# plot(rsteps, sol, label="True Mass", xlabel="Radius", ylabel="Mass", title="Stellar Structure")
# plot!(sol.t, sol.u, label="Numerical Solution")

rng = Random.default_rng()


# dudt2 = Lux.Chain(Lux.Dense(1, 20, tanh),Lux.Dense(20, 20, tanh),Lux.Dense(20, 20, tanh),Lux.Dense(20, 1))
dudt2 = Lux.Chain( Lux.Dense(1, 30, tanh), Lux.Dense(30, 1))
# dudt2 = Lux.Chain(Lux.Dense(1, 50, tanh),Lux.Dense(50, 50, relu),Lux.Dense(50, 1))



p, st = Lux.setup(rng, dudt2)

# prob_neuralode = NeuralODE(dudt2, rspan, KenCarp4(); saveat = rsteps)
prob_neuralode = NeuralODE(dudt2, rspan, CVODE_BDF(); saveat = rsteps,reltol=1e-8, abstol=1e-10)


function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

### Define loss function as the difference between actual ground truth data and Neural ODE prediction
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end


callback1 = function (p, l, pred; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(rsteps, ode_data[1, :]; label = "data")
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
    maxiters = 400)
# result_neuralode = solve(optprob, Optimisers.RMSProp(0.01, 0.99); callback = callback1,
#     maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
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





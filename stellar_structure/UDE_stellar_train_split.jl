# using DifferentialEquations, Plots

using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots,ComponentArrays,OptimizationOptimisers

    

# Define the ODE function
function stellar!(dm, m, p, r)
    
    dm[1] = 4 * π * r^2 * ρ
end

# Initial conditions and problem setup
u0 = [0.0]  # R center = 0
ρ = 1.0
rspan = (0.0, 1.0)
datasize = 30

rsteps = range(rspan[1], rspan[2]; length = datasize)

prob_trueode = ODEProblem(stellar!, u0, rspan)

ode_data = Array(solve(prob_trueode, Tsit5(); saveat = rsteps))


plot(ode_data')

train_percent = 0.1


rng = Random.default_rng()
p0_vec = []

# NN1 = Lux.Chain(Lux.Dense(1,10,relu),Lux.Dense(10,1)) #
NN1 = Lux.Chain(Lux.Dense(1,30,relu),Lux.Dense(30,1)) #


p1, st1 = Lux.setup(rng, NN1)

# p0_vec = (layer_1 = p1)
# p0_vec = ComponentArray(p0_vec)
p0_vec = ComponentArray(layer_1 = p1)

function dmdr_pred(du, u, p, r)
    (r) = u

    # NNρ = abs(NN1([r], p.layer_1, p.state_1)[1][1])
    NNρ = abs(NN1(ComponentArray(r), p.layer_1, st1)[1][1])

    # du[1] = dx =  4*π*NNρ 
    du[1] =  4*π*NNρ 
    
end


α = p0_vec

prob_data = ODEProblem{true}(dmdr_pred,u0,rspan)

function predict_adjoint(θ)
    pred = Array(solve(prob_data,Tsit5(),p=θ,saveat=rsteps,
                  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# according to train percent split the ground truth data
train_datasize = Int(datasize * train_percent)
test_datasize = datasize - train_datasize

train_rsteps = rsteps[1:train_datasize]
test_rsteps = rsteps[train_datasize+1:end]

train_ode_data = ode_data[:, 1:train_datasize]
test_ode_data = ode_data[:, train_datasize+1:end]

println("train_ode_data size: ", size(train_ode_data))


##  loss function
function loss_adjoint(θ)
    pred = predict_adjoint(θ)
    
    train_pred = pred[:,1:train_datasize]
    loss = sum(abs2, train_ode_data .- train_pred)
    return loss
  end
  
  iter = 0
  function callback2(θ,l)
    global iter
    iter += 1
    if iter%100 == 0
      println(l)
    end
    return false
  end

schedule = (iter, opt) -> (iter < 5000 ? opt.eta : opt.eta * 0.1)



###If the loss is stagnant or shoots to a very high value, the optimizer is stuck in a minima. 
###To avoid this, you will need to run the code again so that the neural networks are initialized differently.
### We will have a tutorial on neural network robustness, to ensure that a large number of 
### initializations actually converge.
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, α)
res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.005),schedule=schedule, callback = callback2, maxiters = 3000)
# res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.005),schedule=schedule, callback = callback2, maxiters = 15000)



# t1 = range(tspan[1],tspan[2],step=0.1)

####################

# iter = 0
# function callback3(θ,l)
#   global iter
#   iter += 1
#   if iter%5 == 0
#     println(l)
#   end
#   return false
# end


# do not use this
# optprob2 = remake(optprob;u0 =  res1.u)
# res2 = Optimization.solve(optprob2, Optim.BFGS( ;initial_stepnorm = 0.01), callback = callback3, maxiters = 5000)
# # println("Final training loss after $(length(losses)) iterations: $(losses[end])")




# Visualizing the predictions
data_pred = predict_adjoint(res1.u)
# plot( legend=:topleft)

# bar!(t,prey_Data, label="prey data", color=:red, alpha=0.5)
# bar!(t, pred_Data, label="pred data", color=:blue, alpha=0.5)

# plot!(t, data_pred[1,:], label = "prey prediction")
# plot!(t, data_pred[2,:],label = "pred prediction")



# Visualizing the predictions



# data_pred = predict_adjoint(res2.u)
plot( )

plt = scatter(train_rsteps, ode_data[1, 1:train_datasize]; label = "Ground truth data",color = "blue",title = "Mass with respect to radius",xlabel = "radius",ylabel = "mass")
scatter!(test_rsteps, ode_data[1, train_datasize + 1:end]; label = "Ground truth data",color = "red",title = "Mass with respect to radius",xlabel = "radius",ylabel = "mass")
# scatter!(plt, train_rsteps, pred[:,1:train_datasize]; label = "train data")
plot!(plt, train_rsteps, data_pred[1, 1:train_datasize]; label = "UDE Training prediction",color = "blue",title = "Mass with respect to radius",xlabel = "radius",ylabel = "mass")
plot!(plt, test_rsteps, data_pred[1, train_datasize + 1:end]; label = "UDE Forecast prediction",color = "red",title = "Mass with respect to radius",xlabel = "radius",ylabel = "mass")



using Statistics
# Mean Squared Error
mse = mean((data_pred .- sol).^2)


# Discussion
# The fitted model can be seen to give a good fit both to the training data.
# The training iterations also proceed very fast. We will also have a tutorial to 
# cover forecasting and why UDE forecasting is much better than Neural ODE forecasting.s



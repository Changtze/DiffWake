push!(LOAD_PATH, @__DIR__)
using Enzyme
using DiffWakeJulia

# Load typical run
data_dir = joinpath(@__DIR__, "..", "..", "data", "horn")
cfg = load_input(joinpath(data_dir, "gch.yaml"), joinpath(data_dir, "vestas_v802MW.yaml"))
cfg = set_config(cfg, wind_speeds=[8.0], wind_directions=[0.0], turbulence_intensities=[0.06])
state = create_state(cfg)

# Define objective
function objective_wrapper(yaw::Matrix{Float64})
    new_state = alter_yaw_angles(yaw, create_state(cfg))
    simulate(new_state)
    powers = turbine_powers(new_state)
    return sum(powers)
end

yaw = fill(0.0, 1, cfg.farm["n_turbines"])
grad_yaw = fill(0.0, size(yaw))

println("Objective value: ", objective_wrapper(yaw))
println("Running Enzyme...")

try
    Enzyme.autodiff(Reverse, objective_wrapper, Active, Duplicated(yaw, grad_yaw))
    println("Gradient success! ", grad_yaw)
catch e
    println("Enzyme failed: ", e)
    Base.showerror(stdout, e, catch_backtrace())
end

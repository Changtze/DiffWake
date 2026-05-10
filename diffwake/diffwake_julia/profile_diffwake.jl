using Profile
using BenchmarkTools
using DiffWakeJulia
using Printf

# Hardcoded test paths for quick profiling
data_dir = joinpath(@__DIR__, "..", "data", "horn")
farm_yaml = joinpath(data_dir, "gch.yaml")
generator_yaml = joinpath(data_dir, "vestas_v802MW.yaml")

println("Loading config...")
cfg = load_input(farm_yaml, generator_yaml)
# Set a single wind speed/dir for profiling
cfg = set_config(cfg, wind_speeds=[8.0], wind_directions=[0.0], turbulence_intensities=[0.06])

println("Creating state...")
state = create_state(cfg)

println("JIT compiling (first run)...")
result = simulate(state)
println("Mean power: ", sum(turbine_powers(state)))

println("\n--- Benchmarking Simulator ---")
@btime simulate($state)

println("\n--- Profiling Simulator ---")
Profile.clear()
@profile for _ in 1:100
    simulate(state)
end
Profile.print(format=:flat, sortedby=:count, mincount=10)

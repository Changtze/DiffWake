"""
    DiffWakeJulia

Julia port of the DiffWake wind-farm wake solver (originally JAX-based).

Public API:
- `load_input(farm_path, gen_path)` → `Config`
- `create_state(cfg)` → `State`
- `simulate(state)` → `Result`
- `turbine_powers(state)` → power array
- `alter_yaw_angles(yaw, state)` → new `State`
- `set_config(cfg; ...)` → new `Config`
"""
module DiffWakeJulia

using LinearAlgebra
using Statistics
using YAML

export
    # Config / IO
    load_input, load_yaml, set_config, Config,
    # State construction
    create_state, create_turbine_grid, create_farm, create_flow_field_from_config,
    # Simulation
    simulate, turbine_powers, alter_yaw_angles,
    # Types
    State, Result, Params, DynamicState,
    TurbineGrid, FlowField, Farm, WakeModelManager, Turbine,
    # Wake models
    GaussVelocityDeficit, GaussVelocityDeflection, CrespoHernandez, SOSFS,
    # Turbine operation
    cosine_loss_thrust_coefficient, cosine_loss_axial_induction, op_power

# ── Include order matters: dependencies before dependents ─────────────────────

# 1. Interpolation (no deps)
include("interp1d.jl")

# 2. Grid (no deps besides stdlib)
include("grid.jl")

# 3. Flow field (needs TurbineGrid, _take_along_axis2)
include("flow_field.jl")

# 4. Wake models (self-contained)
include("wake_models/sosfs.jl")
include("wake_models/crespo_hernandez.jl")
include("wake_models/gauss_velocity.jl")
include("wake_models/gauss_deflection.jl")

# 5. Turbine models (needs interp1d)
include("turbine/operation_models.jl")
include("turbine/turbine.jl")

# 6. Farm (needs Turbine)
include("farm.jl")

# 7. Wake manager
include("wake.jl")

# 8. Utilities / core types (needs all above)
include("util.jl")

# 9. Solver (needs types + wake models)
include("solver.jl")

# 10. Simulator (needs solver + types)
include("simulator.jl")

# 11. Model factories (needs everything)
include("model.jl")

end # module

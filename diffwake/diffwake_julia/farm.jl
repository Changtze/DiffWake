const POWER_SETPOINT_DEFAULT = 1.0e12

"""
    Farm

Wind-farm layout and turbine property container.  Mirrors the Flax `Farm`
dataclass from `farm.py`.
"""
Base.@kwdef mutable struct Farm
    layout_x::Vector{Float64}
    layout_y::Vector{Float64}
    n_turbines::Int

    # Turbine properties (scalars, extracted from Turbine)
    hub_height::Float64
    rotor_diameter::Float64
    TSR::Float64
    ref_tilt::Float64
    correct_cp_ct_for_tilt::Bool

    # Function handles
    power_function::Function
    thrust_coefficient_function::Function
    axial_induction_function::Function
    tilt_interp::Union{Nothing, Function}

    power_thrust_table::Dict{String, Any}

    # Turbine struct reference
    turbine::Any

    # Sorted / unsorted angles  (B, T)
    yaw_angles::Union{Nothing, Matrix{Float64}}              = nothing
    tilt_angles::Union{Nothing, Matrix{Float64}}             = nothing
    power_setpoints::Union{Nothing, Matrix{Float64}}         = nothing

    yaw_angles_sorted::Union{Nothing, Matrix{Float64}}       = nothing
    tilt_angles_sorted::Union{Nothing, Matrix{Float64}}      = nothing
    power_setpoints_sorted::Union{Nothing, Matrix{Float64}}  = nothing
end

"""
    create_farm(layout_x, layout_y, turbine_type_dict, turbine_class; yaw_angles=nothing)

Factory — mirrors `Farm.create(...)`.
"""
function create_farm(
    layout_x::Vector{Float64},
    layout_y::Vector{Float64},
    turbine_type_dict::Dict,
    turbine_constructor::Function;
    yaw_angles::Union{Nothing, Matrix{Float64}} = nothing,
)
    turbine = turbine_constructor(turbine_type_dict)

    return Farm(
        layout_x          = layout_x,
        layout_y          = layout_y,
        n_turbines        = length(layout_x),
        hub_height        = turbine.hub_height,
        rotor_diameter    = turbine.rotor_diameter,
        TSR               = turbine.TSR,
        ref_tilt          = turbine.power_thrust_table["ref_tilt"],
        correct_cp_ct_for_tilt = turbine.correct_cp_ct_for_tilt,
        power_function              = turbine.power_function,
        thrust_coefficient_function = turbine.thrust_coefficient_function,
        axial_induction_function    = turbine.axial_induction_function,
        tilt_interp                 = turbine.tilt_interp,
        power_thrust_table          = turbine.power_thrust_table,
        turbine                     = turbine,
        yaw_angles                  = yaw_angles,
    )
end

"""
    initialize_farm!(farm, sorted_indices)

Sort yaw/tilt/power-setpoints by the upstream→downstream ordering in
`sorted_indices` (B, T, Ny, Nz).
"""
function initialize_farm!(farm::Farm, sorted_indices::Array{Int,4})
    B, T, _, _ = size(sorted_indices)
    idx = sorted_indices[:, :, 1, 1]   # (B, T)

    yaw  = farm.yaw_angles  === nothing ? zeros(B, T) : farm.yaw_angles
    tilt = farm.tilt_angles === nothing ? zeros(B, T) : farm.tilt_angles

    yaw_s  = _take_along_axis2_mat(yaw,  idx)
    tilt_s = _take_along_axis2_mat(tilt, idx)

    pset_s = if farm.power_setpoints !== nothing
        _take_along_axis2_mat(farm.power_setpoints, idx)
    else
        nothing
    end

    farm.yaw_angles         = yaw
    farm.tilt_angles        = tilt
    farm.yaw_angles_sorted  = yaw_s
    farm.tilt_angles_sorted = tilt_s
    farm.power_setpoints_sorted = pset_s

    return farm
end

# ── helper: gather along axis 2 for matrices ─────────────────────────────────
function _take_along_axis2_mat(src::Matrix{Float64}, idx::Matrix{Int})
    B, T = size(src)
    out = similar(src)
    for b in 1:B, t in 1:T
        out[b, t] = src[b, idx[b, t]]
    end
    return out
end

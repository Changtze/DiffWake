# ─── Turbine struct & top-level power / thrust / axial_induction ─────────────
# Ported from turbine/turbine.py

"""
    Turbine

Stores per-turbine-type properties and function handles for power, thrust, and
axial induction computation.
"""
Base.@kwdef struct Turbine
    turbine_type::String
    operation_model::String
    rotor_diameter::Float64
    hub_height::Float64
    TSR::Float64

    power_thrust_table::Dict{String, Any}
    correct_cp_ct_for_tilt::Bool = false
    floating_tilt_table::Union{Nothing, Dict} = nothing

    power_function::Function              = op_power
    thrust_coefficient_function::Function  = cosine_loss_thrust_coefficient
    axial_induction_function::Function     = cosine_loss_axial_induction
    tilt_interp::Union{Nothing, Function}  = nothing
end

# Convenience constructor from a Dict (YAML-loaded generator file)
function Turbine(d::Dict)
    ptt = d["power_thrust_table"]
    # Ensure numeric arrays
    for k in ("wind_speed", "power", "thrust_coefficient")
        if haskey(ptt, k) && ptt[k] isa AbstractVector
            ptt[k] = Float64.(ptt[k])
        end
    end
    return Turbine(
        turbine_type        = get(d, "turbine_type", "default"),
        operation_model     = get(d, "operation_model", "cosine-loss"),
        rotor_diameter      = Float64(d["rotor_diameter"]),
        hub_height          = Float64(d["hub_height"]),
        TSR                 = Float64(d["TSR"]),
        power_thrust_table  = ptt,
        correct_cp_ct_for_tilt = get(d, "correct_cp_ct_for_tilt", false),
    )
end

rotor_radius(t::Turbine) = t.rotor_diameter / 2.0
rotor_area(t::Turbine)   = π * rotor_radius(t)^2


# ─── Top-level dispatchers ───────────────────────────────────────────────────

"""
    turbine_power(velocities, air_density, power_function, yaw_angles, tilt_angles, power_thrust_table; cubature_weights=nothing)

Compute power for a single turbine type.
"""
function turbine_power(;
    velocities,
    air_density,
    power_function,
    yaw_angles,
    tilt_angles,
    power_thrust_table,
    cubature_weights = nothing,
)
    return power_function(;
        power_thrust_table,
        velocities,
        air_density,
        yaw_angles,
        tilt_angles,
        cubature_weights,
    )
end


"""
    turbine_thrust_coefficient(velocities, yaw_angles, tilt_angles, thrust_fn, ...; ix_filter=nothing)

Compute thrust coefficient, optionally filtering to a single turbine index.
"""
function turbine_thrust_coefficient(;
    velocities,
    yaw_angles,
    tilt_angles,
    thrust_fn,
    tilt_interp,
    power_thrust_table,
    correct_cp_ct_for_tilt = false,
    ix_filter = nothing,
    cubature_weights = nothing,
)
    if ix_filter !== nothing
        velocities  = velocities[:, ix_filter:ix_filter, :, :]
        yaw_angles  = yaw_angles[:, ix_filter:ix_filter]
        tilt_angles = tilt_angles[:, ix_filter:ix_filter]
    end
    return thrust_fn(;
        power_thrust_table,
        velocities,
        yaw_angles,
        tilt_angles,
        tilt_interp,
        cubature_weights,
        correct_cp_ct_for_tilt,
    )
end


"""
    turbine_axial_induction(velocities, yaw_angles, tilt_angles, ...; ix_filter=nothing)

Compute axial induction, optionally filtering to a single turbine index.
"""
function turbine_axial_induction(;
    velocities,
    yaw_angles,
    tilt_angles,
    axial_induction_function,
    tilt_interp,
    turbine_power_thrust_table,
    correct_cp_ct_for_tilt = false,
    ix_filter = nothing,
    cubature_weights = nothing,
    multidim_condition = nothing,
)
    if ix_filter !== nothing
        idx = ix_filter
        velocities  = velocities[:, idx:idx, :, :]
        yaw_angles  = yaw_angles[:, idx:idx]
        tilt_angles = tilt_angles[:, idx:idx]
    end

    # Select the correct power/thrust table
    if haskey(turbine_power_thrust_table, "thrust_coefficient")
        power_thrust_table = turbine_power_thrust_table
    else
        if multidim_condition === nothing
            error("multidim_condition must be specified when using multiple turbine models.")
        end
        power_thrust_table = turbine_power_thrust_table[multidim_condition]
    end

    return axial_induction_function(;
        power_thrust_table,
        velocities,
        yaw_angles,
        tilt_angles,
        tilt_interp,
        cubature_weights,
        correct_cp_ct_for_tilt,
    )
end

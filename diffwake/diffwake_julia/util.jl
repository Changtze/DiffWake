# ─── Utility types and helpers ────────────────────────────────────────────────
# Ported from util.py

using YAML, Statistics

# ─── Core types ──────────────────────────────────────────────────────────────

"""
    Params

Grid / turbine constants passed to the solver loop.
"""
Base.@kwdef struct Params
    B::Int
    T::Int
    rotor_diameter::Float64
    hub_height::Float64
    TSR::Float64
    wind_shear::Float64
    wind_veer::Float64
    gr_square::Float64
    enable_secondary_steering::Bool   = false
    enable_transverse_velocities::Bool = false
    enable_yaw_added_recovery::Bool   = false
end

"""
    DynamicState

Mutable state carried through the solver turbine loop.
"""
Base.@kwdef mutable struct DynamicState
    turb_u_wake::Array{Float64,4}
    turb_inflow::Array{Float64,4}
    ti::Array{Float64,4}
    v_sorted::Array{Float64,4}
    w_sorted::Array{Float64,4}

    # Model-specific buffers (used by CC model)
    Ctmp::Union{Nothing, Array{Float64,5}}  = nothing
    ct_acc::Union{Nothing, Array{Float64,4}} = nothing
end

"""
    Result

Immutable result returned after simulation.
"""
struct Result
    turb_u_wake::Array{Float64,4}
    u_sorted::Array{Float64,4}
    ti::Array{Float64,4}
    v_sorted::Array{Float64,4}
    w_sorted::Array{Float64,4}
end

"""
    Config

YAML-loaded configuration dictionaries.
"""
Base.@kwdef struct Config
    generator::Dict
    farm::Dict
    flow_field::Dict
    layout::Dict
end

"""
    State

Top-level simulation state grouping Farm, Grid, FlowField, WakeModelManager.
"""
struct State
    farm::Farm
    grid::TurbineGrid
    flow::FlowField
    wake::WakeModelManager
end


# ─── Callable wrappers ──────────────────────────────────────────────────────

"""
    ThrustFn

Callable wrapper for thrust coefficient computation.
"""
struct ThrustFn
    ti::Array{Float64,4}
    rho::Float64
    thrust_fn::Function
    tilt_interp::Union{Nothing, Function}
    correct_cp_ct_for_tilt::Bool
    power_thrust_table::Dict
end

function (t::ThrustFn)(; velocities, yaw_angles, tilt_angles)
    return turbine_thrust_coefficient(;
        velocities,
        yaw_angles,
        tilt_angles,
        thrust_fn           = t.thrust_fn,
        tilt_interp         = t.tilt_interp,
        correct_cp_ct_for_tilt = t.correct_cp_ct_for_tilt,
        power_thrust_table  = t.power_thrust_table,
    )
end

"""
    AxialInductionFn

Callable wrapper for axial induction computation.
"""
struct AxialInductionFn
    ti::Array{Float64,4}
    rho::Float64
    pset::Union{Nothing, Matrix{Float64}}
    axial_induction_function::Function
    tilt_interp::Union{Nothing, Function}
    correct_cp_ct_for_tilt::Bool
    power_thrust_table::Dict
    cubature_weights::Nothing
    multidim_condition::Nothing
end

function (a::AxialInductionFn)(velocities, yaw_angles, tilt_angles; ix_filter = nothing)
    return turbine_axial_induction(;
        velocities,
        yaw_angles,
        tilt_angles,
        axial_induction_function = a.axial_induction_function,
        tilt_interp              = a.tilt_interp,
        turbine_power_thrust_table = a.power_thrust_table,
        correct_cp_ct_for_tilt   = a.correct_cp_ct_for_tilt,
        ix_filter,
        cubature_weights         = a.cubature_weights,
        multidim_condition       = a.multidim_condition,
    )
end


# ─── Helpers ─────────────────────────────────────────────────────────────────

function average_velocity_util(v; method = "cubic-mean")
    if method == "simple-mean"
        return mean(v, dims = (3, 4))
    elseif method == "cubic-mean"
        m3 = mean(v .^ 3, dims = (3, 4))
        return _cbrt(m3)
    else
        error("Unknown averaging method: $method")
    end
end

function smooth_step(x, edge; width = 1.0)
    return 1.0 ./ (1.0 .+ exp.(-(x .- edge) ./ width))
end

function smooth_box(x, centre, half; inv_w = 2.0)
    s1 = 1.0 ./ (1.0 .+ exp.(-(x .- (centre .- half)) .* inv_w))
    s2 = 1.0 ./ (1.0 .+ exp.(-(x .- (centre .+ half)) .* inv_w))
    return s1 .* (1.0 .- s2)
end


"""
    init_dynamic_state(grid, flow, velocity_model_name) → DynamicState

Create the initial mutable solver state.
"""
function init_dynamic_state(grid::TurbineGrid, flow::FlowField, velocity_model_name::String)
    B, T, Ny, Nz = size(grid.x_sorted)
    zeros4 = zeros(B, T, Ny, Nz)
    ti = zeros(B, T, 3, 3)
    for b in 1:B
        ti[b, :, :, :] .= flow.turbulence_intensities[b]
    end

    Ctmp   = nothing
    ct_acc = nothing
    if velocity_model_name == "cc"
        Ctmp   = zeros(T, B, T, Ny, Nz)
        ct_acc = zeros(B, T, 1, 1)
    end

    return DynamicState(
        turb_u_wake = copy(zeros4),
        turb_inflow = copy(flow.u_initial_sorted),
        ti          = ti,
        v_sorted    = copy(zeros4),
        w_sorted    = copy(zeros4),
        Ctmp        = Ctmp,
        ct_acc      = ct_acc,
    )
end


"""
    get_axial_induction_fn(flow, farm, grid) → AxialInductionFn
"""
function get_axial_induction_fn(flow::FlowField, farm::Farm, grid::TurbineGrid)
    return AxialInductionFn(
        flow.turbulence_intensity_field_sorted,
        flow.air_density,
        farm.power_setpoints_sorted,
        farm.axial_induction_function,
        farm.tilt_interp,
        farm.correct_cp_ct_for_tilt,
        farm.power_thrust_table,
        nothing,
        nothing,
    )
end


"""
    get_thrust_fn(flow, farm) → ThrustFn
"""
function get_thrust_fn(flow::FlowField, farm::Farm)
    return ThrustFn(
        flow.turbulence_intensity_field_sorted,
        flow.air_density,
        farm.thrust_coefficient_function,
        farm.tilt_interp,
        farm.correct_cp_ct_for_tilt,
        farm.power_thrust_table,
    )
end


"""
    make_constants(state) → (const_dict, yaw_angles, tilt_angles)
"""
function make_constants(state::State)
    g    = state.grid
    fld  = state.flow
    farm = state.farm

    x, y, z = g.x_sorted, g.y_sorted, g.z_sorted
    x_c = mean(x, dims = (3, 4))
    y_c = mean(y, dims = (3, 4))
    z_c = mean(z, dims = (3, 4))

    cnst = Dict{String, Any}(
        "x_coord"    => x,
        "y_coord"    => y,
        "z_coord"    => z,
        "x_c"        => x_c,
        "y_c"        => y_c,
        "z_c"        => z_c,
        "u_init"     => copy(fld.u_initial_sorted),
        "dudz_init"  => copy(fld.dudz_initial_sorted),
        "ambient_ti" => reshape(fld.turbulence_intensities, :, 1, 1, 1),
    )

    yaw_angles  = farm.yaw_angles_sorted
    tilt_angles = farm.tilt_angles_sorted

    return cnst, yaw_angles, tilt_angles
end


"""
    make_params(state) → (Params, ss, yar, tv)
"""
function make_params(state::State)
    B, T, _, _ = size(state.grid.x_sorted)

    params = Params(
        B  = B,
        T  = T,
        rotor_diameter = state.farm.rotor_diameter,
        hub_height     = state.farm.hub_height,
        TSR            = state.farm.TSR,
        wind_shear     = state.flow.wind_shear,
        wind_veer      = state.flow.wind_veer,
        gr_square      = Float64(state.grid.grid_resolution^2),
        enable_secondary_steering    = state.wake.enable_secondary_steering,
        enable_yaw_added_recovery    = state.wake.enable_yaw_added_recovery,
        enable_transverse_velocities = state.wake.enable_transverse_velocities,
    )

    return (params,
            state.wake.enable_secondary_steering,
            state.wake.enable_yaw_added_recovery,
            state.wake.enable_transverse_velocities)
end


"""
    to_result(st::DynamicState) → Result
"""
function to_result(st::DynamicState)
    return Result(st.turb_u_wake, st.turb_inflow, st.ti, st.v_sorted, st.w_sorted)
end


# ─── YAML loader ─────────────────────────────────────────────────────────────

"""
    load_yaml(path) → Dict

Load a YAML file and convert numeric lists to Float64 vectors.
"""
function load_yaml(path::AbstractString)
    data = YAML.load_file(path)
    return _convert_numerics(data)
end

function _convert_numerics(obj)
    if obj isa Vector && all(x -> x isa Number, obj)
        return Float64.(obj)
    elseif obj isa Dict
        return Dict(k => _convert_numerics(v) for (k, v) in obj)
    elseif obj isa Vector
        return [_convert_numerics(x) for x in obj]
    else
        return obj
    end
end


"""
    set_config(cfg; kwargs...) → Config

Return a new Config with updated wind speeds, directions, TI, layout, yaw.
"""
function set_config(cfg::Config;
    wind_speeds           = nothing,
    wind_directions       = nothing,
    turbulence_intensities = nothing,
    layout_x              = nothing,
    layout_y              = nothing,
    yaw_angles            = nothing,
)
    flow_field = copy(cfg.flow_field)
    layout     = copy(cfg.layout)
    farm       = copy(cfg.farm)

    wind_speeds           !== nothing && (flow_field["wind_speeds"]           = wind_speeds)
    wind_directions       !== nothing && (flow_field["wind_directions"]       = wind_directions)
    turbulence_intensities !== nothing && (flow_field["turbulence_intensities"] = turbulence_intensities)

    layout_x !== nothing && (layout["layout_x"] = layout_x)
    layout_y !== nothing && (layout["layout_y"] = layout_y)

    yaw_angles !== nothing && (farm["yaw_angles"] = yaw_angles)

    return Config(generator = cfg.generator, farm = farm, flow_field = flow_field, layout = layout)
end

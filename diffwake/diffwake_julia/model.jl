# ─── Model: factory functions and YAML → State pipeline ──────────────────────
# Ported from model.py

const MODEL_MAP = Dict(
    "combination_model" => Dict(
        "sosfs" => SOSFS,
    ),
    "deflection_model" => Dict(
        "gauss" => GaussVelocityDeflection,
    ),
    "turbulence_model" => Dict(
        "crespo_hernandez" => CrespoHernandez,
    ),
    "velocity_model" => Dict(
        "gauss" => GaussVelocityDeficit,
    ),
)


"""
    load_input(farm_path, generator_path) → Config

Load YAML configuration files and return a `Config`.
"""
function load_input(farm_path::AbstractString, generator_path::AbstractString)
    farm_dict      = load_yaml(farm_path)
    generator_dict = load_yaml(generator_path)
    flow_field_dict = farm_dict["flow_field"]

    wind_height = flow_field_dict["reference_wind_height"]
    if wind_height < 0.0
        wind_height = farm_dict["farm"]["turbine_type"][1]["hub_height"]
    end

    for (key, value) in flow_field_dict
        if key == "reference_wind_height"
            flow_field_dict[key] = wind_height
        end
        if value isa Vector && all(x -> x isa Number, value)
            flow_field_dict[key] = Float64.(value)
        end
    end

    layout_dict = farm_dict["farm"]
    for (key, value) in layout_dict
        if value isa Vector && !isempty(value) && value[1] isa Float64
            layout_dict[key] = sort(Float64.(value))
        end
    end

    return Config(generator = generator_dict, farm = farm_dict, flow_field = flow_field_dict, layout = layout_dict)
end


"""
    create_wake(farm_dict) → WakeModelManager

Instantiate wake sub-models from the farm configuration dictionary.
"""
function create_wake(farm_dict::Dict)
    wake_dict = farm_dict["wake"]

    # Velocity model
    vel_model_string = lowercase(wake_dict["model_strings"]["velocity_model"])
    vel_model_type = MODEL_MAP["velocity_model"][vel_model_string]
    vel_model_params = wake_dict["wake_velocity_parameters"][vel_model_string]
    vel_model = vel_model_type(; _dict_to_kwargs(vel_model_params)...)

    # Deflection model
    def_model_string = lowercase(wake_dict["model_strings"]["deflection_model"])
    def_model_type = MODEL_MAP["deflection_model"][def_model_string]
    def_model_params = wake_dict["wake_deflection_parameters"][def_model_string]
    def_model = def_model_type(; _dict_to_kwargs(def_model_params)...)

    # Turbulence model
    turb_model_string = lowercase(wake_dict["model_strings"]["turbulence_model"])
    turb_model_type = MODEL_MAP["turbulence_model"][turb_model_string]
    turb_model_params = wake_dict["wake_turbulence_parameters"][turb_model_string]
    turb_model = turb_model_type(; _dict_to_kwargs(turb_model_params)...)

    # Combination model
    combo_model_string = lowercase(wake_dict["model_strings"]["combination_model"])
    combo_model_type = MODEL_MAP["combination_model"][combo_model_string]
    combo_model = combo_model_type()

    return WakeModelManager(
        velocity_model    = vel_model,
        deflection_model  = def_model,
        turbulence_model  = turb_model,
        combination_model = combo_model,
        enable_secondary_steering    = get(wake_dict, "enable_secondary_steering", false),
        enable_yaw_added_recovery    = get(wake_dict, "enable_yaw_added_recovery", false),
        enable_active_wake_mixing    = get(wake_dict, "enable_active_wake_mixing", false),
        enable_transverse_velocities = get(wake_dict, "enable_transverse_velocities", false),
        model_strings = Dict{String,String}(k => string(v) for (k,v) in wake_dict["model_strings"]),
    )
end


"""
    create_grid(layout, generator, farm, flow) → TurbineGrid
"""
function create_grid(layout::Dict, generator::Dict, farm_dict::Dict, flow::Dict)
    layout_x = Float64.(layout["layout_x"])
    layout_y = Float64.(layout["layout_y"])
    hub_h    = Float64(farm_dict["farm"]["turbine_type"][1]["hub_height"])
    z_coords = fill(hub_h, length(layout_x))

    coords = hcat(layout_x, layout_y, z_coords)   # (T, 3)
    D      = Float64(generator["rotor_diameter"])
    wd     = Float64.(flow["wind_directions"])

    return create_turbine_grid(coords, D, wd; grid_resolution = 3)
end


"""
    create_farm_from_config(layout, generator, sorted_indices) → Farm
"""
function create_farm_from_config(layout::Dict, generator::Dict, sorted_indices::Array{Int,4})
    farm = create_farm(
        Float64.(layout["layout_x"]),
        Float64.(layout["layout_y"]),
        generator,
        Turbine,
    )
    initialize_farm!(farm, sorted_indices)
    return farm
end


"""
    create_flow_field_from_config(flow_dict, grid) → FlowField
"""
function create_flow_field_from_config(flow_dict::Dict, grid::TurbineGrid)
    ff = FlowField(
        wind_speeds           = Float64.(flow_dict["wind_speeds"]),
        wind_directions       = Float64.(flow_dict["wind_directions"]),
        wind_shear            = Float64(flow_dict["wind_shear"]),
        wind_veer             = Float64(flow_dict["wind_veer"]),
        air_density           = Float64(flow_dict["air_density"]),
        turbulence_intensities = Float64.(flow_dict["turbulence_intensities"]),
        reference_wind_height = Float64(flow_dict["reference_wind_height"]),
    )
    initialize_velocity_field!(ff, grid)
    return ff
end


"""
    create_state(cfg::Config) → State

Build the full simulation state from a Config.
"""
function create_state(cfg::Config)
    wake = create_wake(cfg.farm)
    grid = create_grid(cfg.layout, cfg.generator, cfg.farm, cfg.flow_field)
    farm = create_farm_from_config(cfg.layout, cfg.generator, grid.sorted_indices)
    flow = create_flow_field_from_config(cfg.flow_field, grid)

    return State(farm, grid, flow, wake)
end


# ── helper: convert string-keyed Dict to Symbol kwargs ───────────────────────
function _dict_to_kwargs(d::Dict)
    return [Symbol(k) => v for (k, v) in d]
end

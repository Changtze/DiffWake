"""
    WakeModelManager

Container holding the four wake sub-models and their configuration flags.
"""
Base.@kwdef struct WakeModelManager
    velocity_model::Any
    deflection_model::Any
    turbulence_model::Any
    combination_model::Any

    model_strings::Dict{String, String} = Dict{String, String}()
    wake_deflection_parameters::Dict    = Dict()
    wake_turbulence_parameters::Dict    = Dict()
    wake_velocity_parameters::Dict      = Dict()

    enable_secondary_steering::Bool     = false
    enable_yaw_added_recovery::Bool     = false
    enable_active_wake_mixing::Bool     = false
    enable_transverse_velocities::Bool  = false
end

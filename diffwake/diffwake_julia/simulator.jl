# ─── Simulator: main entry points ────────────────────────────────────────────
# Ported from simulator.py

"""
    simulate(state::State) → Result

Run the full wake simulation with the configured wake model.
"""
function simulate(state::State)
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)

    params, enable_ss, enable_yar, enable_tv = make_params(state)
    cnst, yaw_angles, tilt_angles = make_constants(state)
    init = init_dynamic_state(state.grid, state.flow, get(state.wake.model_strings, "velocity_model", "gauss"))
    T_int = params.T

    result_state = _simulate(
        T_int, params,
        thrust_fn, axial_fn,
        state.wake.velocity_model,
        get(state.wake.model_strings, "velocity_model", "gauss"),
        state.wake.deflection_model,
        state.wake.turbulence_model,
        state.wake.combination_model,
        yaw_angles, tilt_angles,
        cnst, init;
        enable_secondary_steering    = enable_ss,
        enable_transverse_velocities = enable_tv,
        enable_yaw_added_recovery    = enable_yar,
    )

    return to_result(result_state)
end


"""
    _simulate(T, params, ..., state; kwargs...) → DynamicState

Internal simulation loop dispatching to the correct solver step.
"""
function _simulate(
    T::Int,
    params::Params,
    thrust_function,
    axial_induction_func,
    velocity_model,
    velocity_model_name::String,
    deflection_model,
    turbulence_model,
    combination_model,
    yaw_angles, tilt_angles,
    cnst::Dict, state::DynamicState;
    enable_secondary_steering::Bool,
    enable_transverse_velocities::Bool,
    enable_yaw_added_recovery::Bool,
)
    if velocity_model_name == "cc"
        for i in 1:T
            cc_solver_step!(
                state, i, params,
                thrust_function, axial_induction_func,
                velocity_model, deflection_model, turbulence_model,
                yaw_angles, tilt_angles;
                enable_secondary_steering,
                enable_transverse_velocities,
                enable_yaw_added_recovery,
                x_coord    = cnst["x_coord"],
                y_coord    = cnst["y_coord"],
                z_coord    = cnst["z_coord"],
                x_c        = cnst["x_c"],
                y_c        = cnst["y_c"],
                z_c        = cnst["z_c"],
                u_init     = cnst["u_init"],
                dudz_init  = cnst["dudz_init"],
                ambient_ti = cnst["ambient_ti"],
            )
        end

    elseif velocity_model_name == "gauss"
        for i in 1:T
            sequential_solve_step!(
                state, i, params,
                thrust_function, axial_induction_func,
                velocity_model, deflection_model,
                turbulence_model, combination_model,
                yaw_angles, tilt_angles;
                enable_secondary_steering,
                enable_transverse_velocities,
                enable_yaw_added_recovery,
                x_coord    = cnst["x_coord"],
                y_coord    = cnst["y_coord"],
                z_coord    = cnst["z_coord"],
                x_c        = cnst["x_c"],
                y_c        = cnst["y_c"],
                z_c        = cnst["z_c"],
                u_init     = cnst["u_init"],
                dudz_init  = cnst["dudz_init"],
                ambient_ti = cnst["ambient_ti"],
            )
        end

    else
        error("Velocity model '$velocity_model_name' not implemented.")
    end

    return state
end


"""
    alter_yaw_angles(yaw_angles, state::State) → State

Return a new State with updated yaw angles (sorted to match upstream→downstream
ordering).
"""
function alter_yaw_angles(yaw_angles::Matrix{Float64}, state::State)
    idx = state.grid.sorted_indices[:, :, 1, 1]
    yaw_s = _take_along_axis2_mat(yaw_angles, idx)

    new_farm = Farm(
        layout_x          = state.farm.layout_x,
        layout_y          = state.farm.layout_y,
        n_turbines        = state.farm.n_turbines,
        hub_height        = state.farm.hub_height,
        rotor_diameter    = state.farm.rotor_diameter,
        TSR               = state.farm.TSR,
        ref_tilt          = state.farm.ref_tilt,
        correct_cp_ct_for_tilt = state.farm.correct_cp_ct_for_tilt,
        power_function              = state.farm.power_function,
        thrust_coefficient_function = state.farm.thrust_coefficient_function,
        axial_induction_function    = state.farm.axial_induction_function,
        tilt_interp                 = state.farm.tilt_interp,
        power_thrust_table          = state.farm.power_thrust_table,
        turbine                     = state.farm.turbine,
        yaw_angles                  = yaw_angles,
        tilt_angles                 = state.farm.tilt_angles,
        power_setpoints             = state.farm.power_setpoints,
        yaw_angles_sorted           = yaw_s,
        tilt_angles_sorted          = state.farm.tilt_angles_sorted,
        power_setpoints_sorted      = state.farm.power_setpoints_sorted,
    )

    return State(new_farm, state.grid, state.flow, state.wake)
end


"""
    turbine_powers(state::State) → Array

Compute power output for all turbines.
"""
function turbine_powers(state::State)
    return turbine_power(;
        velocities         = state.flow.u,
        air_density        = state.flow.air_density,
        power_function     = state.farm.power_function,
        yaw_angles         = state.farm.yaw_angles,
        tilt_angles        = state.farm.tilt_angles,
        power_thrust_table = state.farm.power_thrust_table,
    )
end

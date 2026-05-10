# ─── Solver: per-turbine wake update steps ───────────────────────────────────
# Ported from solver.py

"""
    sequential_solve_step!(state, ii, params, thrust_function, axial_induction_func,
        velocity_model, deflection_model, turbulence_model, combination_model,
        yaw_angles, tilt_angles; kwargs...)

One turbine-index update for the Gauss-Curl Hybrid (GCH) model.

Steps (matching the JAX version):
1. Extract local velocity components at turbine `ii`
2. Calculate Ct and axial induction
3. (Optional) secondary steering
4. Wake deflection
5. Transverse velocities
6. (Optional) yaw-added recovery TI update
7. Velocity deficit + wake combination
8. TI field update
"""
function sequential_solve_step!(
    state::DynamicState,
    ii::Int,                         # 1-based turbine index
    params::Params,
    thrust_function,
    axial_induction_func,
    velocity_model,
    deflection_model,
    turbulence_model,
    combination_model,
    yaw_angles::Matrix{Float64},     # (B, T)
    tilt_angles::Matrix{Float64};
    enable_secondary_steering::Bool,
    enable_transverse_velocities::Bool,
    enable_yaw_added_recovery::Bool,
    x_coord::Array{Float64,4},
    y_coord::Array{Float64,4},
    z_coord::Array{Float64,4},
    x_c::Array{Float64},            # (B, T, 1, 1)
    y_c::Array{Float64},
    z_c::Array{Float64},
    u_init::Array{Float64,4},
    dudz_init::Array{Float64,4},
    ambient_ti::Array{Float64},     # (B,1,1,1)
)
    turb_u_wake = state.turb_u_wake
    turb_inflow = state.turb_inflow
    ti          = state.ti
    v_sorted    = state.v_sorted
    w_sorted    = state.w_sorted

    B, T, Ny, Nz = size(x_coord)

    x_i = x_c[:, ii:ii, :, :]   # (B,1,1,1)
    y_i = y_c[:, ii:ii, :, :]
    z_i = z_c[:, ii:ii, :, :]

    # Current flow field at this step
    u_sorted = u_init .- turb_u_wake

    # Update turb_inflow for turbine ii
    turb_inflow[:, ii:ii, :, :] .= u_sorted[:, ii:ii, :, :]

    u_i = turb_inflow[:, ii:ii, :, :]   # (B,1,Ny,Nz)

    yaw_i  = yaw_angles[:, ii:ii]       # (B,1)
    tilt_i = tilt_angles[:, ii:ii]

    # Thrust and axial induction
    turb_Cts_i = thrust_function(;
        velocities  = u_i,
        yaw_angles  = yaw_i,
        tilt_angles = tilt_i,
    )
    # (B,1) → (B,1,1,1)
    turb_Cts_i = reshape(turb_Cts_i, B, 1, 1, 1)

    axial_i = axial_induction_func(turb_inflow, yaw_angles, tilt_angles; ix_filter = ii)
    axial_i = reshape(axial_i, B, 1, 1, 1)

    turb_aIs = axial_i

    u_i = turb_inflow[:, ii:ii, :, :]
    v_i = v_sorted[:, ii:ii, :, :]
    w_i = w_sorted[:, ii:ii, :, :]
    ti_i = ti[:, ii:ii, :, :]
    yaw_i_expanded = reshape(yaw_i, B, 1, 1, 1)   # (B,1,1,1)

    # ── 3. Secondary steering ────────────────────────────────────────────
    if enable_secondary_steering
        y_coord_i = y_coord[:, ii:ii, :, :]
        z_coord_i = z_coord[:, ii:ii, :, :]

        added_yaw = wake_added_yaw(
            u_i, v_i, u_init,
            y_coord_i .- y_i, z_coord_i,
            params.rotor_diameter, params.hub_height,
            turb_Cts_i, params.TSR, axial_i,
            params.wind_shear; scale = 1.0,
        )
        yaw_eff = yaw_i_expanded .+ added_yaw
    else
        yaw_eff = yaw_i_expanded
    end

    # ── 4. Wake deflection ───────────────────────────────────────────────
    def_field = deflection_model(
        x_i, yaw_eff, ti_i, turb_Cts_i,
        params.rotor_diameter,
        x_coord, u_init, params.wind_veer,
    )

    # ── 5. Transverse velocities ─────────────────────────────────────────
    if enable_transverse_velocities
        v_wake, w_wake = calculate_transverse_velocity(
            u_i, u_init, dudz_init,
            x_coord .- x_i, y_coord .- y_i, z_coord,
            params.rotor_diameter, params.hub_height,
            yaw_i_expanded, turb_Cts_i, params.TSR, axial_i,
            params.wind_shear; scale = 1.0,
        )
    else
        v_wake = zeros(size(v_sorted))
        w_wake = zeros(size(w_sorted))
    end

    # ── 6. Yaw-added recovery ────────────────────────────────────────────
    if enable_yaw_added_recovery
        v_wake_i = v_wake[:, ii:ii, :, :]
        w_wake_i = w_wake[:, ii:ii, :, :]

        I_mixing = yaw_added_turbulence_mixing(u_i, ti_i, v_i, w_i, v_wake_i, w_wake_i)

        gch_gain = 2.0
        updated = ti_i .+ gch_gain .* I_mixing
        # Sanitise
        updated = replace(updated, NaN => 0.0, Inf => 0.0, -Inf => 0.0)

        ti[:, ii:ii, :, :] .= updated
        ti_i = updated
    end

    # ── 7. Velocity deficit + combination ────────────────────────────────
    velocity_deficit, _ = velocity_model(
        x_i, y_i, z_i,
        axial_i,
        def_field,
        yaw_i_expanded,
        ti_i,
        turb_Cts_i,
        params.hub_height,
        params.rotor_diameter;
        x = x_coord,
        y = y_coord,
        z = z_coord,
        u_initial = u_init,
        wind_veer = params.wind_veer,
    )

    wake_field, _ = combination_model(turb_u_wake, velocity_deficit .* u_init)

    # ── 8. TI field update ───────────────────────────────────────────────
    wake_added_ti = turbulence_model(ambient_ti, x_coord, x_i, params.rotor_diameter, turb_aIs)

    area_overlap = sum(velocity_deficit .* u_init .> 0.05, dims = (3, 4)) ./ params.gr_square

    downstream_len = 15.0 * params.rotor_diameter
    eps_val = 1e-10

    downstream_start = x_coord .> (x_i .+ eps_val)
    downstream_end   = x_coord .<= (x_i .+ downstream_len .- eps_val)
    down_mask = Float64.(downstream_start .& downstream_end)

    dy_abs   = abs.(y_coord .- y_i)
    lat_mask = Float64.(dy_abs .< (2.0 * params.rotor_diameter .- eps_val))

    wake_ti = replace(wake_added_ti, NaN => 0.0, Inf => 0.0, -Inf => 0.0)

    ao = reshape(area_overlap, B, T, 1, 1)
    ti_added = ao .* wake_ti .* down_mask .* lat_mask

    ti .= max.(hypot.(ti_added, ambient_ti), ti)

    # Accumulate transverse velocities
    v_sorted .+= v_wake
    w_sorted .+= w_wake

    # Update state in-place
    state.turb_u_wake = wake_field
    state.turb_inflow = turb_inflow
    state.ti          = ti
    state.v_sorted    = v_sorted
    state.w_sorted    = w_sorted

    return state
end


"""
    cc_solver_step!(state, ii, params, ...) - stub

CC (Cumulative Curl) solver step.  Included for API completeness.
"""
function cc_solver_step!(
    state::DynamicState,
    ii::Int,
    params::Params,
    thrust_function,
    axial_induction_func,
    velocity_model,
    deflection_model,
    turbulence_model,
    yaw_angles, tilt_angles;
    enable_secondary_steering,
    enable_transverse_velocities,
    enable_yaw_added_recovery,
    x_coord, y_coord, z_coord,
    x_c, y_c, z_c,
    u_init, dudz_init,
    ambient_ti,
)
    # The CC solver step follows a very similar pattern but uses the
    # CumulativeGaussCurl velocity model with different accumulation logic.
    # This is a stub matching the JAX cc_solver_step signature.
    error("CC solver step not yet implemented in the Julia port. Use the Gauss model.")
end

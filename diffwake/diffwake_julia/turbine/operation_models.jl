# ─── Turbine operation models ────────────────────────────────────────────────
# Ported from operation_models.py

"""
    average_velocity(velocities; cubature_weights=nothing)

Cube-root-mean rotor-averaged velocity.  Reduces over the last two dimensions
(grid_y, grid_z).
"""
function average_velocity(velocities::AbstractArray; cubature_weights = nothing)
    # velocities: (..., Ny, Nz)
    dims = ndims(velocities)
    Ny = size(velocities, dims - 1)
    Nz = size(velocities, dims)
    # Compute mean of v^3 over last two dims, then cbrt
    v3 = velocities .^ 3
    # Average over last two dimensions
    leading = size(velocities)[1:end-2]
    out = zeros(leading...)
    for idx in CartesianIndices(out)
        s = 0.0
        for iy in 1:Ny, iz in 1:Nz
            s += v3[idx, iy, iz]
        end
        out[idx] = (s / (Ny * Nz))^(1.0 / 3.0)
    end
    return out
end


"""
    rotor_velocity_air_density_correction(velocities, air_density, ref_air_density)

Scale rotor velocities to reference air density.
"""
function rotor_velocity_air_density_correction(velocities, air_density, ref_air_density)
    scale = (air_density / ref_air_density)^(1.0 / 3.0)
    return scale .* velocities
end


"""
    rotor_velocity_yaw_cosine_correction(cosine_loss_exponent_yaw, yaw_angles, rotor_effective_velocities)

Apply cosine-based yaw loss correction.
"""
function rotor_velocity_yaw_cosine_correction(cosine_loss_exponent_yaw, yaw_angles, rotor_effective_velocities)
    pW = cosine_loss_exponent_yaw / 3.0
    correction = cos.(yaw_angles) .^ pW
    return rotor_effective_velocities .* correction
end


"""
    rotor_velocity_tilt_cosine_correction(...)

Apply cosine-based tilt loss correction.
"""
function rotor_velocity_tilt_cosine_correction(
    tilt_angles, ref_tilt, cosine_loss_exponent_tilt,
    tilt_interp, correct_cp_ct_for_tilt,
    rotor_effective_velocities,
)
    old_tilt_angle = tilt_angles

    if tilt_interp !== nothing
        tilt_angles = tilt_interp(rotor_effective_velocities)
    end

    tilt_angles = ifelse.(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angle)

    relative_tilt = tilt_angles .- ref_tilt
    exponent = cosine_loss_exponent_tilt / 3.0
    corrected = rotor_effective_velocities .* cos.(relative_tilt) .^ exponent

    return corrected
end


"""
    compute_tilt_angles_for_floating_turbines(tilt_angles, tilt_interp, rotor_effective_velocities)

Apply tilt interpolation for floating turbines.
"""
function compute_tilt_angles_for_floating_turbines(tilt_angles, tilt_interp, rotor_effective_velocities)
    if tilt_interp === nothing
        return tilt_angles
    else
        return tilt_interp(rotor_effective_velocities)
    end
end


"""
    interpolation(x, y, x_new)

Interpolate `y(x)` at `x_new`, handling arbitrary shapes.
"""
function interpolation(x::AbstractVector, y::AbstractVector, x_new::AbstractArray)
    original_shape = size(x_new)
    x_new_flat = vec(x_new)
    y_interp_flat = interp1d(x, y, x_new_flat)
    return reshape(y_interp_flat, original_shape)
end


"""
    cosine_loss_thrust_coefficient(...)

Compute thrust coefficient with cosine loss for yaw and tilt misalignment.
"""
function cosine_loss_thrust_coefficient(;
    power_thrust_table::Dict,
    velocities,
    yaw_angles,
    tilt_angles,
    tilt_interp = nothing,
    cubature_weights = nothing,
    correct_cp_ct_for_tilt = false,
)
    rotor_avg_vels = average_velocity(velocities; cubature_weights)
    thrust_curve = Float64.(power_thrust_table["thrust_coefficient"])
    wind_speeds  = Float64.(power_thrust_table["wind_speed"])
    ct = interpolation(wind_speeds, thrust_curve, rotor_avg_vels)
    ct = clamp.(ct, 0.0001, 0.99999)

    old_tilt = tilt_angles

    tilt = compute_tilt_angles_for_floating_turbines(tilt_angles, tilt_interp, rotor_avg_vels)

    # Apply correction conditionally
    tilt = ifelse.(correct_cp_ct_for_tilt, tilt, old_tilt)

    tilt_diff_rad = tilt .- power_thrust_table["ref_tilt"]

    ct .*= cos.(yaw_angles) .* cos.(tilt_diff_rad)

    return ct
end


"""
    cosine_loss_axial_induction(...)

Compute axial induction factor with cosine loss.
"""
function cosine_loss_axial_induction(;
    power_thrust_table::Dict,
    velocities,
    yaw_angles,
    tilt_angles,
    tilt_interp = nothing,
    cubature_weights = nothing,
    correct_cp_ct_for_tilt = false,
)
    ct = cosine_loss_thrust_coefficient(;
        power_thrust_table, velocities, yaw_angles, tilt_angles,
        tilt_interp, cubature_weights, correct_cp_ct_for_tilt,
    )

    tilt_diff_rad = tilt_angles .- power_thrust_table["ref_tilt"]
    misalignment = cos.(yaw_angles) .* cos.(tilt_diff_rad)

    sqrt_term = sqrt.(clamp.(1.0 .- ct, 0.0, 1.0))
    a = 0.5 ./ misalignment .* (1.0 .- sqrt_term)

    return a
end


"""
    op_power(...)

Compute turbine power output using the cosine loss model.
"""
function op_power(;
    power_thrust_table::Dict,
    velocities,
    air_density,
    yaw_angles,
    tilt_angles = nothing,
    cubature_weights = nothing,
    kwargs...,
)
    rotor_avg_vels = average_velocity(velocities; cubature_weights)

    v_eff = rotor_velocity_air_density_correction(rotor_avg_vels, air_density, power_thrust_table["ref_air_density"])
    v_eff = rotor_velocity_yaw_cosine_correction(power_thrust_table["cosine_loss_exponent_yaw"], yaw_angles, v_eff)

    wind_speeds = Float64.(power_thrust_table["wind_speed"])
    powers      = Float64.(power_thrust_table["power"])

    power_output = interpolation(wind_speeds, powers, v_eff)
    return power_output .* 1e3   # kW → W
end

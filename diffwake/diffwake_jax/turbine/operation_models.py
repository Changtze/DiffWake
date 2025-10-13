import torch
from ..util.interp1d import interp1d
POWER_SETPOINT_DEFAULT = 1e12

from typing import Callable, Optional, Dict, Tuple

import jax.numpy as jnp
from jax import lax

def average_velocity(velocities, cubature_weights=None):
    
    return jnp.power(jnp.mean(jnp.power(velocities, 3), axis=(-1, -2)), 1.0 / 3.0)


def rotor_velocity_air_density_correction(
    velocities: jnp.ndarray,
    air_density: float,
    ref_air_density: float,
) -> jnp.ndarray:
    """
    Scale rotor velocities to account for air density differences.

    Args:
        velocities (jnp.ndarray): Rotor effective velocities [m/s].
        air_density (float): Actual air density [kg/m³].
        ref_air_density (float): Reference air density [kg/m³].

    Returns:
        jnp.ndarray: Corrected rotor velocities at reference density.
    """
    scale = (air_density / ref_air_density) ** (1.0 / 3.0)
    return scale * velocities





def rotor_velocity_yaw_cosine_correction(
    cosine_loss_exponent_yaw: float,
    yaw_angles: jnp.ndarray,                   # shape: (N,) or (B, ...)
    rotor_effective_velocities: jnp.ndarray    # same shape as yaw_angles
) -> jnp.ndarray:
    """
    Applies cosine-based yaw loss correction to rotor effective velocities.

    Args:
        cosine_loss_exponent_yaw (float): Exponent for cosine loss.
        yaw_angles (jnp.ndarray): Yaw angles in **radians**.
        rotor_effective_velocities (jnp.ndarray): Effective rotor velocities [m/s].

    Returns:
        jnp.ndarray: Yaw-corrected rotor velocities, same shape as input.
    """
    pW = cosine_loss_exponent_yaw / 3.0
    correction = jnp.cos(yaw_angles) ** pW
    return rotor_effective_velocities * correction


def rotor_velocity_tilt_cosine_correction(
    tilt_angles: jnp.ndarray,                         # shape: (N,)
    ref_tilt: jnp.ndarray,                            # shape: (N,) or broadcastable
    cosine_loss_exponent_tilt: float,
    tilt_interp: Optional[Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]],
    correct_cp_ct_for_tilt: jnp.ndarray,              # Bool array: shape (N,)
    rotor_effective_velocities: jnp.ndarray           # shape: (N,)
) -> jnp.ndarray:
    """
    Apply cosine-based correction for rotor tilt on effective velocities.

    All angles must be in radians.

    Returns:
        jnp.ndarray: Corrected rotor effective velocities.
    """
    old_tilt_angle = tilt_angles

    if tilt_interp is not None:
        x_vals, y_vals = next(iter(tilt_interp.values()))
        tilt_angles = interp1d(x_vals, y_vals, rotor_effective_velocities)

    # Only apply new tilt where correction is enabled
    tilt_angles = jnp.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angle)

    relative_tilt = tilt_angles - ref_tilt
    exponent = cosine_loss_exponent_tilt / 3.0
    corrected_velocities = rotor_effective_velocities * jnp.cos(relative_tilt) ** exponent

    return corrected_velocities

def cosine_loss_power(
    power_thrust_table: Dict[str, jnp.ndarray],
    velocities: jnp.ndarray,
    air_density: float,
    yaw_angles: jnp.ndarray,
    tilt_angles: jnp.ndarray,
    cubature_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    rotor_avg_vels = average_velocity(velocities, cubature_weights)

    v_eff = rotor_velocity_air_density_correction(
        rotor_avg_vels, air_density, power_thrust_table["ref_air_density"]
    )

    v_eff = rotor_velocity_yaw_cosine_correction(
        power_thrust_table["cosine_loss_exponent_yaw"], yaw_angles, v_eff
    )

    v_eff = rotor_velocity_tilt_cosine_correction(
        tilt_angles,
        power_thrust_table["ref_tilt"],
        power_thrust_table["cosine_loss_exponent_tilt"],
        v_eff,
    )

    # Interpolate power output
    wind_speeds = power_thrust_table["wind_speed"]
    power_curve = power_thrust_table["power"]

    power_output = interp1d(wind_speeds.unsqueeze(0), power_curve.unsqueeze(0), v_eff)
    return power_output  # in kW


def compute_tilt_angles_for_floating_turbines(
    tilt_angles: jnp.ndarray,
    tilt_interp: Callable | None,
    rotor_effective_velocities: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply tilt angle interpolation for floating turbines.

    Args:
        tilt_angles (Tensor): Initial tilt angles, possibly all zeros. (shape: [N])
        tilt_interp (callable or None): Function that maps velocities → tilt angles.
        rotor_effective_velocities (Tensor): Effective rotor velocities. (shape: [N])

    Returns:
        Tensor: Updated tilt angles. (shape: [N])
    """
    if tilt_interp is None:
        # No change if no interpolation function is given
        return tilt_angles
    else:
        # Apply interpolator to compute new tilt angles
        return tilt_interp(rotor_effective_velocities)


def interpolation(x, y, x_new):
    original_shape = x_new.shape
    x_new_flat = jnp.reshape(x_new, (-1,))
    y_interp_flat = interp1d(x, y, x_new_flat)
    return jnp.reshape(y_interp_flat, original_shape)


def cosine_loss_thrust_coefficient(
    power_thrust_table: dict,
    velocities: jnp.ndarray,
    yaw_angles: jnp.ndarray,
    tilt_angles: jnp.ndarray,
    tilt_interp=None,
    cubature_weights=None,
    correct_cp_ct_for_tilt=False,
):
    rotor_avg_vels = average_velocity(velocities, cubature_weights)
    thrust_curve = jnp.array(power_thrust_table["thrust_coefficient"])
    wind_speeds = jnp.array(power_thrust_table["wind_speed"])
    ct = interpolation(wind_speeds, thrust_curve, rotor_avg_vels)
    ct = jnp.clip(ct, 0.0001, 0.99999)

    old_tilt = tilt_angles

    tilt = compute_tilt_angles_for_floating_turbines(tilt_angles, tilt_interp, rotor_avg_vels)

    # Apply correction conditionally
    tilt = lax.select(correct_cp_ct_for_tilt, tilt, old_tilt)

    
    tilt_diff_rad = (tilt - power_thrust_table["ref_tilt"])

    ct *= jnp.cos(yaw_angles) * jnp.cos(tilt_diff_rad)

    return ct


def cosine_loss_axial_induction(
    power_thrust_table: dict,
    velocities: jnp.ndarray,
    yaw_angles: jnp.ndarray,
    tilt_angles: jnp.ndarray,
    tilt_interp=None,
    cubature_weights=None,
    correct_cp_ct_for_tilt=False,
):

    ct = cosine_loss_thrust_coefficient(
        power_thrust_table, velocities, yaw_angles, tilt_angles,
        tilt_interp, cubature_weights, correct_cp_ct_for_tilt
    )

    tilt_diff_rad = (tilt_angles - power_thrust_table["ref_tilt"])

    misalignment = jnp.cos(yaw_angles) * jnp.cos(tilt_diff_rad)

    sqrt_term = jnp.sqrt(jnp.clip(1.0 - ct * misalignment, 0.0, 1.0))
    a = 0.5 / misalignment * (1.0 - sqrt_term)

    return a


def power(
    power_thrust_table: dict,
    velocities: jnp.ndarray,
    air_density: float,
    yaw_angles: jnp.ndarray,
    tilt_angles: jnp.ndarray = None,  # kept for interface consistency
    cubature_weights=None,
    **_
) -> jnp.ndarray:
    rotor_avg_vels = average_velocity(velocities, cubature_weights)
    v_eff = rotor_velocity_air_density_correction(rotor_avg_vels, air_density, power_thrust_table["ref_air_density"])
    v_eff = rotor_velocity_yaw_cosine_correction(power_thrust_table["cosine_loss_exponent_yaw"], yaw_angles, v_eff)

    wind_speeds = jnp.array(power_thrust_table["wind_speed"])
    powers = jnp.array(power_thrust_table["power"])

    power_output = interp1d(wind_speeds, powers, v_eff)
    return power_output * 1e3  # Convert kW to W




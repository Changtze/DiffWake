import torch
import torch.nn.functional as F
from ..interp1d import interp1d
from ..utils import average_velocity
POWER_SETPOINT_DEFAULT = 1e12



def rotor_velocity_air_density_correction(
    velocities: torch.Tensor,
    air_density: float,
    ref_air_density: float,
) -> torch.Tensor:
    """
    Scale rotor velocities to account for air density differences.

    Args:
        velocities (Tensor): Rotor effective velocities [m/s].
        air_density (float): Actual air density [kg/m³].
        ref_air_density (float): Reference air density [kg/m³].

    Returns:
        Tensor: Corrected rotor velocities at reference density.
    """
    scale = (air_density / ref_air_density) ** (1. / 3.)
    return scale * velocities

import torch
import math

def cosd(x: torch.Tensor) -> torch.Tensor:
    """Cosine with degree input (like NumPy's cosd)."""
    return torch.cos(torch.deg2rad(x))

def rotor_velocity_yaw_cosine_correction(
    cosine_loss_exponent_yaw: float,
    yaw_angles: torch.Tensor,                    # shape: (N,) or (B, ...)
    rotor_effective_velocities: torch.Tensor     # same shape as yaw_angles
) -> torch.Tensor:
    """
    Applies cosine-based yaw loss correction to rotor effective velocities.

    Args:
        cosine_loss_exponent_yaw (float): Exponent for cosine loss.
        yaw_angles (Tensor): Yaw angles in degrees.
        rotor_effective_velocities (Tensor): Effective rotor velocities [m/s].

    Returns:
        Tensor: Yaw-corrected rotor velocities, same shape as input.
    """
    pW = cosine_loss_exponent_yaw / 3.0
    correction = torch.cos(yaw_angles) ** pW
    return rotor_effective_velocities * correction


def rotor_velocity_tilt_cosine_correction(
    tilt_angles: torch.Tensor,                         # shape: (N,)
    ref_tilt: torch.Tensor,                            # shape: (N,) or broadcastable
    cosine_loss_exponent_tilt: float,
    tilt_interp: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    correct_cp_ct_for_tilt: torch.Tensor,              # Bool tensor: shape (N,)
    rotor_effective_velocities: torch.Tensor           # shape: (N,)
) -> torch.Tensor:
    """
    Apply cosine-based correction for rotor tilt on effective velocities.

    Args:
        tilt_angles (Tensor): Current tilt angles (deg).
        ref_tilt (Tensor): Reference tilt angles (deg).
        cosine_loss_exponent_tilt (float): Exponent applied to cosine loss.
        tilt_interp (dict or None): Dictionary of (x, y) interpolation tensors.
        correct_cp_ct_for_tilt (BoolTensor): Whether tilt should be corrected.
        rotor_effective_velocities (Tensor): Initial effective rotor velocities.

    Returns:
        Tensor: Corrected rotor effective velocities.
    """
    old_tilt_angle = tilt_angles.clone()

    if tilt_interp is not None:
        # Only one turbine type assumed here
        x_vals, y_vals = next(iter(tilt_interp.values()))
        tilt_angles = interp1d(x_vals, y_vals, rotor_effective_velocities)

    # Only update tilt if correction is enabled
    tilt_angles = torch.where(
        correct_cp_ct_for_tilt.bool(), tilt_angles, old_tilt_angle
    )

    # Apply cosine loss correction
    relative_tilt = tilt_angles - ref_tilt
    exponent = cosine_loss_exponent_tilt / 3.0
    corrected_velocities = rotor_effective_velocities * torch.cos(relative_tilt) ** exponent

    return corrected_velocities

def cosine_loss_power(
    power_thrust_table: dict,
    velocities: torch.Tensor,
    air_density: float,
    yaw_angles: torch.Tensor,
    tilt_angles: torch.Tensor,
    average_method="cubic-mean",
    cubature_weights=None,
):
    rotor_avg_vels = average_velocity(velocities, average_method, cubature_weights)
    v_eff = rotor_velocity_air_density_correction(rotor_avg_vels, air_density, power_thrust_table["ref_air_density"])
    v_eff = rotor_velocity_yaw_cosine_correction(power_thrust_table["cosine_loss_exponent_yaw"], yaw_angles, v_eff)
    v_eff = rotor_velocity_tilt_cosine_correction(tilt_angles, power_thrust_table["ref_tilt"],
                                                  power_thrust_table["cosine_loss_exponent_tilt"], v_eff)
    
    # Interpolate power output
    power_curve = power_thrust_table["power"]  # shape [N]
    wind_speeds = power_thrust_table["wind_speed"]

 
    power_output = interp1d.interp1d(wind_speeds.unsqueeze(0), power_curve.unsqueeze(0), v_eff)
    return power_output #kWatt

import torch
from typing import Callable, Optional

def compute_tilt_angles_for_floating_turbines(
    tilt_angles: torch.Tensor,
    tilt_interp: Callable | None,
    rotor_effective_velocities: torch.Tensor,
) -> torch.Tensor:
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


def interpolation(x,y,x_new):
    original_shape = x_new.shape
    x_new_flat = x_new.flatten()  # shape [P]
    y_interp_flat = interp1d(x, y, x_new_flat)
    return y_interp_flat.view(original_shape)


def cosine_loss_thrust_coefficient(
    power_thrust_table: dict,
    velocities: torch.Tensor,
    yaw_angles: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_interp=None,
    average_method="cubic-mean",
    cubature_weights=None,
    correct_cp_ct_for_tilt=False,
):
    rotor_avg_vels = average_velocity(velocities)
    thrust_curve = torch.tensor(power_thrust_table["thrust_coefficient"]).to(velocities.device)
    wind_speeds = torch.tensor(power_thrust_table["wind_speed"]).to(velocities.device)
    ct = interpolation(wind_speeds, thrust_curve, rotor_avg_vels)
    ct = torch.clamp(ct, 0.0001, 0.9999)

    old_tilt = tilt_angles.clone()
    tilt = compute_tilt_angles_for_floating_turbines(tilt_angles, tilt_interp, rotor_avg_vels)
    if not correct_cp_ct_for_tilt:
        tilt = old_tilt

    ct *= torch.cos((yaw_angles)) * torch.cos((tilt - power_thrust_table["ref_tilt"]))
    return ct

def cosine_loss_axial_induction(
    power_thrust_table: dict,
    velocities: torch.Tensor,
    yaw_angles: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_interp=None,
    average_method="cubic-mean",
    cubature_weights=None,
    correct_cp_ct_for_tilt=False,
):
    ct = cosine_loss_thrust_coefficient(
        power_thrust_table, velocities, yaw_angles, tilt_angles,
        tilt_interp, average_method, cubature_weights, correct_cp_ct_for_tilt
    )

    misalignment = torch.cos((yaw_angles)) * torch.cos((tilt_angles - power_thrust_table["ref_tilt"]))
    a = 0.5 / misalignment * (1.0 - torch.sqrt(1.0 - ct * misalignment))
    return a

class CosineLossTurbine:

    @staticmethod
    def power(
        power_thrust_table: dict,
        velocities: torch.Tensor,
        air_density: float,
        yaw_angles: torch.Tensor,
        tilt_angles: torch.Tensor,
        average_method="cubic-mean",
        cubature_weights=None,
        **_
    ) -> torch.Tensor:
        rotor_avg_vels = average_velocity(velocities, average_method)
        v_eff = rotor_velocity_air_density_correction(rotor_avg_vels, air_density, power_thrust_table["ref_air_density"])
        v_eff = rotor_velocity_yaw_cosine_correction(power_thrust_table["cosine_loss_exponent_yaw"], yaw_angles, v_eff)

        #v_eff = rotor_velocity_tilt_cosine_correction(tilt_angles = tilt_angles,
        #                                               ref_tilt = ,
        # #                                             power_thrust_table["ref_tilt"],
        #                                              power_thrust_table["cosine_loss_exponent_tilt"], 
        #                                              v_eff)

        # Replace this with differentiable torch-based interpolation
        wind_speeds = torch.tensor(power_thrust_table["wind_speed"])
        powers = torch.tensor(power_thrust_table["power"])

        power_output = interp1d(wind_speeds, powers, v_eff)
        return power_output * 1e3




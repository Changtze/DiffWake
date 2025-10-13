import torch
from typing import Union, Iterable, Callable


def cosd(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(torch.deg2rad(x))


def rotor_velocity_yaw_cosine_correction(
    cosine_loss_exponent_yaw: Union[float, torch.Tensor],
    yaw_angles: torch.Tensor,
    rotor_effective_velocities: torch.Tensor,
) -> torch.Tensor:
    pW = cosine_loss_exponent_yaw / 3.0
    return rotor_effective_velocities * cosd(yaw_angles) ** pW


def rotor_velocity_tilt_cosine_correction(
    tilt_angles: torch.Tensor,
    ref_tilt: torch.Tensor,
    cosine_loss_exponent_tilt: Union[float, torch.Tensor],
    tilt_interp: Callable[[torch.Tensor], torch.Tensor],
    correct_cp_ct_for_tilt: torch.Tensor,
    rotor_effective_velocities: torch.Tensor,
) -> torch.Tensor:
    old_tilt_angle = tilt_angles.clone()
    interpolated_tilts = tilt_interp(rotor_effective_velocities)
    updated_tilts = torch.where(correct_cp_ct_for_tilt.bool(), interpolated_tilts, old_tilt_angle)
    relative_tilt = updated_tilts - ref_tilt
    return rotor_effective_velocities * cosd(relative_tilt) ** (cosine_loss_exponent_tilt / 3.0)


def average_velocity(
    velocities: torch.Tensor,
    ix_filter: Union[torch.Tensor, Iterable[int], None] = None,
    method: str = "cubic-mean",
    cubature_weights: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]

    axis = tuple(range(2, velocities.ndim))

    if method == "simple-mean":
        return torch.mean(velocities, dim=axis)
    elif method == "cubic-mean":
        return torch.mean(velocities ** 3.0, dim=axis) ** (1.0 / 3.0)
    elif method == "simple-cubature":
        if cubature_weights is None:
            raise ValueError("cubature_weights is required for 'simple-cubature' method.")
        weights = cubature_weights.flatten()
        weights = weights * len(weights) / weights.sum()
        weighted = velocities * weights.view(1, 1, -1, 1)
        return torch.mean(weighted, dim=axis)
    elif method == "cubic-cubature":
        if cubature_weights is None:
            raise ValueError("cubature_weights is required for 'cubic-cubature' method.")
        weights = cubature_weights.flatten()
        weights = weights * len(weights) / weights.sum()
        weighted = (velocities ** 3) * weights.view(1, 1, -1, 1)
        return torch.mean(weighted, dim=axis) ** (1.0 / 3.0)
    else:
        raise ValueError("Invalid averaging method.")


def rotor_velocity_air_density_correction(
    velocities: torch.Tensor,
    air_density: float,
    ref_air_density: float
) -> torch.Tensor:
    return (air_density / ref_air_density) ** (1 / 3) * velocities


def rotor_effective_velocity(
    air_density: float,
    ref_air_density: float,
    velocities: torch.Tensor,
    yaw_angle: torch.Tensor,
    tilt_angle: torch.Tensor,
    ref_tilt: torch.Tensor,
    cosine_loss_exponent_yaw: Union[float, torch.Tensor],
    cosine_loss_exponent_tilt: Union[float, torch.Tensor],
    tilt_interp: Callable[[torch.Tensor], torch.Tensor],
    correct_cp_ct_for_tilt: torch.Tensor,
    ix_filter: Union[torch.Tensor, Iterable[int], None] = None,
    average_method: str = "cubic-mean",
    cubature_weights: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        yaw_angle = yaw_angle[:, ix_filter]
        tilt_angle = tilt_angle[:, ix_filter]
        ref_tilt = ref_tilt[:, ix_filter]
        cosine_loss_exponent_yaw = cosine_loss_exponent_yaw[:, ix_filter]
        cosine_loss_exponent_tilt = cosine_loss_exponent_tilt[:, ix_filter]

    avg_velocities = average_velocity(
        velocities,
        method=average_method,
        cubature_weights=cubature_weights
    )

    rotor_effective = rotor_velocity_air_density_correction(avg_velocities, air_density, ref_air_density)
    rotor_effective = rotor_velocity_yaw_cosine_correction(cosine_loss_exponent_yaw, yaw_angle, rotor_effective)
    rotor_effective = rotor_velocity_tilt_cosine_correction(
        tilt_angle, ref_tilt, cosine_loss_exponent_tilt, tilt_interp,
        correct_cp_ct_for_tilt, rotor_effective
    )

    return rotor_effective

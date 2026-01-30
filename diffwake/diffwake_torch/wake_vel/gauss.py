import torch
from torch import nn
from typing import Any, Dict

def cosd(angle):
    """
    Cosine of an angle with the angle given in degrees.
    :param angle: (float) angle in degrees
    :return: float
    """
    return torch.cos(torch.deg2rad(angle))

def sind(angle):
    """
    Sine of an angle with the angle given in degrees.
    :param angle: (float) angle in degrees
    :return: float
    """
    return torch.sin(torch.deg2rad(angle))

def tand(angle):
    """
    Tangent of an angle with the angle given in degrees.
    :param angle: (float) angle in degrees
    :return: float
    """
    return torch.tan(torch.deg2rad(angle))

def safe_sqrt(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))

def gaussian_function(C, r, n, sigma):
    return C * torch.exp(-1 * r ** n / (2 * sigma ** 2))

def mask_upstream_wake(mesh_y_rotated: torch.Tensor,
                       x_coord_rotated: torch.Tensor,
                       y_coord_rotated: torch.Tensor,
                       turbine_yaw: torch.Tensor) -> torch.Tensor:
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * torch.tan(torch.rad2deg(turbine_yaw)) + x_coord_rotated

    return xR, yR

def rC(wind_veer,
       sigma_y,
       sigma_z,
       y,
       y_i,
       delta,
       z,
       HH,
       Ct,
       yaw,
       D):
    a = cosd(wind_veer) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer) ** 2 / (2 * sigma_z ** 2)
    b = -sind(2 * wind_veer) / (4 * sigma_y ** 2) + sind(2 * wind_veer) / (4 * sigma_z ** 2)
    c = sind(wind_veer) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer) ** 2 / (2 * sigma_z ** 2)
    r = (
        a * (y - y_i - delta) ** 2
        - 2 * b * (y - y_i - delta) * (z - HH)
        + c * (z - HH) ** 2
    )
    C = 1 - safe_sqrt(torch.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0))

    # Precalculate some parts
    twox_sigmay_2 = 2 * sigma_y ** 2
    twox_sigmaz_2 = 2 * sigma_z ** 2
    a = cosd(wind_veer) ** 2 / (twox_sigmay_2) + sind(wind_veer) ** 2 / (twox_sigmaz_2)
    b = -sind(2 * wind_veer) / (2 * twox_sigmay_2) + sind(2 * wind_veer) / (2 * twox_sigmaz_2)
    c = sind(wind_veer) ** 2 / (twox_sigmay_2) + cosd(wind_veer) ** 2 / (twox_sigmaz_2)
    delta_y = y - y_i - delta
    delta_z = z - HH
    r = (a * (delta_y ** 2) - 2 * b * (delta_y) * (delta_z) + c * (delta_z ** 2))
    C = 1 - safe_sqrt(torch.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / (D * D))), 0.0, 1.0))

    return r, C


class GaussVelocityDeficit(nn.Module):
    def __init__(self,
                 alpha = 0.58,
                 beta = 0.077,
                 ka = 0.38,
                 kb = 0.004):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("ka", torch.tensor(ka))
        self.register_buffer("kb", torch.tensor(kb))


    def forward(self,
                x_i: torch.Tensor,
                y_i: torch.Tensor,
                z_i: torch.Tensor,
                axial_induction_i: torch.Tensor,
                deflection_field_i: torch.Tensor,
                yaw_angle_i: torch.Tensor,
                turbulence_intensity_i: torch.Tensor,
                ct_i: torch.Tensor,
                hub_height_i: float,
                rotor_diameter_i: torch.Tensor,  # scalar
                Ctmp: torch.Tensor,
                x: torch.Tensor,
                y: torch.Tensor,
                z: torch.Tensor,
                u_initial: torch.Tensor,
                wind_veer: float,
                ) -> None:

        # yaw_angle is all turbine yaw angles for each wind speed

        # Opposite sign convention in GCH
        yaw_angle = -1 * yaw_angle_i

        # Initialise velocity deficit
        uR = u_initial * ct_i / (2.0 * (1 - safe_sqrt(1 - ct_i)))
        u0 = u_initial * safe_sqrt(1 - ct_i)

        # Initialise lateral bounds
        sigma_z0 = rotor_diameter_i * 0.5 * safe_sqrt(uR / (u_initial + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(wind_veer)

        # Compute bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i.clone()

        # Start of the far wake
        x0 = torch.ones_like(u_initial)
        x0 *= rotor_diameter_i * cosd(yaw_angle) * (1 + safe_sqrt(1 - ct_i))
        x0 /= torch.sqrt(torch.tensor([2])) * (
            4.0 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - safe_sqrt(1 - ct_i))
        )
        x0 += x_i

        # Initialise velocit deficit array
        velocity_deficit = torch.zeros_like(u_initial)

        # Masks
        # When we have only an inequality, the current turbine may be applied its own
        # wake in cases where numerical precision cause in incorrect comparison. We've
        # applied a small bump to avoid this. "0.1" is arbitrary but it is a small, non
        # zero value.

        # This mask defines the near wake; keeps the areas downstream of xR and upstream of x0
        near_wake_mask = (x > xR + 0.1) * (x < x0)
        far_wake_mask = (x >= x0)

        # Compute the velocity deficit in the near wake region
        # ONLY if there are points within the near wake boundary
        if torch.sum(near_wake_mask):
            # Calculate wake expansion
            # Linear ramp from 0 to 1 from the start of the near wake to the start of the fear wake
            near_wake_ramp_up = (x - xR) / (x0 - xR)

            # Another linear ramp, but positive upstream of the far wake and negative in the far wake;
            # 0 at the start of the far wake
            near_wake_ramp_down = (x0 - x) / (x0 - xR)

            sigma_y = near_wake_ramp_down * 0.501 * rotor_diameter_i * safe_sqrt(ct_i / 2.0)
            sigma_y += near_wake_ramp_up * sigma_y0
            sigma_y *= (x >= xR)
            sigma_y += torch.ones_like(sigma_y) * (x < xR) * 0.5 * rotor_diameter_i

            sigma_z = near_wake_ramp_down * 0.501 * rotor_diameter_i * safe_sqrt(ct_i / 2.0)
            sigma_z += near_wake_ramp_up * sigma_z0
            sigma_z *= (x >= xR)
            sigma_z += torch.ones_like(sigma_z) * (x < xR) * 0.5 * rotor_diameter_i

            r, C = rC(
                wind_veer,
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i,
            )

            near_wake_deficit = gaussian_function(C, r, 1, safe_sqrt(0.5))
            near_wake_deficit *= near_wake_mask

            velocity_deficit += near_wake_deficit

            if torch.sum(far_wake_mask):
                ky = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
                kz = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
                sigma_y = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * (x < x0)
                sigma_z = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * (x < x0)

                r, C = rC(
                    wind_veer,
                    sigma_y,
                    sigma_z,
                    y,
                    y_i,
                    deflection_field_i,
                    z,
                    hub_height_i,
                    ct_i,
                    yaw_angle,
                    rotor_diameter_i
                )

                far_wake_deficit = gaussian_function(C, r, 1, safe_sqrt(0.5))
                far_wake_deficit *= far_wake_mask
                velocity_deficit += far_wake_deficit

            return velocity_deficit





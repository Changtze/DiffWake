import torch
from torch import nn
from typing import Any, Dict

def sind(angle):
    """
    Cosine of an angle with the angle given in degrees.
    Args:
        angle (float): angle in degrees
    :param angle:
    :return:
    """

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
        sigma_y0 = sigma_z0 *

        pass



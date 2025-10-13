import torch
import torch.nn as nn

class CrespoHernandez(nn.Module):
    """
    Crespo-Hernandez wake-added turbulence model.
    Computes additional turbulence introduced by turbine wakes based on 
    downstream distance, rotor size, and axial induction.
    
    Args:
        initial (float): Exponent applied to ambient TI.
        constant (float): Scaling constant.
        ai (float): Exponent applied to axial induction.
        downstream (float): Exponent applied to downstream distance (in D).
    """
    def __init__(self, initial=0.1, constant=0.9, ai=0.8, downstream=-0.32):
        super().__init__()
        self.initial = float(initial)
        self.constant = float(constant)
        self.ai = float(ai)
        self.downstream = float(downstream)

    def prepare_function(self) -> dict:
        # No setup needed
        return {}

    def forward(
        self,
        ambient_TI: torch.Tensor,       # shape: (B, 1,1,1) or broadcastable
        x: torch.Tensor,                # shape: (B, T, H, W)
        x_i: torch.Tensor,              # shape: (B, 1, 1, 1)
        rotor_diameter: float,   # shape: float
        axial_induction: torch.Tensor   # shape: (B, 1, 1, 1)
    ) -> torch.Tensor:
        """
        Compute wake-added turbulence using Crespo-Hernandez model.

        Returns:
            torch.Tensor: wake-added turbulence intensity field (B x T x H x W)
        """

        delta_x = x - x_i

        delta_x_safe = torch.where(delta_x > 0.1, delta_x, torch.ones_like(delta_x))

        ti = (
            self.constant
            * (axial_induction ** self.ai)
            * (ambient_TI ** self.initial)
            * ((delta_x_safe / rotor_diameter) ** self.downstream)
        )

        ti = ti * (delta_x > -0.1).float()

        return torch.nan_to_num(ti, posinf=0.0)
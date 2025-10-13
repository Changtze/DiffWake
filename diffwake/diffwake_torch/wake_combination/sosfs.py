import torch
import torch.nn as nn

class SOSFS(nn.Module):
    """
    SOSFS uses sum of squares freestream superposition to combine the
    wake velocity deficits with the base flow field.
    """

    def __init__(self):
        super().__init__()

    def prepare_function(self):
        # No setup needed, but present for API consistency
        return {}

    def forward(self, wake_field: torch.Tensor, velocity_field: torch.Tensor) -> torch.Tensor:
        """
        Combine wake and freestream using sum-of-squares (Euclidean norm).

        Args:
            wake_field (Tensor): Wake velocity deficit.
            velocity_field (Tensor): Base flow velocity.

        Returns:
            Tensor: Combined velocity field.
        """
        return torch.sqrt(wake_field ** 2 + velocity_field ** 2)

import torch 
import math

def mask_non_influencers(mask):
    """
    Identify turbines that do not influence any other turbines.

    Args:
        mask: (B, N, N) boolean tensor, where mask[b, i, j] = True
              if turbine i affects turbine j.

    Returns:
        non_influencers: (B, N) boolean tensor, where True means
                         the turbine does not influence any other.
    """
    # Sum across the outgoing edges: axis 2
    has_outgoing = mask.any(dim=2)  # (B, N)
    non_influencers = ~has_outgoing  # True if no outgoing edges
    return non_influencers

def wake_mask_all(x, y, alpha, L, d):
    """
    Compute a mask of which turbines lie in the wake of others.

    Args:
        x: (B, N) tensor of x positions of turbines
        y: (B, N) tensor of y positions of turbines
        alpha: scalar (wake half-angle in radians)
        L: scalar (wake length)
        d: scalar (turbine diameter)

    Returns:
        mask: (B, N, N) bool tensor, where mask[b, i, j] = True 
              if turbine j is in the wake of turbine i
    """
    B, N = x.shape
    tan_a = math.tan(alpha)

    # (B, N, 1) vs (B, 1, N): pairwise dx and dy
    x_c = x.unsqueeze(2)   # (B, N, 1)
    y_c = y.unsqueeze(2)
    x_o = x.unsqueeze(1)   # (B, 1, N)
    y_o = y.unsqueeze(1)

    dx = x_o - x_c  # (B, N, N)
    dy = y_o - y_c  # (B, N, N)

    # Mask for being downstream within wake length
    in_x = (dx >= 0) & (dx <= L)

    # Wake spread from y_c ± dx * tan(alpha)
    spread = dx * tan_a
    y_min_wake = y_c - spread - d / 2
    y_max_wake = y_c + spread + d / 2

    y_o_min = y_o - d / 2
    y_o_max = y_o + d / 2

    in_y = (y_o_max >= y_min_wake) & (y_o_min <= y_max_wake)

    mask = in_x & in_y  # (B, N, N)

    # Remove self-influence (i == j)
    eye = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)  # (1, N, N)
    mask = mask & ~eye

    return mask

def wake_levels_from_mask(mask):
    """
    Args:
        mask: (B, N, N) bool tensor, where mask[b, i, j] = True 
              if turbine j is in the wake of turbine i (i influences j).

    Returns:
        levels: (B, N) int tensor of influence levels per turbine.
    """
    B, N, _ = mask.shape
    mask = mask.clone()
    levels = torch.full((B, N), -1, dtype=torch.long, device=mask.device)
    remaining = torch.ones_like(levels, dtype=torch.bool)

    level = 0
    while remaining.any():
        current_mask = mask & remaining.unsqueeze(1) & remaining.unsqueeze(2)  # (B, N, N)

        # Turbines with no incoming edges
        has_incoming = current_mask.any(dim=1)  # (B, N)
        sources = ~has_incoming & remaining     # (B, N)

        if not sources.any():
            # Catch cycles or isolated remainder
            levels[remaining] = level
            break

        levels[sources] = level
        remaining = remaining & ~sources  # mark as assigned

        mask = mask & ~sources.unsqueeze(1)  # zero out their row (they no longer influence)

        level += 1

    return levels


def test_wake_levels_from_mask():
    alpha = math.radians(30)
    L = 10.0
    d = 1.0

    # Batch size = 2, Number of turbines = 4
    # Batch 0: 0 → 1 → 2, 3 isolated
    # Batch 1: 0 → 1, 2 and 3 isolated

    x = torch.tensor([
        [0.0, 3.0, 6.5, 10.0],
        [0.0, 3.0, 10.0, 0.0]
    ])  # shape (2, 4)

    y = torch.tensor([
        [0.0, 0.0, 0.0, 10.0],
        [0.0, 0.0, 10.0, -10.0]
    ])  # shape (2, 4)

    mask = wake_mask_all(x, y, alpha, L, d)
    levels = wake_levels_from_mask(mask)

    expected = torch.tensor([
        [0, 1, 2, 0],  # Batch 0: chain 0→1→2, 3 is isolated
        [0, 1, 0, 0]   # Batch 1: 0→1 only
    ], dtype=torch.long)

    assert torch.equal(levels, expected), f"Expected:\n{expected}\nGot:\n{levels}"
    print("Test passed: wake_levels_from_mask")

import torch

def simple_mean(array: torch.Tensor, axis) -> torch.Tensor:
    return array.mean(dim=axis)

def cubic_mean(array: torch.Tensor, axis) -> torch.Tensor:
    mean_cubed = torch.mean(array ** 3.0, dim=axis)
    return torch.copysign(torch.abs(mean_cubed).pow(1/3), mean_cubed)

def average_velocity(
    velocities: torch.Tensor,
    method: str = "cubic-mean"
) -> torch.Tensor:
    """
    Calculates average velocity over the rotor swept area using the specified method.

    Args:
        velocities (torch.Tensor): Tensor with shape (B, T, Ny, Nz)
        method (str): "simple-mean" or "cubic-mean"

    Returns:
        torch.Tensor: Tensor of shape (B, T, 1, 1)
    """
    if velocities.ndim < 3:
        raise ValueError("velocities must have at least 3 dimensions (B, T, Ny, Nz)")

    # axes = last two dimensions for rotor plane
    axis = tuple(range(velocities.ndim - 2, velocities.ndim))

    if method == "simple-mean":
        avg = simple_mean(velocities, axis)
    elif method == "cubic-mean":
        avg = cubic_mean(velocities, axis)
    else:
        raise ValueError(f"Unsupported averaging method: {method}")

    return avg

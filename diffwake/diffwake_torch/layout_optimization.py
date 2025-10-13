import torch
import torch.nn as nn
from pyDOE import lhs
import math
from .LBFGS import LBFGS

def lower_off_diagonal(matrix):

    # Get the lower triangle mask (excluding diagonal)
    mask = torch.tril(torch.ones_like(matrix), diagonal=-1).bool()

    # Apply mask
    lower_elements = matrix[mask]

    return lower_elements

def power_loss_function(x,y, model, weights):
    model.set(
    layout_x=x,
    layout_y=y
    )
    model.run()
    power = model.get_turbine_powers()/1e6
    loss = -(power.mean(dim=1)*(weights.flatten())).sum()
    return loss


def distance_penalty(x, y, min_distance=1.0):
    """
    Compute a penalty if points are closer than min_distance.
    
    Args:
        x: (N,) tensor of x coordinates
        y: (N,) tensor of y coordinates
        min_distance: minimum allowed distance
    Returns:
        penalty: scalar penalty to add to loss
    """
    points = torch.stack([x, y], dim=1)  # (N, 2)
    
    # Compute pairwise distances
    dists = torch.cdist(points, points, p=2)  # (N, N)
    dists = lower_off_diagonal(dists) 
    
    # Penalty: if distance < min_distance
    violation = torch.clamp(min_distance - dists, min=0.0)  # (N, N)
    penalty = (violation**2).sum()

    return penalty


import torch
import torch.nn as nn

def optimize_layout_in_box_LBFGS(model, 
                           x_inits=None, 
                           y_inits=None, 
                           lower=[0, 1000], 
                           upper=[0, 1000], 
                           weights=None,
                           loss_function=power_loss_function, 
                           distance_penalty=distance_penalty,
                           diamter=136.0, 
                           factor=1000.0, 
                           max_iter=40):
    """
    Optimize (x, y) layout inside a bounding box using scaled sigmoid parameters.

    Returns:
        x_opt (Tensor): Optimized x coordinates (scaled back)
        y_opt (Tensor): Optimized y coordinates (scaled back)
        loss_hist (list): Loss values per accepted optimizer step
    """

    if weights is None:
        weights = torch.ones_like(model.flow_field.wind_speeds)
        weights = weights / weights.sum()

    with torch.no_grad():
        power_init = loss_function(x_inits, y_inits, model, weights).detach().abs().item()


    device = x_inits.device
    x_min, y_min = lower[0] / factor, lower[1] / factor
    x_max, y_max = upper[0] / factor, upper[1] / factor

    # Scale to [0, 1] then invert sigmoid
    x_ = x_inits / factor
    y_ = y_inits / factor

    x_scaled = ((x_ - x_min) / (x_max - x_min)).clamp(1e-6, 1 - 1e-6)
    y_scaled = ((y_ - y_min) / (y_max - y_min)).clamp(1e-6, 1 - 1e-6)

    x_inits = torch.log(x_scaled / (1 - x_scaled)).detach().clone().requires_grad_()
    y_inits = torch.log(y_scaled / (1 - y_scaled)).detach().clone().requires_grad_()

    x_inits = nn.Parameter(x_inits)
    y_inits = nn.Parameter(y_inits)

    optimizer = LBFGS(
        [x_inits, y_inits],
        max_iter=max_iter,
        max_eval = max_iter * 2,
        line_search_fn="strong_wolfe",
        history_size=20,
        tolerance_grad=1e-5,
        tolerance_change=1e-5,
    )

    # Use shared variable to capture the loss from the closure
    loss_hist = []
    penalty_weights = []
    n_iter = -1

    def closure():
        nonlocal n_iter
        optimizer.zero_grad()
        x_sig = x_min + (x_max - x_min) * torch.sigmoid(x_inits)
        y_sig = y_min + (y_max - y_min) * torch.sigmoid(y_inits)

        loss = loss_function(x_sig * factor, y_sig * factor, model, weights) / power_init
        d = distance_penalty(x_sig, y_sig, (diamter * 2.01 ) / factor)
        penalty_weight = min(10.0, 1.0 + 10.0 * (len(loss_hist) / max_iter))
        total_loss = loss + d * penalty_weight
        if optimizer.n_iter > n_iter:
            n_iter = optimizer.n_iter
            loss_hist.append(loss.item())
            penalty_weights.append(d.item() * penalty_weight)

        # Save raw loss (without penalty) for the post-step hook
        total_loss.backward()
        return total_loss


    optimizer.step(closure)


    # Rescale optimized parameters
    x_opt = (x_min + (x_max - x_min) * torch.sigmoid(x_inits.detach())) * factor
    y_opt = (y_min + (y_max - y_min) * torch.sigmoid(y_inits.detach())) * factor

    return x_opt, y_opt, loss_hist, penalty_weights

def optimize_layout_in_box_Adam(model, 
                           x_inits=None, 
                           y_inits=None, 
                           lower=[0, 1000], 
                           upper=[0, 1000], 
                           weights=None,
                           loss_function=power_loss_function, 
                           distance_penalty=distance_penalty,
                           diamter=136.0, 
                           factor=1000.0, 
                           max_iter=1000,    # Adam needs more steps
                           lr=0.1):         # Learning rate for Adam
    """
    Optimize (x, y) layout inside a bounding box using scaled sigmoid parameters, with Adam optimizer.

    Args:
        x_inits (Tensor): Initial x coordinates, shape (N,)
        y_inits (Tensor): Initial y coordinates, shape (N,)
        lower (tuple): (x_min, y_min) bounds
        upper (tuple): (x_max, y_max) bounds
        loss_function (callable): Loss function taking (x, y) in original scale
        distance_penalty (callable): Penalty function for (x, y)
        diamter (float): Minimum spacing diameter (used in penalty)
        factor (float): Scale factor to bring values to a manageable range
        max_iter (int): Max Adam steps
        lr (float): Learning rate for Adam

    Returns:
        x_opt (Tensor): Optimized x coordinates (scaled back)
        y_opt (Tensor): Optimized y coordinates (scaled back)
        loss_hist (list): Loss values per iteration
    """

    if weights is None:
        weights = torch.ones_like(model.flow_field.wind_speeds)
        weights = weights / weights.sum()

    device = x_inits.device
    x_min, y_min = lower[0] / factor, lower[1] / factor
    x_max, y_max = upper[0] / factor, upper[1] / factor

    # Scale to [0, 1] then invert sigmoid
    x_ = x_inits / factor
    y_ = y_inits / factor

    x_scaled = ((x_ - x_min) / (x_max - x_min)).clamp(1e-6, 1 - 1e-6)
    y_scaled = ((y_ - y_min) / (y_max - y_min)).clamp(1e-6, 1 - 1e-6)

    x_inits = torch.log(x_scaled / (1 - x_scaled)).detach().clone().requires_grad_()
    y_inits = torch.log(y_scaled / (1 - y_scaled)).detach().clone().requires_grad_()

    x_inits = nn.Parameter(x_inits)
    y_inits = nn.Parameter(y_inits)

    optimizer = torch.optim.AdamW([x_inits, y_inits], lr=lr)

    loss_hist = []

    for step in range(max_iter):
        optimizer.zero_grad()
        
        x_sig = x_min + (x_max - x_min) * torch.sigmoid(x_inits)
        y_sig = y_min + (y_max - y_min) * torch.sigmoid(y_inits)

        loss = loss_function(x_sig * factor, y_sig * factor, model, weights)
        d = distance_penalty(x_sig, y_sig, (diamter * 2 + 1) / factor)

        penalty_weight = min(50.0, 1.0 + 50.0 * (step / max_iter))
        total_loss = loss + d * penalty_weight
        if step == 10: 
            optimizer = torch.optim.AdamW([x_inits, y_inits], lr=0.1)
        total_loss.backward()
        optimizer.step()

        loss_hist.append(total_loss.item())

    # Return optimized and scaled coordinates
    x_opt = (x_min + (x_max - x_min) * torch.sigmoid(x_inits.detach())) * factor
    y_opt = (y_min + (y_max - y_min) * torch.sigmoid(y_inits.detach())) * factor

    return x_opt, y_opt, loss_hist


def get_fps_inits(n_candidates: int, lower: list = [0,0], upper: list = [1000,1000], points_per_unit=0.5):
    xmin, ymin = lower
    xmax, ymax = upper

    # Calculate number of points along each axis based on domain size
    Nx = max(int((xmax - xmin) * points_per_unit), 2)
    Ny = max(int((ymax - ymin) * points_per_unit), 2)

    # Create linspace for x and y in real coordinates
    grid_x = torch.linspace(xmin, xmax, Nx)
    grid_y = torch.linspace(ymin, ymax, Ny)
    grid = torch.cartesian_prod(grid_x, grid_y)  # shape (Nx * Ny, 2)

    # Farthest Point Sampling
    N, D = grid.shape
    selected = torch.zeros(n_candidates, dtype=torch.long)
    distances = torch.full((N,), float('inf'))

    # Randomly pick first point
    selected[0] = torch.randint(0, N, ())

    for i in range(1, n_candidates):
        dist = torch.norm(grid - grid[selected[i-1]], dim=1)
        distances = torch.minimum(distances, dist)
        selected[i] = torch.argmax(distances)

    return grid[selected]


def get_lhs_inits(n_candidates, D = 2, lower = [0,0], upper = [1000,1000]):
    lhs_samples = lhs(D, samples=n_candidates)  # shape: [n_candidates, D]
    lhs_samples = torch.tensor(lhs_samples)
    lhs_samples[:,0] =   lower[0] + (upper[0] - lower[0])*lhs_samples[:,0]
    lhs_samples[:,1] =   lower[1] + (upper[1] - lower[1])*lhs_samples[:,1]
    return lhs_samples




def spread_points_over_grid(n_points: int, lower: list, upper: list, points_per_unit=1.0):
    xmin, ymin = lower
    xmax, ymax = upper

    # Calculate number of points along each axis based on domain size
    Nx = max(int((xmax - xmin) * points_per_unit), 2)
    Ny = max(int((ymax - ymin) * points_per_unit), 2)

    # Create linspace for x and y in real coordinates
    grid_x = torch.linspace(xmin, xmax, Nx)
    grid_y = torch.linspace(ymin, ymax, Ny)
    grid = torch.cartesian_prod(grid_x, grid_y)  # shape (Nx * Ny, 2)

    # Farthest Point Sampling
    N, D = grid.shape
    selected = torch.zeros(n_points, dtype=torch.long)
    distances = torch.full((N,), float('inf'))

    # Randomly pick first point
    selected[0] = int(N/2)#torch.randint(0, N, ())

    for i in range(1, n_points):
        dist = torch.norm(grid - grid[selected[i-1]], dim=1)
        distances = torch.minimum(distances, dist)
        selected[i] = torch.argmax(distances)

    return grid[selected]
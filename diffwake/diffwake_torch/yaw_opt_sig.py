import torch
import torch.nn as nn
from pyDOE import lhs
import math
from .LBFGS import LBFGS


"""
Smooth sigmoid reparameterisation to help with L-BFGS convergence on constrained yaw angle optimisation
"""

def power_loss_function(yaw_angles, model):
    model.set(
    yaw_angles=yaw_angles,
    )
    model.run()
    power = model.get_turbine_powers()/1e6
    loss = -power.mean()
    return loss




def optimize_yaw_angles_LBFGS(model, 
                           yaw_inits=None, 
                           max_angle = 10, 
                           loss_function=power_loss_function, 
                           max_iter=40):
    """
    Optimize (x, y) layout inside a bounding box using scaled sigmoid parameters.

    Returns:
        x_opt (Tensor): Optimized x coordinates (scaled back)
        y_opt (Tensor): Optimized y coordinates (scaled back)
        loss_hist (list): Loss values per accepted optimizer step
    """

    factor = max_angle

    with torch.no_grad():
        power_init = loss_function(yaw_inits, model).detach().abs().item()

    yaw_inits = yaw_inits/factor

    eps = 1e-6
    yaw_inits = yaw_inits.clamp(eps, 1 - eps)
    
    print(yaw_inits)
    yaw_inits = torch.log(yaw_inits / (1 - yaw_inits)).detach().clone().requires_grad_()

    yaw_inits = nn.Parameter(yaw_inits)

    optimizer = LBFGS(
        [yaw_inits],
        max_iter=max_iter,
        max_eval = max_iter * 2,
        line_search_fn="strong_wolfe",
        history_size=10,
        tolerance_grad=1e-6,
        tolerance_change=1e-6,
    )

    # Use shared variable to capture the loss from the closure
    loss_hist = []
    n_iter = -1

    def closure():
        nonlocal n_iter
        optimizer.zero_grad()
        yaw_sig = factor * torch.sigmoid(yaw_inits)


        loss = loss_function(yaw_sig, model) / power_init
        print("Grad:", yaw_inits.grad)

        print(optimizer.n_iter)
        if optimizer.n_iter > n_iter:

            n_iter = optimizer.n_iter
            loss_hist.append(loss.item())

        # Save raw loss (without penalty) for the post-step hook
        loss.backward()
        print("Grad:", yaw_inits.grad)

        print(loss)
        return loss


    optimizer.step(closure)
    # Rescale optimized parameters
    yaw_opt = torch.sigmoid(yaw_inits.detach()) * factor

    return yaw_opt, loss_hist
 

import torch
import torch.nn as nn
from pyDOE import lhs
import math
from .LBFGS import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau



def power_loss_function(yaw_angles, model):
    model.set(
    yaw_angles=yaw_angles,
    )
    model.run()
    power = model.get_turbine_powers()/1e6
    loss = -power.mean()
    return loss



def soft_bounds_penalty(x, eps=0.0, weight=1.0):
    below = torch.relu(-x + eps)       # penalizes x < 0
    above = torch.relu(x - 1 - eps)    # penalizes x > 1
    return weight * (below + above).mean()



def optimize_yaw_angles_Adam(
    model,
    yaw_inits=None,
    max_angle=10,
    loss_function=power_loss_function,
    max_iter=100,
    lr=0.05,
):
    """
    Optimize yaw angles in [0, max_angle] using AdamW and soft constraints.

    Args:
        model: differentiable model accepting yaw angles in degrees
        yaw_inits (Tensor): initial yaw angles in degrees [N]
        max_angle (float): maximum allowed yaw in degrees
        loss_function (callable): computes loss from yaw and model
        max_iter (int): optimization steps
        lr (float): AdamW learning rate
        penalty_weight (float): strength of soft bound regularization

    Returns:
        yaw_opt (Tensor): optimized yaw angles [N]
        loss_hist (list): raw (unpenalized) loss over iterations
    """
    unsort_indicies = model.grid.unsorted_indices[:,:,1,1]
    non_influencers = torch.gather( model.grid.non_influencers_sorted, dim=1, index=unsort_indicies)

    yaw_inits  = yaw_inits.clamp(0,max_angle)
    yaw_inits[non_influencers] = 0.0
    with torch.no_grad():
        power_init = loss_function(yaw_inits, model).detach().abs().item()



    # Normalize yaw to [0, 1]
    yaw_inits = nn.Parameter(yaw_inits.clone().detach() / max_angle)

    optimizer = torch.optim.AdamW([yaw_inits], lr=lr, weight_decay=0, betas = (0.2, 0.7))


    scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 0.2 ** (step // 5))
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True,eps = 1e-2)

    loss_hist = []


    for i in range(max_iter):
        optimizer.zero_grad()
        yaw = torch.nn.functional.relu(yaw_inits) 
        yaw = max_angle * yaw.clamp( max = 1.)
        loss = loss_function(yaw, model) + power_init
        loss.backward()
        torch.nn.utils.clip_grad_norm_([yaw_inits], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())

        if i % 2 == 0 or i == max_iter - 1:
            print(f"[Step {i:03d}] Loss: {loss.item():.6f} Power gain: {-loss.item() * 100:.3f} %")

    yaw_opt = ((torch.nn.functional.relu(yaw_inits).detach()).clamp( max = 1.) * max_angle)
    return yaw_opt, loss_hist

def optimize_yaw_angles_in_levels_Adam2(
    model,
    yaw_inits=None,
    max_angle=10,
    loss_function=power_loss_function,
    max_iter=100,
    lr=0.05,
):
    """
    Optimize yaw angles per wake level using AdamW.
    At each level, previously optimized angles are kept fixed.
    """
    wake_levels_sorted = model.grid.wake_levels_sorted  # (B, N)
    unsort_indicies = model.grid.unsorted_indices[:,:,1,1]
    wake_levels = torch.gather(wake_levels_sorted, dim=1, index=unsort_indicies)
    influencers = torch.gather( model.grid.non_influencers_sorted == False, dim=1, index=unsort_indicies)

    yaw_inits = torch.zeros_like(wake_levels).float()

    iter_per_all_levels = 3
    max_iter_per_level = 5
    loss_hist = []

    with torch.no_grad():
        power_init = loss_function(yaw_inits, model).abs().item()

    # Running total of yaw angles (updated level-by-level)
    yaw_accum = yaw_inits.clone().detach()

    for k in range(iter_per_all_levels):
        if k == 1:
            max_iter_per_level = 2
        for level in range(wake_levels.max() ):
            level_mask = wake_levels == level
            active_mask = torch.logical_and(level_mask,influencers)  # (B, N)

            if k ==0:
                yaw_accum[active_mask] = max_angle/3
            # Normalize current total yaw to [0,1]
            yaw_param = nn.Parameter((yaw_accum / max_angle).clone())
            optimizer = torch.optim.AdamW([yaw_param], lr=lr/(k + 2), weight_decay=0.0, betas=(0.2, 0.7))

            for i in range(max_iter_per_level):
                optimizer.zero_grad()

                yaw = torch.relu(yaw_param).clamp(max=1.0) * max_angle
                loss = loss_function(yaw, model) + power_init

                loss.backward()
                torch.nn.utils.clip_grad_norm_([yaw_param], max_norm=1.0)

                # Zero gradients for non-active turbines
                with torch.no_grad():
                    yaw_param.grad *= active_mask.float()

                optimizer.step()
                loss_hist.append(loss.item())

                if i % 2 == 0 or i == max_iter_per_level - 1:
                    print(f"[Level {level} | Step {i:02d}] Loss: {loss.item():.6f} | ΔPower: {( loss.item()*(-1)/power_init) * 100:.2f}%")

            # Update the accumulated yaw angles with optimized results at this level
            with torch.no_grad():
                updated_yaw = torch.relu(yaw_param).clamp(max=1.0) * max_angle
                yaw_accum[active_mask] = updated_yaw[active_mask]

    return yaw_accum.detach(), loss_hist


import torch
import torch.nn as nn
from .LBFGS import LBFGS


def optimize_yaw_angles_in_levels_LBFGS(
    model,
    yaw_inits=None,
    max_angle=10,
    loss_function=power_loss_function,
    max_iter=100,
    lr=0.05,
):
    """
    Optimize yaw angles in [0, max_angle] using AdamW and soft constraints.

    Args:
        model: differentiable model accepting yaw angles in degrees
        yaw_inits (Tensor): initial yaw angles in degrees [N]
        max_angle (float): maximum allowed yaw in degrees
        loss_function (callable): computes loss from yaw and model
        max_iter (int): optimization steps
        lr (float): AdamW learning rate
        penalty_weight (float): strength of soft bound regularization

    Returns:
        yaw_opt (Tensor): optimized yaw angles [N]
        loss_hist (list): raw (unpenalized) loss over iterations
    """
    yaw_inits  = yaw_inits.clamp(0,max_angle)

    with torch.no_grad():
        power_init = loss_function(yaw_inits, model).detach().abs().item()


    wake_levels = model.grid.wake_levels

    # Normalize yaw to [0, 1]
    yaw_inits = nn.Parameter(yaw_inits.clone().detach() / max_angle)

    iter_per_all_levels = 2
    loss_hist = []
    print_losses = True
    optimizer = LBFGS(
        [yaw_inits],
        max_iter=5,
        max_eval=10,
        tolerance_grad=1e-6,
        tolerance_change=1e-7,
        line_search_fn="strong_wolfe"
    )
    n_iter = 0

    for k in range(iter_per_all_levels):
        for level in range(wake_levels.max() + 1):
            active_mask = (wake_levels == level).float()  # shape (B, N)

            def closure():
                nonlocal n_iter
                nonlocal active_mask
                optimizer.zero_grad()

                yaw = torch.nn.functional.relu(yaw_inits) + 1e-4
                yaw = max_angle * yaw.clamp( max = 1.)
                loss = loss_function(yaw, model) + power_init

                loss.backward()

                # Log accepted step losses
                if optimizer.n_iter > n_iter:
                    n_iter = optimizer.n_iter
                    loss_hist.append(loss.item())
                    if print_losses is True:
                        print(f"[Step {n_iter:03d}] Loss: {loss :.6f}")
                with torch.no_grad():
                    yaw_inits.grad *= active_mask  # gradient masking

                return loss
            optimizer.step(closure)

    yaw_opt = ((torch.nn.functional.relu(yaw_inits).detach()).clamp( max = 1.) * max_angle)
    return yaw_opt, loss_hist


def optimize_yaw_angles_Adam_then_LBFGS(
    model,
    yaw_inits=None,
    max_angle=10,
    loss_function=power_loss_function,
    max_iter=100,
    lr=0.05,
):
    yaw_inits = yaw_inits.clamp(0, max_angle)
    with torch.no_grad():
        power_init = loss_function(yaw_inits, model).detach().abs().item()

    # Normalize and make parameter
    yaw_inits = nn.Parameter(yaw_inits.clone().detach() / max_angle)

    # Adam phase
    optimizer = torch.optim.Adam([yaw_inits], lr=lr, weight_decay=0.0, betas=(0.2, 0.7))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.5 ** (step // 5))

    loss_hist = []

    for i in range(max_iter):
        optimizer.zero_grad()
        yaw = yaw_inits.relu().clamp(0, 1) * max_angle
        loss = loss_function(yaw, model) + power_init
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_hist.append(loss.item())

        if i % 2 == 0 or i == max_iter - 1:
            print(f"[Adam Step {i:03d}] Loss: {loss.item():.6f} Power gain: {-loss.item() * 100:.3f} %")

    # Final L-BFGS step
    print("\n🔁 Starting final L-BFGS refinement...")

    def closure():
        optimizer_lbfgs.zero_grad()
        yaw = yaw_inits.relu().clamp(0, 1) * max_angle
        loss = loss_function(yaw, model) + power_init
        loss.backward()
        return loss

    optimizer_lbfgs = LBFGS(
        [yaw_inits],
        max_iter=1,
        max_eval=5,
        tolerance_grad=1e-7,
        tolerance_change=1e-8,
        line_search_fn="strong_wolfe"
    )

    final_loss = optimizer_lbfgs.step(closure)
    print(f"✅ Final L-BFGS loss: {final_loss:.6f} Power gain: {-final_loss * 100:.3f} %")

    yaw_opt = yaw_inits.detach().relu().clamp(0, 1) * max_angle
    return yaw_opt, loss_hist


import torch
import torch.nn as nn

def optimize_yaw_angles_LBFGS(
    model,
    yaw_inits=None,
    max_angle=10,
    loss_function=power_loss_function,
    max_iter=20,
    print_losses = False,

):
    """
    Optimize yaw angles in [0, max_angle] using LBFGS and sigmoid reparameterization.

    Args:
        model: differentiable model accepting yaw angles in degrees
        yaw_inits (Tensor): initial yaw angles in degrees [N]
        max_angle (float): maximum allowed yaw in degrees
        loss_function (callable): computes loss from yaw and model
        max_iter (int): LBFGS outer iterations
        penalty_weight (float): strength of regularization (if needed)

    Returns:
        yaw_opt (Tensor): optimized yaw angles [N]
        loss_hist (list): raw (unpenalized) loss per step
    """

    factor = max_angle

    with torch.no_grad():
        power_init = loss_function(yaw_inits, model).detach().abs().item()

    # --- NEW: map yaw_init ∈ [0, factor] → z_init ∈ ℝ via sigmoid⁻¹
    eps = 1e-4
    norm_yaw = (yaw_inits / max_angle).clamp(eps, 1 - eps)
    #z_init = (1.0 / alpha) * torch.log(norm_yaw / (1 - norm_yaw))
    z_param = nn.Parameter(norm_yaw.clone())


    optimizer = LBFGS(
        [z_param],
        max_iter=max_iter,
        max_eval=max_iter * 2,
        line_search_fn="strong_wolfe",
        history_size=4,
        tolerance_grad=1e-5,
        tolerance_change=1e-6,
    )

    loss_hist = []
    n_iter = -1

    def closure():
        nonlocal n_iter
        optimizer.zero_grad()

        yaw = z_param.relu()
        yaw = factor * yaw.clamp(0, 1)
        loss = loss_function(yaw, model)
        loss.backward()

        # Log accepted step losses
        if optimizer.n_iter > n_iter:
            n_iter = optimizer.n_iter
            loss_hist.append(loss_raw.item() / power_init)
            if print_losses is True:
                print(f"[Step {n_iter:03d}] Loss: {loss_raw / power_init:.6f}")

        return loss

    optimizer.step(closure)
    with torch.no_grad():

        yaw = z_param.relu()
        yaw = factor * yaw.clamp(0, 1 )
        loss_raw = loss_function(yaw, model)    
        loss_hist.append(loss_raw.item() / power_init)

    return yaw, loss_hist

"""
DiffWake/JAX: deterministic wind turbine yaw angle optimisation with LBFGS (and other optax optimisers)
"""

from __future__ import annotations
import argparse
import csv
import functools
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

import jax
import jax.numpy as jnp

import numpy as np
import bayex

# DiffWake imports
from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.yaw_runner import make_yaw_runner
from diffwake.diffwake_jax.util import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power


@dataclass
class YawConstraints:
    gamma_min: float  # minimum allowable yaw angle
    gamma_max: float  # maximum allowable yaw angle

    def mins(self, dtype) -> jax.Array:
        return jnp.deg2rad(jnp.array([self.gamma_min], dtype=dtype))

    def maxs(self, dtype) -> jax.Array:
        return jnp.deg2rad(jnp.array([self.gamma_max], dtype=dtype))


def setup_dtype(use_float64: bool = True) -> jnp.dtype:
    jax.config.update("jax_enable_x64", use_float64)
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _latin_hypercube_unit(key, M: int, D: int, dtype) -> jax.Array:
    base = jnp.arange(M, dtype=jnp.int32)
    keys = jax.random.split(key, D + 1)
    k_jit, k_perms = keys[0], keys[1:]
    jitter = jax.random.uniform(k_jit, shape=(D, M), dtype=dtype)

    def _perm_one(k):
        return jax.random.permutation(k, base)

    perms = jax.vmap(_perm_one)(k_perms)
    U = (perms.astype(dtype) + jitter) / dtype(M)
    return U.T


def sample_initial_yaw_lhs(key,
                           M: int,
                           N: int,
                           constraints: YawConstraints,
                           dtype) -> jax.Array:
    U = _latin_hypercube_unit(key, M, 1 * N, dtype)
    U = U.reshape(M, N, 1)
    mins = constraints.mins(dtype)
    maxs = constraints.maxs(dtype)
    gamma = mins + (maxs - mins) * U[:, :, 0]
    return gamma


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimise yaw angles in DiffWake/JAX with Bayex"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/horn"),
                   help="Directory with weather data and YAML configs.")
    p.add_argument("--farm-yaml", type=str, default="gch.yaml",
                   help="Farm configuration YAML (relative to --data-dir).")
    p.add_argument("--turbine-yaml", type=str, default="vestas_v802MW.yaml",
                   help="Turbine configuration YAML (relative to --data-dir).")
    p.add_argument("--weather-npz", type=str, default="benchmark.npz",
                   help="Weather data file (relative to --data-dir).")

    p.add_argument("--gamma-min", type=float, default=0.0, help="Minimum allowable yaw angle in degrees")
    p.add_argument("--gamma-max", type=float, default=25.0, help="Maximum allowable yaw angle in degrees")
    p.add_argument("--diameter", type=float, default=80.0, help="Turbine rotor diameter (m).")
    p.add_argument("--acq", type=str, default="EI", help="Acquisition function: EI, PI, UCB or LCB")
    p.add_argument("--lcb-kappa", type=float, default=2.576, help="Kappa parameter for LCB acquisition")
    p.add_argument("--ucb-kappa", type=float, default=0.01, help="Kappa parameter for UCB acquisition")

    p.add_argument("--max-iter", type=int, default=200, help="Maximum number of optimiser iterations.")
    p.add_argument("--patience", type=int, default=10, help="Early stop if loss change is less than min_delta")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--min-delta", type=float, default=1e-6, help="Minimum change in loss to continue optimisation.")

    p.add_argument("--float64", action="store_true", help="Enable float64. Default is float32.")
    p.add_argument("--out-dir", type=Path, default=Path("results/yaw_bayes"), help="Base output directory.")
    return p.parse_args()


def build_state_runner(
        data_dir: Path,
        farm_yaml: str,
        turbine_yaml: str,
        wind_dir_rad: jax.Array,
        wind_speed: jax.Array,
        turb_intensity: jax.Array,
        dtype,
):
    cfg = load_input(
        str(data_dir / farm_yaml),
        str(data_dir / turbine_yaml),
    ).set(
        wind_directions=wind_dir_rad,
        wind_speeds=wind_speed,
        turbulence_intensities=turb_intensity,
    )
    state = create_state(cfg)
    runner = make_yaw_runner(state)

    x0 = jnp.asarray(cfg.layout["layout_x"], dtype=dtype)
    N = int(x0.shape[0])

    return state, runner, N


def make_losses(state,
                runner,
                weights: jax.Array,
                dtype):


    def loss_from_yaw(yaw_angles_flat: jnp.ndarray):
        """Physics-based objective: negative total farm power (weighted over wind cases)."""
        # Reshape back to (1, N) for the diffwake runner
        yaw_angles = yaw_angles_flat.reshape(1, -1)
        out = runner(yaw_angles)
        vel = average_velocity_jax(out.u_sorted)
        pow_mw = power(
            state.farm.power_thrust_table,
            vel,
            state.flow.air_density,
            yaw_angles=yaw_angles
        )
        case_power = jnp.sum(pow_mw, axis=1)
        return -jnp.sum(case_power * weights) / 1e6  # convert to megawatts

    return loss_from_yaw


def save_run(out_dir: Path,
             *,
             best_yaw: jax.Array,
             best_power_MW: float,
             baseline_power_MW: float,
             power_increase_pct: float,
             final_loss: float,
             per_case_power_MW: jax.Array,
             wind_dir_deg: np.ndarray,
             wind_speed: np.ndarray,
             weights: np.ndarray,
             config_meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "arrays.npz",
        best_yaw=np.asarray(best_yaw),
        per_case_power_MW=np.asarray(per_case_power_MW),
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed,
        weights=weights,
        config_meta=config_meta
    )

    with open(out_dir / "per_case_power.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wind_dir_deg", "wind_speed", "weight", "mean_power_MW"])
        for d, s, wt, p in zip(wind_dir_deg, wind_speed, weights, per_case_power_MW):
            w.writerow([float(d), float(s), float(wt), float(p)])

    meta = dict(
        best_power_MW=float(best_power_MW),
        baseline_power_MW=float(baseline_power_MW),
        power_increase_pct=float(power_increase_pct),
        final_loss=float(final_loss),
        N_turbines=int(best_yaw.shape[1]),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config_meta,
    )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return None


def main():
    # Setup and I/O
    args = parse_args()
    DTYPE = setup_dtype(args.float64)

    yaw_constraints = YawConstraints(args.gamma_min, args.gamma_max)
    diameter = float(args.diameter)

    data_dir = args.data_dir
    npz_path = data_dir / args.weather_npz
    if not npz_path.is_file():
        raise FileNotFoundError(f"Weather file not found: {npz_path}")

    # Load weather data from npz
    wd = np.load(npz_path)
    if not all(k in wd for k in ("wind_direction", "wind_speed", "weight")):
        raise KeyError("weather_data.npz must contain 'wind_direction', 'wind_speed', 'weight'.")

    # Appropriate conversions for wind conditions
    wind_dir_rad = jnp.deg2rad(jnp.asarray(wd["wind_direction"], dtype=DTYPE))
    wind_speed = jnp.asarray(wd["wind_speed"], dtype=DTYPE)
    weights = jnp.asarray(wd["weight"], dtype=DTYPE).reshape(-1)
    turbulence = jnp.full_like(wind_dir_rad, 0.06, dtype=DTYPE)

    # Build state and simulation runner
    state, runner, N = build_state_runner(
        data_dir, args.farm_yaml, args.turbine_yaml, wind_dir_rad, wind_speed, turbulence, DTYPE
    )

    baseline_yaw = jnp.full((1, N), 0.0, dtype=DTYPE)

    # Non-zero arbitrary starting yaw
    loss_from_yaw = make_losses(state, runner, weights, DTYPE)
    baseline_power = -loss_from_yaw(baseline_yaw)

    # Min and max yaw in radians
    gamma_min = yaw_constraints.mins(DTYPE).item()
    gamma_max = yaw_constraints.maxs(DTYPE).item()

    key = jax.random.PRNGKey(args.seed)

    # Define Bayex domain
    domain = {f'x{i}': bayex.domain.Real(gamma_min, gamma_max) for i in range(N)}

    # Initialize the Optimizer with the base acquisition string
    optimizer = bayex.Optimizer(domain=domain, maximize=False, acq=args.acq)

    # Overwrite the acquisition function to inject custom kappas if applicable
    acq_upper = args.acq.upper()
    if acq_upper == "UCB":
        custom_acq = functools.partial(bayex.acq.upper_confidence_bounds, kappa=args.ucb_kappa)
        optimizer.acq = jax.jit(custom_acq)
    elif acq_upper == "LCB":
        custom_acq = functools.partial(bayex.acq.lower_confidence_bounds, kappa=args.lcb_kappa)
        optimizer.acq = jax.jit(custom_acq)
    # Note: EI and PI will default to their standard xi=0.01 as defined in the library

    # Define prior evaluations to initialise the GP
    num_init_evals = int(3 * N) # number of initial evaluations = 2 * N_turbines. Conservative value for cheap hardware
    key, subkey = jax.random.split(key)


    init_evals = sample_initial_yaw_lhs(key, num_init_evals - 1, N, yaw_constraints, DTYPE)
    unyawed_eval = jnp.zeros_like(init_evals[0])

    # Not all initial samples should be LHS. Have one evaluation unyawed
    init_evals = jnp.concatenate([init_evals, unyawed_eval.reshape(1, N)], axis=0)
    params = {f'x{i}': init_evals[:, i].astype(DTYPE) for i in range(N)}

    ys = jax.vmap(loss_from_yaw)(init_evals).astype(DTYPE)

    # Memory debugging
    # ys_list = []
    # for i in range(num_init_evals):
    #     # Pass a single row but keep it 2D (1, N) for the runner
    #     single_loss = loss_from_yaw(init_evals[i])
    #     jax.block_until_ready(single_loss) # Force execution and clear memory
    #     ys_list.append(single_loss)
    # ys = jnp.array(ys_list, dtype=DTYPE)

    opt_state = optimizer.init(ys, params)

    best_power = 0
    best_yaw = None
    best_loss = jnp.inf

    last_iter = 0
    no_improve = 0

    prev_loss = jnp.inf

    t0 = time.time()
    for iter in range(1, args.max_iter + 1):
        t1 = time.time()
        key, subkey = jax.random.split(key)
        params = optimizer.sample(subkey, opt_state)
        gamma = jnp.array([params[f'x{i}'] for i in range(N)])
        loss = loss_from_yaw(gamma.reshape(1, N))
        jax.block_until_ready(loss)
        loss_val = float(loss)
        opt_state = optimizer.fit(opt_state, loss, params)

        last_iter = iter

        # Update best loss
        if loss_val <= best_loss:
            best_power = -loss_val
            best_yaw = gamma.reshape(1, N)
            best_loss = loss_val
            prev_loss = loss_val

        if prev_loss - loss_val >= float(args.min_delta):
            no_improve = 0
            prev_loss = loss_val
        else:
            no_improve += 1

        if iter % 1 == 0:
            print(
                f"Iter {iter:04d}, loss={float(loss):.6e}  "
                f"No improvement count = {no_improve}, time={time.time()-t1:.3f}s"
            )

        if no_improve >= args.patience:
            print(
                f"Stopping early at iteration {iter} due to no improvement for {args.patience} iterations."
            )
            break
    print(f"Bayesian optimisation: {time.time()-t0:.3f}s, iters={last_iter}")

    final_loss = loss_from_yaw(gamma.reshape(1, N))
    jax.block_until_ready(final_loss)
    final_power = -float(final_loss)

    if final_power > best_power:
        best_power = final_power
        best_yaw = gamma.reshape(1, N)
        best_loss = float(final_loss)

    elapsed_time = time.time() - t0
    print(f"Best mean power (MW): {best_power:.6f}")
    print("Best yaw angles (degrees): ")
    print(np.rad2deg(np.array(best_yaw)))

    out = runner(best_yaw)
    vel = average_velocity_jax(out.u_sorted)

    pow_mw = power(
        state.farm.power_thrust_table,
        vel,
        state.flow.air_density,
        yaw_angles=best_yaw,
    ) / 1e6
    per_case_power_MW = jnp.sum(pow_mw,  axis=1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    wind_dir_deg = np.asarray(jnp.rad2deg(wind_dir_rad), dtype=float)
    wind_speed_np = np.asarray(wind_speed, dtype=float)
    weights_np = np.asarray(weights.reshape(-1), dtype=float)

    config_meta = dict(
        optimizer="bayes-jax",
        maxiter=int(args.max_iter),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        seed=int(args.seed),
        acq=str(args.acq),
        num_init_evals=int(num_init_evals),
        float64=bool(args.float64),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
        diameter=float(diameter),
        dtype="float64" if DTYPE == jnp.float64 else "float32",
        elapsed_time=float(elapsed_time),
        lcb_kappa=float(args.lcb_kappa),
        ucb_kappa=float(args.ucb_kappa),
    )

    save_run(
        out_dir,
        best_yaw=best_yaw,
        best_power_MW=best_power,
        baseline_power_MW=baseline_power,
        power_increase_pct=((best_power - baseline_power)/baseline_power * 100),
        final_loss=best_loss,
        per_case_power_MW=per_case_power_MW,
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed_np,
        weights=weights_np,
        config_meta=config_meta,
    )

if __name__ == "__main__":
    main()
"""
DiffWake/JAX: deterministic wind turbine yaw angle optimisation with LBFGS (and other optax optimisers)

"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

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


def omega_from_gamma_sig(gamma: jax.Array,
                          gamma_max: jax.Array,
                          eps=1e-7) -> jax.Array:
    """
    Map points in [gamma_min, gamma_max] to unconstrained omega via sigmoid function
    :param gamma: JAX array of physical yaw angles in radians
    :param gamma_max: maximum allowable physical yaw angle in radians
    :param gamma_min: minimum allowable physical yaw angle in radians
    """
    p = gamma / gamma_max
    p = jnp.clip(p, eps, 1.0 - eps)  # avoid log(0) or log(1)

    return jnp.log(p / (1.0 - p))


def gamma_from_omega_sig(omega: jax.Array,
                         gamma_max: jax.Array,
                         ) -> jax.Array:
    return gamma_max * jax.nn.sigmoid(omega)


def yaw_penalty_sq(gamma: jax.Array,
                   gamma_min: jax.Array,
                   gamma_max: jax.Array,
                   ) -> jax.Array:
    violation_min = jax.nn.softplus(gamma_min - gamma)
    violation_max = jax.nn.softplus(gamma - gamma_max)

    return jnp.sum(violation_min + violation_max)



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
    """
    (M, N, 2) Latin hypercube samples
    :param key: JAX PRNG key
    :param M: number of samples
    :param N: number of turbines
    :param constraints: yaw angle constraints
    :param dtype: custom dtype (JAX defaults to jax.float32)
    """

    U = _latin_hypercube_unit(key, M, 1 * N, dtype)
    U = U.reshape(M, N, 1)
    mins = constraints.mins(dtype)
    maxs = constraints.maxs(dtype)
    gamma = mins + (maxs - mins) * U[:, :, 0]
    return gamma


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimise yaw angles in DiffWake/JAX with <TBC>"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/horn"),
                   help="Directory with weather data and YAML configs.")
    p.add_argument("--farm-yaml", type=str, default="cc_simple.yaml",
                   help="Farm configuration YAML (relative to --data-dir).")
    p.add_argument("--turbine-yaml", type=str, default="vestas_v802MW.yaml",
                   help="Turbine configuration YAML (relative to --data-dir).")
    p.add_argument("--weather-npz", type=str, default="weather_data.npz",
                   help="Weather data file (relative to --data-dir).")

    # Domain & constraints
    p.add_argument("--gamma-min", type=float, default=0.0, help="Minimum allowable yaw angle in degrees")
    p.add_argument("--gamma-max", type=float, default=25.0, help="Maximum allowable yaw angle in degrees")
    p.add_argument("--diameter", type=float, default=80.0, help="Turbine rotor diameter (m).")
    p.add_argument("--init-mode", choices=["lhs", "perturb"], default="lhs",
                   help="How to initialize restarts.")

    # Optimisation configuration
    p.add_argument("--penalty-weight", type=float, default=1e-3, help="Weight of yaw penalty term in objective.")
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=200, help="Maximum number of optimiser iterations.")
    p.add_argument("--patience", type=int, default=10, help="Early stop if loss change is less than min_delta for this many steps")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--min-delta", type=float, default=1e-7, help="Minimum change in loss to continue optimisation.")

    # Output
    p.add_argument("--float64", action="store_true", help="Enable float64. Default is float32.")
    p.add_argument("--out-dir", type=Path, default=Path("results/yaw_lbfgs"), help="Base output directory.")
    return p.parse_args()


# Do we write a state_runner or yaw_runner function?
# Well state runner is just to run a wind condition or a "state"

def build_state_runner(
        data_dir: Path,
        farm_yaml: str,
        turbine_yaml: str,
        wind_dir_rad: jax.Array,
        wind_speed: jax.Array,
        turb_intensity: jax.Array,
        dtype,
):
    """
    :param data_dir: parent directory of turbine, farm and weather data
    :param farm_yaml: YAML file for farm layout
    :param turbine_yaml: YAML file for turbine characteristics
    :param wind_dir_rad: JAX array of wind directions in radians
    :param wind_speed: JAX array of wind speeds in m/s
    :param turb_intensity: JAX array of turbulence intensity values
    :param dtype: custom dtype (JAX defaults to jax.float32)
    :return:
    """
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

    # N from config
    x0 = jnp.asarray(cfg.layout["layout_x"], dtype=dtype)
    N = int(x0.shape[0])

    return state, runner, N


def make_losses(state,
                runner,
                weights: jax.Array,
                penalty_weight: float,
                dtype):

    pw = jnp.asarray(penalty_weight, dtype=dtype)

    def loss_from_yaw(yaw_angles: jnp.ndarray):
        """Physics-based objective: negative total farm power (weighted over wind cases)."""
        out = runner(yaw_angles)
        vel = average_velocity_jax(out.u_sorted)
        pow_mw = power(
            state.farm.power_thrust_table,
            vel,
            state.flow.air_density,
            yaw_angles=state.farm.yaw_angles
        )
        case_power = jnp.sum(pow_mw, axis=1)
        return -jnp.sum(case_power * weights)  # scalar

    def loss_from_omega(omega: jax.Array,
                        gamma_min: jax.Array,
                        gamma_max: jax.Array) -> jax.Array:
        gammas = gamma_from_omega_sig(omega, gamma_max)
        phys = loss_from_yaw(gammas)
        yaw = yaw_penalty_sq(gammas, gamma_min, gamma_max)
        return phys + pw * yaw

    return loss_from_yaw, loss_from_omega, pw

# Save helpers
def save_run(out_dir: Path,
             *,
             best_yaw: jax.Array,
             best_omega: jax.Array,
             best_power_MW: float,
             final_loss: float,
             final_yaw_penalty: float,
             per_case_power_MW: jax.Array,
             wind_dir_deg: np.ndarray,
             wind_speed: np.ndarray,
             weights: np.ndarray,
             config_meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "arrays.npz",
        best_yaw=np.asarray(best_yaw),
        best_omega=np.asarray(best_omega),
        per_case_power_MW=np.asarray(per_case_power_MW),
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed,
        weights=weights,
        config_meta=config_meta
    )

    # CSV table per case
    with open(out_dir / "per_case_power.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wind_dir_deg", "wind_speed", "weight", "mean_power_MW"])
        for d, s, wt, p in zip(wind_dir_deg, wind_speed, weights, per_case_power_MW):
            w.writerow([float(d), float(s), float(wt), float(p)])

    # JSON metadata
    meta = dict(
        best_power_MW=float(best_power_MW),
        final_loss=float(final_loss),
        final_sep_penalty=float(final_yaw_penalty),
        N_turbines=int(best_yaw.shape[0]),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config_meta,
    )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return None


def main():
    args = parse_args()
    DTYPE = setup_dtype(args.float64)

    # define yaw misalignment angle constraint
    yaw_constraints = YawConstraints(args.gamma_min, args.gamma_max)
    diameter = float(args.diameter)

    # Weather data
    data_dir = args.data_dir
    npz_path = data_dir / args.weather_npz
    if not npz_path.is_file():
        raise FileNotFoundError(f"Weather file not found: {npz_path}")

    wd = np.load(npz_path)
    if not all(k in wd for k in ("wind_direction", "wind_speed", "weight")):
        raise KeyError("weather_data.npz must contain 'wind_direction', 'wind_speed', 'weight'.")

    wind_dir_rad = jnp.deg2rad(jnp.asarray(wd["wind_direction"], dtype=DTYPE))
    wind_speed = jnp.asarray(wd["wind_speed"], dtype=DTYPE)
    weights = jnp.asarray(wd["weight"], dtype=DTYPE).reshape(-1)
    turbulence = jnp.full_like(wind_dir_rad, 0.06, dtype=DTYPE)

    # Build state/runner
    state, runner, N = build_state_runner(
        data_dir, args.farm_yaml, args.turbine_yaml, wind_dir_rad, wind_speed, turbulence, DTYPE
    )

    # TO-DO: Reference yaw layout for perturb mode (completely unyawed)
    ref_yaw = jnp.zeros((1, N), dtype=DTYPE)

    # Loss function
    loss_from_yaw, loss_from_omega, pw = make_losses(state, runner, weights, args.penalty_weight, DTYPE)

    gamma_min = yaw_constraints.mins(DTYPE)
    gamma_max = yaw_constraints.maxs(DTYPE)

    # Warmup compilation
    dummy_yaw = jnp.linspace(gamma_min, gamma_max, N, dtype=DTYPE).reshape(1, -1)
    dummy_omega = omega_from_gamma_sig(dummy_yaw, gamma_max)
    _ = loss_from_omega(dummy_omega, gamma_min, gamma_max).block_until_ready()

    # Optimiser: try L-BFGS with Strong-Wolfe zoom line search
    opt = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=5,
            verbose=False,
        )
    )

    val_and_grad = jax.value_and_grad(lambda omega: loss_from_omega(omega, gamma_min, gamma_max))

    @jax.jit
    def step(omega, opt_state):
        value, grad = val_and_grad(omega)
        updates, opt_state = opt.update(
            grad, opt_state, omega, value=value, grad=grad, value_fn=lambda _omega: loss_from_omega(_omega, gamma_min, gamma_max)
        )
        omega = optax.apply_updates(omega, updates)

        # Diagnostics
        gammas = gamma_from_omega_sig(omega, gamma_max)
        yaw_vio = yaw_penalty_sq(gammas, gamma_min, gamma_max)
        phys = value - pw * yaw_vio # loss function value
        g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
        gnorm = jnp.sqrt(g2)
        return omega, opt_state, value, gnorm, phys, yaw_vio

    # Restarts
    key = jax.random.PRNGKey(args.seed)


    if args.init_mode == "lhs":
        yaw0 = sample_initial_yaw_lhs(key, args.restarts, N, yaw_constraints, DTYPE)
    else:
        if args.restarts > 1:
            subkey = jax.random.split(key, 1)[0]
            yaw0 = sample_initial_yaw_lhs(subkey, args.restarts - 1, N, yaw_constraints, DTYPE)
        else:
            yaw0 = jnp.empty((0, N, 1), dtype=DTYPE)


    # Initialise restart loop
    best_power = -jnp.inf
    best_yaw = None
    best_idx = -1
    best_omega = None
    best_yaw_vio = None
    best_loss = None

    total_t0 = time.time()
    for m in range(args.restarts):
        print(f"\n=== Restart {m+1}/{args.restarts} ===")
        if m == 0:
            yaw_init = ref_yaw
        else:
            yaw_init = yaw0[m - 1][None, :]

        # why is this needed?
        omega = jax.lax.stop_gradient(omega_from_gamma_sig(yaw_init, gamma_max))
        opt_state = opt.init(omega)

        # Warmup to start @jax.jit on first call
        t0 = time.time()
        omega, opt_state, v, g, phys, yaw_vio = step(omega, opt_state)
        jax.block_until_ready(v)
        print(f"warmup compile+run: {time.time()-t0:.3f}s  "
              f"loss0={float(v):.6e}, phys0={float(phys):.6e}, yaw_vio0={float(yaw_vio):.6e}, |g|0={float(g):.3e}")

        v_prev = float(v)
        no_improve = 0
        last_it = 0

        t0 = time.time()

        # Post-compilation optimisation loop
        for it in range(1, args.max_iter + 1):
            t1 = time.time()
            omega, opt_state, v, g, phys, yaw_vio = step(omega, opt_state)
            jax.block_until_ready(v)
            last_it = it

            dv = v_prev - float(v)  # positive if improved
            if dv >= float(args.min_delta):
                no_improve = 0
                v_prev = float(v)
            else:
                no_improve += 1

            if it % 10 == 0:
                print(
                    f"iter {it:04d} loss={float(v):.6e}  "
                    f"phys={float(phys):.6e}"
                    f"pw*yaw_vio={float(yaw_vio):.6e}"
                    f"|g|={float(g):.3} delta_loss={dv:.3e}  "
                    f"no_improvements={no_improve}  time={time.time()-t1:.3f}s"
                )

            if no_improve >= args.patience:
                print(
                    f"Stopping early at iter {it}"
                    f"(no improvement >= {args.min_delta} for {args.patience} steps)"
                )
                break

            print(f"L-BFGS time (restart {m}): {time.time()-t0:.3f}s  iters={last_it}")

            # Final evaluation
            gamma_final = gamma_from_omega_sig(omega, gamma_max)
            final_loss = loss_from_yaw(gamma_final)
            jax.block_until_ready(final_loss)
            final_power = -float(final_loss)
            print(f"[restart {m}] final mean power (MW): {final_power:.6f}")

            if final_power > best_power:
                best_power = final_power
                best_yaw = gamma_final
                best_idx = m
                best_omega = omega
                best_yaw_vio = float(
                    yaw_penalty_sq(gamma_final, gamma_min, gamma_max)
                )
                best_loss = float(final_loss)

        print(f"\n === Finished {args.restarts} restarts in {time.time() - total_t0:.3f}s ===")
        print(f"Best restart: {best_idx}, best mean power (MW): {best_power:.6f}")
        print("Best yaw angles (degrees): ")
        print(np.rad2deg(np.asarray(best_yaw)))

        # Per-case mean power (MW) for best yaw angles
        out = runner(best_yaw)
        vel = average_velocity_jax(out.u_sorted)
        pow_mw = power(
            state.farm.power_thrust_table,
            vel,
            state.flow.air_density,
            yaw_angles=state.farm.yaw_angles,
        ) / (1e6 if DTYPE == jnp.float32 else jnp.asarray(1e6, dtype=DTYPE))
        per_case_power_MW = jnp.sum(pow_mw,  axis=1)

        # Output dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.out_dir / stamp
        wind_dir_deg = np.asarray(jnp.rad2deg(wind_dir_rad), dtype=float)
        wind_speed_np = np.asarray(wind_speed, dtype=float)
        weights_np = np.asarray(weights.reshape(-1), dtype=float)

        # Write optimisation metadata
        config_meta = dict(
            optimizer="optax.lbfgs+sw_zoom",
            restarts=int(args.restarts),
            maxiter=int(args.max_iter),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            seed=int(args.seed),
            float64=bool(args.float64),
            penalty_weight=float(args.penalty_weight),
            gamma_min=float(args.gamma_min),
            gamma_max=float(args.gamma_max),
            diameter=float(diameter),
            dtype="float64" if DTYPE == jnp.float64 else "float32",
            init_mode=args.init_mode,
        )

        save_run(
            out_dir,
            best_yaw=best_yaw,
            best_omega=best_omega,
            best_power_MW=best_power,
            final_loss=best_loss,
            final_yaw_penalty=(0.0 if best_yaw_vio is None else best_yaw_vio),
            per_case_power_MW=per_case_power_MW,
            wind_dir_deg=wind_dir_deg,
            wind_speed=wind_speed_np,
            weights=weights_np,
            config_meta=config_meta,
        )

if __name__ == "__main__":
    main()























"""
DiffWake/JAX: deterministic wind turbine yaw angle optimisation using the serial refine method
"""

from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# DiffWake imports
from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.yaw_runner import make_yaw_runner
from diffwake.diffwake_jax.util import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power


def setup_dtype(use_float64: bool = True) -> jnp.dtype:
    jax.config.update("jax_enable_x64", use_float64)
    return jnp.float64 if jax.config.x64_enabled else jnp.float32

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimise yaw angles in DiffWake/JAX with Serial-Refine"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/horn"),
                   help="Directory with weather data and YAML configs.")
    p.add_argument("--farm-yaml", type=str, default="cc_simple.yaml",
                   help="Farm configuration YAML (relative to --data-dir).")
    p.add_argument("--turbine-yaml", type=str, default="vestas_v802MW.yaml",
                   help="Turbine configuration YAML (relative to --data-dir).")
    p.add_argument("--weather-npz", type=str, default="weather_data.npz",
                   help="Weather data file (relative to --data-dir).")

    p.add_argument("--Nyaw", type=int, default=5, help="Number of first yaw angles to consider (serial run)")
    p.add_argument("--Nyaw-refine", type=int, default=4, help="Number of refined yaw angles to consider (refine run)")
    p.add_argument("--gamma-max", type=float, default=30.0, help="Maximum allowable yaw angle in degrees")
    p.add_argument("--gamma-min", type=float, default=-30.0, help="Minimum allowable yaw angle in degrees")

    p.add_argument("--float64", action="store_true", help="Enable float64. Default is float32.")
    p.add_argument("--out-dir", type=Path, default=Path("results/yaw_serial"), help="Base output directory.")
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
    B = int(wind_dir_rad.shape[0])

    return state, runner, N, B


def save_run(out_dir: Path,
             *,
             best_yaw: jax.Array,
             best_power_MW: float,
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
        N_turbines=int(best_yaw.shape[1]),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config_meta,
    )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return None


def main():
    args = parse_args()
    DTYPE = setup_dtype(args.float64)

    Nys, Nyr = args.Nyaw, args.Nyaw_refine
    if not isinstance(Nys, int) or not isinstance(Nyr, int):
        raise ValueError("Nyaw and Nyaw_refine must be integers.")
    if Nyr < 2:
        raise ValueError("Nyaw_refine must be at least 2.")
    if (Nys > 0) & ((Nyr + 1) % 2 == 0):
        raise ValueError(
            "Nyaw-refine must be an even number. "
            "This ensures the same yaw angles are not evaluated twice."
        )

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

    # Convert angle boundaries to radians for computation
    gamma_min = jnp.deg2rad(jnp.array(args.gamma_min, dtype=DTYPE))
    gamma_max = jnp.deg2rad(jnp.array(args.gamma_max, dtype=DTYPE))

    # Build state and simulation runner
    state, runner, N, B = build_state_runner(
        data_dir, args.farm_yaml, args.turbine_yaml, wind_dir_rad, wind_speed, turbulence, DTYPE
    )

    # Evaluates the power across all batches (wind cases)
    def power_from_yaw(yaw_angles: jnp.ndarray) -> jnp.ndarray:
        out = runner(yaw_angles)
        vel = average_velocity_jax(out.u_sorted)
        pow_mw = power(
            state.farm.power_thrust_table,
            vel,
            state.flow.air_density,
            yaw_angles=yaw_angles,
        )
        return jnp.sum(pow_mw, axis=1) / 1e6  # Returns shape (B,)

    zero_yaw = jnp.zeros((B, N), dtype=DTYPE)

    # -------------------------------------------------------------
    # ULTRA-FAST JAX SERIAL-REFINE ALGORITHM
    # -------------------------------------------------------------
    @jax.jit
    def sr_opt(initial_yaws: jax.Array) -> jax.Array:
        b_idx = jnp.arange(B)

        # Step size calculation for the refine pass
        step_size = jnp.abs(0.5 * (gamma_max - gamma_min) / (args.Nyaw - 1))

        def fused_turbine_pass(yaws, depth_idx):
            # 1. Dynamically get the turbine physical index for this depth across all B batches
            # sorted_coord_indices contains upstream -> downstream mapping per wind direction
            turb_indices_b = state.grid.sorted_coord_indices[:, depth_idx] # Shape (B,)

            # --- A) Coarse Pass ---
            coarse_angles = jnp.linspace(gamma_min, gamma_max, args.Nyaw, dtype=DTYPE)

            def make_coarse_cand(angle):
                return yaws.at[b_idx, turb_indices_b].set(angle)

            # vmap evaluates all coarse candidates simultaneously
            cands_coarse = jax.vmap(make_coarse_cand)(coarse_angles) # Shape (Ny, B, N)
            powers_coarse = jax.vmap(power_from_yaw)(cands_coarse)   # Shape (Ny, B)

            best_idx_coarse = jnp.argmax(powers_coarse, axis=0)      # Shape (B,)
            best_yaws_coarse = cands_coarse[best_idx_coarse, b_idx, :] # Shape (B, N)
            best_angles_coarse = coarse_angles[best_idx_coarse]      # Shape (B,)

            # --- B) Refine Pass ---
            offsets = jnp.linspace(-step_size, step_size, args.Nyaw_refine + 1, dtype=DTYPE)
            # mask = jnp.abs(offsets) > 1e-5
            # offsets = offsets[mask] # Filter exactly 0 offset. Shape (Nyr,)

            def make_refine_cand(offset):
                cand_angles = best_angles_coarse + offset
                cand_angles = jnp.clip(cand_angles, gamma_min, gamma_max)
                return best_yaws_coarse.at[b_idx, turb_indices_b].set(cand_angles)

            cands_refine = jax.vmap(make_refine_cand)(offsets) # Shape (Nyr, B, N)
            powers_refine = jax.vmap(power_from_yaw)(cands_refine) # Shape (Nyr, B)

            # Combine coarse winner with refine candidates to find the ultimate best
            cands_all = jnp.concatenate([best_yaws_coarse[None, ...], cands_refine], axis=0) # (1+Nyr, B, N)
            best_powers_coarse = powers_coarse[best_idx_coarse, b_idx]
            powers_all = jnp.concatenate([best_powers_coarse[None, :], powers_refine], axis=0) # (1+Nyr, B)

            best_idx_all = jnp.argmax(powers_all, axis=0) # Shape (B,)
            best_yaws_final = cands_all[best_idx_all, b_idx, :] # Shape (B, N)

            return best_yaws_final, None

        # Iterate down the farm. We don't optimize the very last turbine (depth N-1)
        # because its wake does not impact any downstream turbines.
        depth_indices = jnp.arange(N - 1)
        final_yaws, _ = jax.lax.scan(fused_turbine_pass, initial_yaws, depth_indices)

        return final_yaws

    # -------------------------------------------------------------
    # EXECUTION & LOGGING
    # -------------------------------------------------------------
    baseline_power = power_from_yaw(zero_yaw)
    total_baseline_MW = jnp.sum(baseline_power * weights)
    print(f"Baseline mean power (MW): {float(total_baseline_MW):.6f}")

    print("Compiling JAX graph and running warmup...")
    t0 = time.time()
    _ = sr_opt(zero_yaw).block_until_ready()
    print(f"Warmup took {time.time()-t0:.3f}s")

    print(f"Running Serial-Refine (Nyaw={args.Nyaw}, Nrefine={args.Nyaw_refine})...")
    t0 = time.time()
    opt_yaws = sr_opt(zero_yaw).block_until_ready()
    elapsed_time = time.time() - t0

    print(f"Opt_yaws: {jnp.rad2deg(opt_yaws)}")

    # Extract optimized powers
    opt_case_powers = power_from_yaw(opt_yaws)
    total_opt_MW = jnp.sum(opt_case_powers * weights)

    print(f"Execution took {elapsed_time:.3f}s")
    print(f"Optimised mean power (MW): {float(total_opt_MW):.6f}")
    uplift = ((total_opt_MW - total_baseline_MW) / total_baseline_MW) * 100
    print(f"Power Uplift: {float(uplift):.3f}%")

    # Output formatting
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp

    config_meta = dict(
        optimizer="serial-refine-jax",
        Nyaw=int(args.Nyaw),
        Nyaw_refine=int(args.Nyaw_refine),
        float64=bool(args.float64),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
        dtype="float64" if DTYPE == jnp.float64 else "float32",
        elapsed_time=float(elapsed_time),
    )

    save_run(
        out_dir,
        best_yaw=opt_yaws,
        best_power_MW=float(total_opt_MW),
        per_case_power_MW=opt_case_powers,
        wind_dir_deg=np.asarray(jnp.rad2deg(wind_dir_rad), dtype=float),
        wind_speed=np.asarray(wind_speed, dtype=float),
        weights=np.asarray(weights, dtype=float),
        config_meta=config_meta,
    )



if __name__ == "__main__":
    main()
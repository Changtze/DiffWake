"""
DiffWake/JAX: deterministic wind turbine yaw angle optimisation using the serial refine method
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Callable
from datetime import datetime
from pathlib import Path
import equinox

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
    p.add_argument("--gamma-min", type=float, default=-30.0, help="Maximum allowable yaw angle in degrees")

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

    return state, runner, N

def power_from_yaw(state, runner,
                  yaw_angles: jnp.ndarray,
                  dtype):
    out = runner(yaw_angles)
    vel = average_velocity_jax(out.u_sorted)
    pow_mw = power(
        state.farm.power_thrust_table,
        vel,
        state.flow.air_density,
        yaw_angles=yaw_angles,
    )
    case_power = jnp.sum(pow_mw, axis=1)
    return case_power / 1e6  # convert to megawatts


def save_run(out_dir: Path,
             *,
             best_yaw: jax.Array,
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
        final_loss=float(final_loss),
        final_sep_penalty=float(final_yaw_penalty),
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

    Nys, Nyr = args.Nyaw, args.Nyaw_refine  #
    if not isinstance(Nys, int) or not isinstance(Nyr, int):
        raise ValueError("Nyaw and Nyaw_refine must be integers.")
    if Nyr < 2:
        raise ValueError("Nyaw_refine must be at least 2.")
    if (Nys > 0) & ((Nyr + 1) % 2 == 0):
        raise ValueError(
            "Nyaw-refine must be an even number. "
            "This is to ensure the same yaw angles are not evaluated twice between passes."
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

    # Build state and simulation runner
    state, runner, N = build_state_runner(
        data_dir, args.farm_yaml, args.turbine_yaml, wind_dir_rad, wind_speed, turbulence, DTYPE
    )

    zero_yaw = jnp.zeros((1, N), 0.0, dtype=DTYPE)
    baseline_power = power_from_yaw(state, runner, zero_yaw, DTYPE)

    # Start Serial-Refine algorithm
    best_power = baseline_power  # Initialise optimal power variable

    # TO-DO: refine for each wind case

    # Get serial
    serial_yaw = jnp.linspace(args.gamma_min, args.gamma_max, args.Nyaw, dtype=DTYPE).reshape(1, -1)
    step_size = jnp.abs(0.5 * (args.gamma_max - args.gamma_min) / (args.Nyaw - 1))
    refine_yaw = jnp.linspace(-step_size, step_size, args.Nyaw_refine, dtype=DTYPE).reshape(1, -1)

    # Remove zeros from refine-pass to avoid duplicate angles
    mask = (refine_yaw != 0)
    refine_yaw_filtered = refine_yaw[mask]

    def sr_opt(state,
               runner,
               initial_yaws: jax.Array,
                farm_power_fn: Callable) -> jax.Array:
        def test_single_angle(yaw_angles: jax.Array,
                              turb_idx,
                              candidates):

        def combined_pass():
            """
            Do a coarse pass to roughly find the optimal yaw angle.
            Follow with a refine pass to improve the previous candidate.
            """

        return best_yaws





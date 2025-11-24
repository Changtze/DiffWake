#!/usr/bin/env python3
# Copyright (c) 2025 Maria Bånkestad
# SPDX-License-Identifier: Apache-2.0
"""
DiffWake/JAX: deterministic wind-farm layout optimization with L-BFGS+zoom.

Features
- CLI arguments instead of hard-coded paths
- Reproducible PRNG handling
- Configurable dtype (float32/float64)
- Three initialization modes: grid, lhs, perturb
- Safe file I/O + structured outputs (NPZ/CSV/JSON)
- Minimal, dependency-clean imports
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

# --- Project imports (assumed available in your package) ---
from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.layout_runner import make_layout_runner
from diffwake.diffwake_jax.util import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power


# ----------------------------- CLI & Config ----------------------------- #

@dataclass
class Box:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def mins(self, dtype) -> jax.Array:
        return jnp.array([self.x_min, self.y_min], dtype=dtype)

    def maxs(self, dtype) -> jax.Array:
        return jnp.array([self.x_max, self.y_max], dtype=dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize wind-farm layout in DiffWake/JAX with L-BFGS+zoom."
    )
    # Data & configs
    p.add_argument("--data-dir", type=Path, default=Path("data/horn"),
                   help="Directory with weather data and YAML configs.")
    p.add_argument("--farm-yaml", type=str, default="cc_hornsRev.yaml",
                   help="Farm configuration YAML (relative to --data-dir).")
    p.add_argument("--turbine-yaml", type=str, default="vestas_v802MW.yaml",
                   help="Turbine configuration YAML (relative to --data-dir).")
    p.add_argument("--weather-npz", type=str, default="weather_data.npz",
                   help="Weather data file (relative to --data-dir).")

    # Domain & constraints
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=3900.0)
    p.add_argument("--y-min", type=float, default=0.0)
    p.add_argument("--y-max", type=float, default=5520.0)
    p.add_argument("--diameter", type=float, default=136.0,
                   help="Turbine rotor diameter (m).")
    p.add_argument("--min-dist-mult", type=float, default=2.01,
                   help="Minimum center-to-center distance in units of diameter.")

    # Initialization
    p.add_argument("--init-mode", choices=["grid", "lhs", "perturb"], default="perturb",
                   help="How to initialize restarts.")
    p.add_argument("--nx", type=int, default=8, help="Grid cols for --init-mode grid.")
    p.add_argument("--ny", type=int, default=10, help="Grid rows for --init-mode grid.")
    p.add_argument("--perturb-sigma-x", type=float, default=68.0,
                   help="σ_x (m) for perturb init; default 0.5*D for D=136.")
    p.add_argument("--perturb-sigma-y", type=float, default=68.0,
                   help="σ_y (m) for perturb init.")

    # Optimization
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--penalty-weight", type=float, default=1e-3,
                   help="Weight for soft separation penalty.")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stop if Δloss < min-delta for this many steps.")
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--seed", type=int, default=0)

    # Numerics & output
    p.add_argument("--float64", action="store_true",
                   help="Enable float64. Default is float32.")
    p.add_argument("--out-dir", type=Path, default=Path("results/horn_lbfgs"),
                   help="Base output directory.")
    return p.parse_args()


def setup_dtype(use_float64: bool) -> jnp.dtype:
    # Must be set before heavy JAX computations
    jax.config.update("jax_enable_x64", use_float64)
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


# ----------------------------- Geometry utils ----------------------------- #

def z_from_points_tanh(points: jax.Array, mins: jax.Array, maxs: jax.Array, eps=1e-6) -> jax.Array:
    """Map points in [mins,maxs] to unconstrained z via atanh."""
    scaled = jnp.clip((points - mins) / (maxs - mins), eps, 1.0 - eps)
    arg = jnp.clip(2.0 * scaled - 1.0, -1.0 + 2.0 * eps, 1.0 - 2.0 * eps)
    return jnp.arctanh(arg)


def points_from_z_tanh(z: jax.Array, mins: jax.Array, maxs: jax.Array) -> jax.Array:
    """Inverse map: unconstrained z -> points in [mins,maxs]."""
    u = 0.5 * (jnp.tanh(z) + 1.0)  # (N,2) in (0,1)
    return mins + (maxs - mins) * u


def distance_penalty_sq(points: jax.Array,
                        min_distance: float,
                        dtype=jnp.float32) -> jax.Array:
    """Soft, differentiable penalty for violating pairwise min distances."""
    diffs = points[:, None, :] - points[None, :, :]         # (N,N,2)
    sq = jnp.sum(diffs * diffs, axis=-1)                    # (N,N)
    md2 = jnp.asarray(min_distance, dtype=dtype) ** 2
    N = points.shape[0]
    # lower triangle mask (i > j)
    mask = jnp.tril(jnp.ones((N, N), dtype=bool), k=-1)
    violation = jax.nn.softplus(md2 - sq)
    return jnp.sum(jnp.where(mask, violation, 0.0))


# ----------------------------- Initializations ----------------------------- #

def grid_layout(box: Box, nx: int, ny: int, dtype) -> jax.Array:
    x = np.linspace(box.x_min + 50.0, box.x_max - 50.0, nx)  # small margins
    y = np.linspace(box.y_min + 50.0, box.y_max - 50.0, ny)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    return jnp.asarray(pts, dtype=dtype)


def _latin_hypercube_unit(key, M: int, D: int, dtype) -> jax.Array:
    base = jnp.arange(M, dtype=jnp.int32)
    keys = jax.random.split(key, D + 1)
    k_jit, k_perms = keys[0], keys[1:]
    jitter = jax.random.uniform(k_jit, shape=(D, M), dtype=dtype)  # (D,M) in (0,1)

    def _perm_one(k):
        return jax.random.permutation(k, base)

    perms = jax.vmap(_perm_one)(k_perms)  # (D,M)
    U = (perms.astype(dtype) + jitter) / dtype.type(M)
    return U.T  # (M,D)


def sample_initial_layouts_lhs(key, M: int, N: int, box: Box, dtype) -> jax.Array:
    """(M,N,2) Latin hypercube samples in the box."""
    U = _latin_hypercube_unit(key, M, 2 * N, dtype)  # (M,2N)
    U = U.reshape(M, N, 2)
    mins = box.mins(dtype)
    maxs = box.maxs(dtype)
    x = mins[0] + (maxs[0] - mins[0]) * U[:, :, 0]
    y = mins[1] + (maxs[1] - mins[1]) * U[:, :, 1]
    return jnp.stack([x, y], axis=-1)


def sample_initial_layouts_perturb(
    key,
    M: int,
    ref_layout: jax.Array,
    box: Box,
    sigma_xy: Tuple[float, float],
    dtype,
) -> jax.Array:
    """
    Create M-1 noisy layouts around ref_layout using Gaussian noise in z-space.
    Returns shape (M-1, N, 2). The exact ref_layout can be used for restart 0.
    """
    N = ref_layout.shape[0]
    mins = box.mins(dtype)
    maxs = box.maxs(dtype)
    width = (maxs - mins)

    # crude mapping: meters -> z std near center: dz ≈ dx * 2/width
    sigma_xy = jnp.asarray(sigma_xy, dtype=dtype)
    sigma_z = 2.0 * sigma_xy / width

    z0 = z_from_points_tanh(ref_layout, mins, maxs)

    if M <= 1:
        return jnp.empty((0, N, 2), dtype=dtype)

    keys = jax.random.split(key, M - 1)

    def one(k):
        eps = jax.random.normal(k, shape=z0.shape, dtype=dtype) * sigma_z
        return points_from_z_tanh(z0 + eps, mins, maxs)

    return jax.vmap(one)(keys)  # (M-1, N, 2)


# ----------------------------- Runner / Loss ----------------------------- #

def build_state_runner(
    data_dir: Path,
    farm_yaml: str,
    turbine_yaml: str,
    wind_dir_rad: jax.Array,
    wind_speed: jax.Array,
    turb_intensity: jax.Array,
    dtype,
):
    cfg = (
        load_input(
            str(data_dir / farm_yaml),
            str(data_dir / turbine_yaml),
        )
        .set(
            wind_directions=wind_dir_rad,
            wind_speeds=wind_speed,
            turbulence_intensities=turb_intensity,
        )
    )
    state = create_state(cfg)
    runner = make_layout_runner(state)
    # N from config
    x0 = jnp.asarray(cfg.layout["layout_x"], dtype=dtype)
    N = int(x0.shape[0])
    return state, runner, N


def make_losses(state, runner, weights: jax.Array, penalty_weight: float, min_sep: float, dtype):
    pw = jnp.asarray(penalty_weight, dtype=dtype)
    min_sep = jnp.asarray(min_sep, dtype=dtype)

    def loss_from_layout(points_2d: jax.Array) -> jax.Array:
        """Physics objective: negative total farm power (weighted over wind cases)."""
        out = runner(points_2d)
        vel = average_velocity_jax(out.u_sorted)
        pow_mw = power(
            state.farm.power_thrust_table,
            vel,
            state.flow.air_density,
            yaw_angles=state.farm.yaw_angles,
        ) / jnp.asarray(1e6, dtype=dtype)  # [K, N]
        case_power = jnp.sum(pow_mw, axis=1)         # [K]
        return -jnp.sum(case_power * weights)        # scalar

    def loss_from_z(z: jax.Array, mins: jax.Array, maxs: jax.Array) -> jax.Array:
        points = points_from_z_tanh(z, mins, maxs)
        phys = loss_from_layout(points)
        sep = distance_penalty_sq(points, float(min_sep), dtype=dtype)
        return phys + pw * sep

    return loss_from_layout, loss_from_z, pw


# ----------------------------- Save helpers ----------------------------- #

def save_run(
    out_dir: Path,
    *,
    best_layout: jax.Array,          # (N,2)
    best_z: jax.Array,               # (N,2)
    best_power_MW: float,            # scalar
    final_loss: float,               # scalar
    final_sep_penalty: float,        # scalar
    per_case_power_MW: jax.Array,    # (K,)
    wind_dir_deg: np.ndarray,        # (K,)
    wind_speed: np.ndarray,          # (K,)
    weights: np.ndarray,             # (K,)
    config_meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Arrays
    np.savez(
        out_dir / "arrays.npz",
        best_layout=np.asarray(best_layout),
        best_z=np.asarray(best_z),
        per_case_power_MW=np.asarray(per_case_power_MW),
        wind_dir_deg=np.asarray(wind_dir_deg),
        wind_speed=np.asarray(wind_speed),
        weights=np.asarray(weights),
    )

    # 2) CSV table per case
    with open(out_dir / "per_case_power.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wind_dir_deg", "wind_speed", "weight", "mean_power_MW"])
        for d, s, wt, p in zip(wind_dir_deg, wind_speed, weights, per_case_power_MW):
            w.writerow([float(d), float(s), float(wt), float(p)])

    # 3) JSON metadata
    meta = dict(
        best_power_MW=float(best_power_MW),
        final_loss=float(final_loss),
        final_sep_penalty=float(final_sep_penalty),
        N_turbines=int(best_layout.shape[0]),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config_meta,
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[saved] {out_dir}")


# ----------------------------- Main ----------------------------- #

def main():
    args = parse_args()
    DTYPE = setup_dtype(args.float64)

    # Domain & constraints
    box = Box(args.x_min, args.x_max, args.y_min, args.y_max)
    diameter = float(args.diameter)
    min_distance_between_turbines = diameter * float(args.min_dist_mult)

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

    # Build state/runner (compiled once)
    state, runner, N = build_state_runner(
        data_dir, args.farm_yaml, args.turbine_yaml, wind_dir_rad, wind_speed, turbulence, DTYPE
    )

    # Initial reference layout for perturb mode (use farm cfg layout)
    ref_layout = jnp.column_stack(
        [
            jnp.asarray(state.farm.layout["layout_x"], dtype=DTYPE),
            jnp.asarray(state.farm.layout["layout_y"], dtype=DTYPE),
        ]
    )

    # If grid mode is chosen, override ref_layout for restart 0
    if args.init_mode == "grid":
        ref_layout = grid_layout(box, args.nx, args.ny, DTYPE)
        if int(ref_layout.shape[0]) != N:
            raise ValueError(
                f"Grid produced N={ref_layout.shape[0]} points, but farm expects N={N}."
            )

    # Losses
    loss_from_layout, loss_from_z, pw = make_losses(
        state, runner, weights, args.penalty_weight, min_distance_between_turbines, DTYPE
    )
    mins = box.mins(DTYPE)
    maxs = box.maxs(DTYPE)

    # Warmup compile (robust)
    dummy_points = jnp.stack(
        [
            jnp.linspace(args.x_min, args.x_max, N, dtype=DTYPE),
            jnp.linspace(args.y_min, args.y_max, N, dtype=DTYPE),
        ],
        axis=1,
    )
    dummy_z = z_from_points_tanh(dummy_points, mins, maxs, 1e-6)
    _ = loss_from_z(dummy_z, mins, maxs).block_until_ready()

    # Optimizer: L-BFGS with Strong-Wolfe zoom line search
    opt = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=5,
            verbose=False,
        )
    )

    val_and_grad = jax.value_and_grad(lambda z: loss_from_z(z, mins, maxs))

    @jax.jit
    def step(z, opt_state):
        value, grad = val_and_grad(z)
        updates, opt_state = opt.update(
            grad, opt_state, z, value=value, grad=grad, value_fn=lambda _z: loss_from_z(_z, mins, maxs)
        )
        z = optax.apply_updates(z, updates)

        # diagnostics
        points = points_from_z_tanh(z, mins, maxs)
        sep = distance_penalty_sq(points, min_distance_between_turbines, dtype=DTYPE)
        phys = value - pw * sep  # physics-only part
        g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
        gnorm = jnp.sqrt(g2)
        return z, opt_state, value, gnorm, phys, sep

    # Prepare restarts
    key = jax.random.PRNGKey(args.seed)

    if args.init_mode == "lhs":
        layouts0 = sample_initial_layouts_lhs(key, args.restarts, N, box, DTYPE)  # (M,N,2)
    elif args.init_mode == "perturb":
        layouts0 = sample_initial_layouts_perturb(
            key, args.restarts, ref_layout, box, (args.perturb_sigma_x, args.perturb_sigma_y), DTYPE
        )  # (M-1,N,2)
    else:
        # grid: we’ll use ref_layout for restart 0 and make (M-1) LHS for diversity
        if args.restarts > 1:
            subkey = jax.random.split(key, 1)[0]
            layouts0 = sample_initial_layouts_lhs(subkey, args.restarts - 1, N, box, DTYPE)
        else:
            layouts0 = jnp.empty((0, N, 2), dtype=DTYPE)

    # Restart loop
    best_power = -jnp.inf
    best_layout = None
    best_idx = -1
    best_z = None
    best_sep = None
    best_loss = None

    total_t0 = time.time()
    for m in range(args.restarts):
        print(f"\n=== Restart {m+1}/{args.restarts} ===")
        if m == 0:
            points_init = ref_layout
        else:
            points_init = layouts0[m - 1]

        z = jax.lax.stop_gradient(z_from_points_tanh(points_init, mins, maxs))
        opt_state = opt.init(z)

        # Warmup (compiles @jax.jit step on first call)
        t0 = time.time()
        z, opt_state, v, g, phys, sep = step(z, opt_state)
        jax.block_until_ready(v)
        print(
            f"warmup compile+run: {time.time()-t0:.3f}s  "
            f"loss0={float(v):.6e}, phys0={float(phys):.6e}, sep0={float(sep):.6e}, |g|0={float(g):.3e}"
        )

        v_prev = float(v)
        no_improve = 0
        last_it = 0

        t0 = time.time()
        for it in range(1, args.maxiter + 1):
            t1 = time.time()
            z, opt_state, v, g, phys, sep = step(z, opt_state)
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
                    f"iter {it:04d}  loss={float(v):.6e}  "
                    f"phys={float(phys):.6e}  sep={float(sep):.6e}  "
                    f"pw*sep={float((pw*sep)):.6e}  "
                    f"|g|={float(g):.3e}  Δloss={dv:.3e}  "
                    f"no_improve={no_improve}  time={time.time()-t1:.3f}s"
                )

            if no_improve >= args.patience:
                print(
                    f"stopping early at iter {it} "
                    f"(no improvement ≥ {args.min_delta} for {args.patience} steps)"
                )
                break

        print(f"L-BFGS time (restart {m}): {time.time()-t0:.3f}s  iters={last_it}")

        # Final eval (physics-only)
        layout = points_from_z_tanh(z, mins, maxs)
        final_loss = loss_from_layout(layout)
        jax.block_until_ready(final_loss)
        final_power = -float(final_loss)
        print(f"[restart {m}] final mean power (MW): {final_power:.6f}")

        if final_power > best_power:
            best_power = final_power
            best_layout = layout
            best_idx = m
            best_z = z
            best_sep = float(
                distance_penalty_sq(layout, min_distance_between_turbines, dtype=DTYPE)
            )
            best_loss = float(final_loss)

    print(f"\n=== Finished {args.restarts} restarts in {time.time()-total_t0:.3f}s ===")
    print(f"Best restart: {best_idx}  best mean power (MW): {best_power:.6f}")
    print("Best layout (x,y):")
    print(np.asarray(best_layout))

    # Per-case mean power (MW) for best layout
    out = runner(best_layout)
    vel = average_velocity_jax(out.u_sorted)
    pow_mw = power(
        state.farm.power_thrust_table,
        vel,
        state.flow.air_density,
        yaw_angles=state.farm.yaw_angles,
    ) / (1e6 if DTYPE == jnp.float32 else jnp.asarray(1e6, dtype=DTYPE))
    per_case_power_MW = jnp.sum(pow_mw, axis=1)

    # Output dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    wind_dir_deg = np.asarray(jnp.rad2deg(wind_dir_rad), dtype=float)
    wind_speed_np = np.asarray(wind_speed, dtype=float)
    weights_np = np.asarray(weights.reshape(-1), dtype=float)

    config_meta = dict(
        optimizer="optax.lbfgs+zoom",
        restarts=int(args.restarts),
        maxiter=int(args.maxiter),
        penalty_weight=float(args.penalty_weight),
        seed=int(args.seed),
        min_delta=float(args.min_delta),
        patience=int(args.patience),
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
        diameter=float(diameter),
        min_distance_between_turbines=float(min_distance_between_turbines),
        dtype="float64" if DTYPE == jnp.float64 else "float32",
        init_mode=args.init_mode,
        nx=int(args.nx),
        ny=int(args.ny),
    )

    save_run(
        out_dir,
        best_layout=best_layout,
        best_z=best_z,
        best_power_MW=best_power,
        final_loss=best_loss,
        final_sep_penalty=(0.0 if best_sep is None else best_sep),
        per_case_power_MW=per_case_power_MW,
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed_np,
        weights=weights_np,
        config_meta=config_meta,
    )


if __name__ == "__main__":
    main()

from setup_cashe import init_jax_persistent_cache
init_jax_persistent_cache()

import time
import jax
import jax.numpy as jnp
import optax

from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.layout_runner import make_layout_runner
from diffwake.diffwake_jax.util import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power
import numpy as np
import os, json, csv
from datetime import datetime

import numpy as np

x_min, x_max = 50.0, 3850.0
y_min, y_max = 50.0, 5470.0
nx, ny = 8, 10  # 8*10 = 80, best match to aspect ratio

x = np.linspace(x_min, x_max, nx)  # spacing_x ≈ (3850-50)/(8-1) ≈ 542.86 m
y = np.linspace(y_min, y_max, ny)  # spacing_y ≈ (5470-50)/(10-1) ≈ 602.22 m
XX, YY = np.meshgrid(x, y, indexing="xy")


jax.config.update("jax_enable_x64", False)
DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32
layout_x = jnp.array(XX.ravel().tolist(), dtype =  DTYPE)
layout_y = jnp.array(YY.ravel().tolist(), dtype =  DTYPE)
input_layout = jnp.column_stack([layout_x,layout_y])

wind_data = np.load("data/horn/weather_data.npz")
wind_directions_ = jnp.deg2rad(jnp.array(wind_data["wind_direction"], dtype = DTYPE))
weights = jnp.array(wind_data["weight"], dtype = DTYPE).reshape(-1)

wind_speeds_ = jnp.array(wind_data["wind_speed"], dtype = DTYPE)
turbulence_intensities_ = jnp.full_like(wind_directions_, 0.06, dtype=DTYPE)

x_min = DTYPE(0.0); x_max = DTYPE(3900.0)
y_min = DTYPE(0.0);     y_max = DTYPE(5520.0)
diameter = DTYPE(136.0)
min_distance_between_turbines = diameter * DTYPE(2.01)
mins = jnp.array([x_min, y_min], dtype=DTYPE)   # (2,)
maxs = jnp.array([x_max, y_max], dtype=DTYPE)   # (2,)
_EPS = DTYPE(1e-6)

# ---------------- Latin Hypercube sampler (spread-out in the box) ----------------
def _latin_hypercube_unit(key, M: int, D: int, dtype=DTYPE):
    base = jnp.arange(M, dtype=jnp.int32)  # 0..M-1
    k_all = jax.random.split(key, D + 1)
    k_jit, k_perms = k_all[0], k_all[1:]
    jitter = jax.random.uniform(k_jit, shape=(D, M), dtype=dtype)  # (D,M) in (0,1)
    def _perm_one(k): return jax.random.permutation(k, base)
    perms = jax.vmap(_perm_one)(k_perms)  # (D,M)
    U = (perms.astype(dtype) + jitter) / dtype(M)  # (D,M)
    return U.T  # (M,D)

def sample_initial_layouts_lhs(key, M: int, N: int):
    D = 2 * N
    U = _latin_hypercube_unit(key, M, D, dtype=DTYPE)  # (M,2N)
    U = U.reshape(M, N, 2)                             # (M,N,2)
    x = mins[0] + (maxs[0] - mins[0]) * U[:, :, 0]
    y = mins[1] + (maxs[1] - mins[1]) * U[:, :, 1]
    return jnp.stack([x, y], axis=-1)  # (M,N,2)

def sample_initial_layouts_perturb_z(
    key, M: int, input_layout: jax.Array, *,
    sigma_xy=(50.0, 50.0)  # std-dev in meters (x,y) around the grid
):
    """
    Make M initial layouts as small perturbations around input_layout by adding
    Gaussian noise in *z-space* (box-unconstrained). This keeps samples inside bounds.

    sigma_xy: tuple of (σx, σy) in meters (roughly).
    """
    N = input_layout.shape[0]
    width = (maxs - mins)  # (2,)
    # Map "meters" std to z-space std near center: dz ≈ dx * 2/width
    sigma_xy = jnp.asarray(sigma_xy, dtype=DTYPE)  # (2,)
    sigma_z  = DTYPE(2.0) * sigma_xy / width      # (2,)

    z0 = z_from_points_tanh(input_layout, mins, maxs, _EPS)  # (N,2)

    # build M-1 noisy copies (+1 exact copy we’ll insert in main loop)
    keys = jax.random.split(key, M - 1) if M > 1 else jnp.array([], dtype=jnp.uint32)
    def one(noisekey):
        eps = jax.random.normal(noisekey, shape=z0.shape, dtype=DTYPE) * sigma_z
        z = z0 + eps
        return points_from_z_tanh(z, mins, maxs)
    layouts = jax.vmap(one)(keys) if M > 1 else jnp.empty((0, N, 2), dtype=DTYPE)  # (M-1,N,2)
    return layouts

# ---------------- Param maps ----------------
def z_from_points_tanh(points, mins, maxs, eps=1e-7):
    eps = jnp.asarray(eps, dtype=DTYPE)
    scaled = jnp.clip((points - mins) / (maxs - mins), eps, DTYPE(1.0) - eps)
    arg = jnp.clip(DTYPE(2.0) * scaled - DTYPE(1.0),
                   -DTYPE(1.0) + DTYPE(2.0) * eps,
                    DTYPE(1.0) - DTYPE(2.0) * eps)
    return jnp.arctanh(arg)

def points_from_z_tanh(z, mins, maxs):
    u = DTYPE(0.5) * (jnp.tanh(z) + DTYPE(1.0))  # (N,2) in (0,1)
    return mins + (maxs - mins) * u

# ---------------- Penalty ----------------
@jax.jit
def distance_penalty_sq(points: jax.Array, min_distance: float | jax.Array = 1.0) -> jax.Array:
    diffs = points[:, None, :] - points[None, :, :]    # (N,N,2)
    sq = jnp.sum(diffs * diffs, axis=-1)               # (N,N)
    md2 = jnp.asarray(min_distance, dtype=DTYPE) ** 2
    N = points.shape[0]
    mask = jnp.tril(jnp.ones((N, N), dtype=bool), k=-1)
    zero = jnp.asarray(0.0, dtype=DTYPE)
    violation = jax.nn.softplus(md2 - sq)
    return jnp.sum(jnp.where(mask, violation, zero))

# ---------------- State / runner ----------------
def build_state_and_N():
    cfg = (load_input("data/horn/cc_hornsRev.yaml", "data/horn/vestas_v802MW.yaml")
           .set(wind_directions=wind_directions_,
                wind_speeds=wind_speeds_,
                turbulence_intensities=turbulence_intensities_))
    state = create_state(cfg)
    x0 = jnp.asarray(cfg.layout["layout_x"], dtype=DTYPE)
    N = x0.shape[0]
    return state, N

# ---------------- Loss builders ----------------
def make_losses(state, runner, penalty_weight: float):
    pw = jnp.asarray(penalty_weight, dtype=DTYPE)

    def loss_from_layout(layout_2d):
        out = runner(layout_2d)
        vel = average_velocity_jax(out.u_sorted)
        pow_mw = power(state.farm.power_thrust_table,
                       vel, state.flow.air_density,
                       yaw_angles=state.farm.yaw_angles) / DTYPE(1e6)
        p =  -jnp.sum(pow_mw,axis = 1)  # physics loss (negative mean power)
        p_ = p*weights
        return p_.sum()

    def loss_from_z(z):
        points = points_from_z_tanh(z, mins, maxs)
        phys = loss_from_layout(points)
        sep  = distance_penalty_sq(points, min_distance=min_distance_between_turbines)
        return phys + pw * sep

    return loss_from_layout, loss_from_z, pw




def save_run(out_dir,
             *,
             best_layout,        # (N,2)
             best_z,             # (N,2)
             best_power_MW,      # scalar
             final_loss,         # scalar (same as -best_power if your loss = -mean_power + penalty)
             final_sep_penalty,  # scalar
             per_case_power_MW,  # (K,) mean over turbines per wind case
             wind_dir_deg,       # (K,)
             wind_speed,         # (K,)
             weights,            # (K,)
             config_meta: dict):

    os.makedirs(out_dir, exist_ok=True)

    # 1) NPZ with arrays
    np.savez(os.path.join(out_dir, "arrays.npz"),
             best_layout=np.asarray(best_layout),
             best_z=np.asarray(best_z),
             per_case_power_MW=np.asarray(per_case_power_MW),
             wind_dir_deg=np.asarray(wind_dir_deg),
             wind_speed=np.asarray(wind_speed),
             weights=np.asarray(weights))

    # 2) CSV with per-case table
    csv_path = os.path.join(out_dir, "per_case_power.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wind_dir_deg", "wind_speed", "weight", "mean_power_MW"])
        for d, s, wt, p in zip(wind_dir_deg, wind_speed, weights, per_case_power_MW):
            w.writerow([float(d), float(s), float(wt), float(p)])

    # 3) JSON metadata (scalars + config)
    meta = dict(
        best_power_MW=float(best_power_MW),
        final_loss=float(final_loss),
        final_sep_penalty=float(final_sep_penalty),
        N_turbines=int(best_layout.shape[0]),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config_meta,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[saved] {out_dir}")
    
# ---- before main(), define once if you haven't already ----
val_and_grad = None  # will bind after make_losses()

def main(M_restarts=10,
         maxiter=80,
         penalty_weight=1e-3,
         seed=0,
         min_delta=1e-7,
         patience=8):

    # Build state and runner once (compile once)
    state, N = build_state_and_N()
    runner = make_layout_runner(state)

    # Prebuild losses
    loss_from_layout, loss_from_z, pw = make_losses(state, runner, penalty_weight)

    val_and_grad = jax.value_and_grad(loss_from_z)

    # ---- warm up ONLY the scalar loss (robust across runner implementations)
    dummy_points = jnp.stack([
        jnp.linspace(x_min, x_max, N, dtype=DTYPE),
        jnp.linspace(y_min, y_max, N, dtype=DTYPE)
    ], axis=1)
    dummy_z = z_from_points_tanh(dummy_points, mins, maxs, _EPS)
    _ = loss_from_z(dummy_z).block_until_ready()

    # L-BFGS optimizer with Strong-Wolfe zoom line search
    opt = optax.chain(
        optax.lbfgs(
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=5,
                verbose=False,
            )
        )
    )

    @jax.jit
    def step(z, opt_state):
        # total loss & grad (compatible with zoom line search)
        value, grad = val_and_grad(z)
        updates, opt_state = opt.update(
            grad, opt_state, z,
            value=value, grad=grad, value_fn=loss_from_z
        )
        z = optax.apply_updates(z, updates)

        # diagnostics
        points = points_from_z_tanh(z, mins, maxs)
        sep = distance_penalty_sq(points, min_distance=min_distance_between_turbines)
        phys = value - pw * sep  # physics-only part
        g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
        gnorm = jnp.sqrt(g2)
        return z, opt_state, value, gnorm, phys, sep

    # Sample M initial layouts (no recompilation—shapes fixed)
    key = jax.random.PRNGKey(seed)
    #layouts0 = sample_initial_layouts_lhs(key, M_restarts, N)  # (M, N, 2)

    layouts0 = sample_initial_layouts_perturb_z(
            key, M_restarts, input_layout,
            sigma_xy=(0.5 * diameter, 0.5 * diameter)  # tweak noise scale as you like
        )

    best_power = -jnp.inf
    best_layout = None
    best_idx = -1
    best_z = None           # NEW
    best_sep = None         # NEW
    best_loss = None        # NEW

    total_t0 = time.time()
    for m in range(M_restarts):
        print(f"\n=== Restart {m+1}/{M_restarts} ===")
                                    # (N,2)
        if m == 0:
            points_init = input_layout
        else:
            points_init = layouts0[m-1]
        z = jax.lax.stop_gradient(z_from_points_tanh(points_init, mins, maxs, _EPS))
        opt_state = opt.init(z)

        # Warmup once (compilation of step occurs on first call)
        t0 = time.time()
        z, opt_state, v, g, phys, sep = step(z, opt_state); jax.block_until_ready(v)
        print(f"warmup compile+run: {time.time()-t0:.3f}s  "
              f"loss0={float(v):.6e}, phys0={float(phys):.6e}, sep0={float(sep):.6e}, |g|0={float(g):.3e}")

        # Loss-based early stopping
        v_prev = float(v)
        no_improve = 0
        last_it = 0

        t0 = time.time()
        for it in range(1, maxiter + 1):
            t1 = time.time()
            z, opt_state, v, g, phys, sep = step(z, opt_state); jax.block_until_ready(v)
            last_it = it

            dv = v_prev - float(v)  # positive if improved
            if dv >= float(min_delta):
                no_improve = 0
                v_prev = float(v)
            else:
                no_improve += 1

            if it % 10 == 0:
                print(f"iter {it:04d}  loss={float(v):.6e}  "
                      f"phys={float(phys):.6e}  sep={float(sep):.6e}  "
                      f"pw*sep={float((pw*sep)):.6e}  "
                      f"|g|={float(g):.3e}  Δloss={dv:.3e}  "
                      f"no_improve={no_improve}  time={time.time()-t1:.3f}s")

            if no_improve >= patience:
                print(f"stopping early at iter {it} (no improvement ≥ {min_delta} for {patience} steps)")
                break

        print(f"L-BFGS time (restart {m}): {time.time()-t0:.3f}s  iters={last_it}")

        # Final eval for this restart (physics-only power)
        layout = points_from_z_tanh(z, mins, maxs)
        final_loss = loss_from_layout(layout); jax.block_until_ready(final_loss)
        final_power = -float(final_loss)  # MW
        print(f"[restart {m}] final mean power (MW): {final_power:.6f}")

        if final_power > best_power:
            best_power = final_power
            best_layout = layout
            best_idx = m
            best_z = z                                              # NEW
            best_sep = float(distance_penalty_sq(layout,            # NEW
                                 min_distance=min_distance_between_turbines))
            best_loss = float(final_loss)                            # NEW

    print(f"\n=== Finished {M_restarts} restarts in {time.time()-total_t0:.3f}s ===")
    print(f"Best restart: {best_idx}  best mean power (MW): {best_power:.6f}")
    print("Best layout (x,y):", best_layout)

    # ---------- SAVE RESULTS ----------
    # Per-case mean power (MW) for the best layout
    out = runner(best_layout)
    vel = average_velocity_jax(out.u_sorted)
    pow_mw = power(state.farm.power_thrust_table,
                   vel, state.flow.air_density,
                   yaw_angles=state.farm.yaw_angles) / DTYPE(1e6)      # [K, N]
    per_case_power_MW = jnp.sum(pow_mw, axis=1)                        # [K]

    run_name = "horn_lbfgs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", run_name, stamp)

    config_meta = dict(
        optimizer="optax.lbfgs+zoom",
        M_restarts=int(M_restarts),
        maxiter=int(maxiter),
        penalty_weight=float(penalty_weight),
        seed=int(seed),
        min_delta=float(min_delta),
        patience=int(patience),
        x_min=float(x_min), x_max=float(x_max),
        y_min=float(y_min), y_max=float(y_max),
        diameter=float(diameter),
        min_distance_between_turbines=float(min_distance_between_turbines),
        dtype="float64" if jax.config.x64_enabled else "float32",
    )

    wind_dir_deg = jnp.rad2deg(wind_directions_).astype(float)
    wind_speed = jnp.asarray(wind_speeds_, dtype=float)
    w_save = jnp.asarray(weights.reshape(-1), dtype=float)

    save_run(out_dir,
             best_layout=best_layout,
             best_z=best_z,
             best_power_MW=best_power,
             final_loss=best_loss,
             final_sep_penalty=(0.0 if best_sep is None else best_sep),
             per_case_power_MW=per_case_power_MW,
             wind_dir_deg=wind_dir_deg,
             wind_speed=wind_speed,
             weights=w_save,
             config_meta=config_meta)

if __name__ == "__main__": 
    main(M_restarts=1, maxiter=200, penalty_weight=1e-3, seed=0, min_delta=1e-7, patience=10)
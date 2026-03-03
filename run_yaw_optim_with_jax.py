from setup_cashe import init_jax_persistent_cache
init_jax_persistent_cache()  # optional; keep if you want persistent cache

import time
import jax, jax.numpy as jnp
from jax.nn import sigmoid
import optax

from ccflow_jax.model import load_input, create_state
from ccflow_jax.yaw_runner import make_yaw_runner


from ccflow_jax.util import average_velocity_jax
from ccflow_jax.turbine.operation_models import power

def runtime_float():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32

jax.config.update("jax_enable_x64", False)  # f32 on GPU

DTYPE = runtime_float()
MAX_ANGLE = jnp.deg2rad(jnp.array(25.0, dtype=DTYPE))

# Problem setup (constant shapes)
wind_directions_ = jnp.deg2rad(jnp.arange(0.0, 360.0, 10.0))   # (36,)
wind_speeds_ = jnp.ones_like(wind_directions_) * 8.0
turbulence_intensities_ = jnp.ones_like(wind_directions_) * 0.06

def build_state():
    cfg = (load_input("data/cc_large.yml", "data/vestas.yaml")
           .set(wind_directions=wind_directions_,
                wind_speeds=wind_speeds_,
                turbulence_intensities=turbulence_intensities_))
    return create_state(cfg)



# ---------- optimization script ----------
def main(steps=20, lr=0.5, seed=0):
    key   = jax.random.PRNGKey(seed)
    state = build_state()

    runner = make_yaw_runner(state)  # compiled closure
    idx = state.grid.unsorted_indices[:, :, 0, 0]  # (B,T)

    # Pure-JAX loss; called inside train_step
    def loss_from_yaw(yaw_layout):
        out = runner(yaw_layout)
        vel = average_velocity_jax(out.u_sorted)  # traced inside step
        pow_mw = power(state.farm.power_thrust_table,
                       vel, state.flow.air_density,
                       yaw_angles=yaw_layout) / 1e6
        return -jnp.mean(pow_mw)

    def loss_from_z(z):
        yaw_layout = sigmoid(z) * MAX_ANGLE
        return loss_from_yaw(yaw_layout)

    opt = optax.adamw(lr, weight_decay=0.0, b1=0.2, b2=0.7)

    @jax.jit
    def train_step(z, opt_state):
        loss, grad = jax.value_and_grad(loss_from_z)(z)
        updates, opt_state = opt.update(grad, opt_state, params=z)
        z = optax.apply_updates(z, updates)
        return z, opt_state, loss

    # Warmup (compilation)
    z = 0.5 * jax.random.normal(key, state.farm.yaw_angles.shape, dtype=DTYPE)
    opt_state = opt.init(z)
    t0 = time.time()
    z, opt_state, loss0 = train_step(z, opt_state)
    jax.block_until_ready(loss0)
    print(f"warmup compile+run: {time.time()-t0:.3f}s")

    # Training loop (no recompiles if shape/dtype fixed)
    t0 = time.time()
    for t in range(steps):
        t1 = time.time()
        z, opt_state, loss = train_step(z, opt_state)
        jax.block_until_ready(loss)  # only for timing/logging
        if t % 5 == 0:
            print(f"step {t:03d}  loss={float(loss):.6e}  step_time={time.time()-t1:.3f}s")
    print(f"Training time: {time.time()-t0:.3f}s")

    # Final eval
    yaw_layout = sigmoid(z) * MAX_ANGLE
    final = loss_from_yaw(yaw_layout)
    jax.block_until_ready(final)
    final_inflow = -float(final)

    yaw = jnp.take_along_axis(yaw_layout, idx, axis=1)
    print("final mean power:", final_inflow)
    print("optimal yaw (deg):", jnp.rad2deg(yaw))

if __name__ == "__main__":
    main()

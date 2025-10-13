from __future__ import annotations
import math
import jax.numpy as jnp
from jax import lax, jit, device_put
from jax.tree_util import tree_map

from .util import (
    CCState,
    init_dynamic_state, get_axial_induction_fn, get_thrust_fn,
    make_params, to_result, _to_jax
)
from .solver import cc_solver_step


def _rotate_and_sort_from_1d(
    x_t: jnp.ndarray,           # (T,) turbine x (shared across all wind directions)
    y_t: jnp.ndarray,           # (T,) turbine y (shared across all wind directions)
    yaw_angles: jnp.ndarray,    # (B, T)
    tilt_angles: jnp.ndarray,   # (B, T)
    theta: jnp.ndarray,         # (B, 1) per-direction rotation (radians)
    dy_exp: jnp.ndarray,        # (B, T, R, R) lateral offsets
    template: jnp.ndarray,      # (B, T, R, R) ones
):
    """
    Broadcast (T,) layout to (B,T), rotate by theta per wind direction,
    build (R x R) sampling grids, sort by streamwise x, and return sorted yaw/tilt + grids.
    """
    xc = 0.5 * (x_t.min() + x_t.max())
    yc = 0.5 * (y_t.min() + y_t.max())

    B, T = yaw_angles.shape
    x = jnp.broadcast_to(x_t[None, :], (B, T))
    y = jnp.broadcast_to(y_t[None, :], (B, T))

    x_rel = x - xc
    y_rel = y - yc
    c, s = jnp.cos(theta), jnp.sin(theta)        # (B,1)
    x_rot = x_rel * c + y_rel * s + xc           # (B,T)
    y_rot = -x_rel * s + y_rel * c + yc          # (B,T)

    x_grid = x_rot[:, :, None, None] * template
    y_grid = y_rot[:, :, None, None] * template + dy_exp

    sorted_idx = jnp.argsort(x_grid, axis=1)     # (B,T,R,R)
    idx_turb   = sorted_idx[:, :, 0, 0]          # (B,T)

    yaw_sorted   = jnp.take_along_axis(yaw_angles,  idx_turb, axis=1)
    tilt_sorted  = jnp.take_along_axis(tilt_angles, idx_turb, axis=1)
    x_sorted     = jnp.take_along_axis(x_grid, sorted_idx, axis=1)
    y_sorted     = jnp.take_along_axis(y_grid, sorted_idx, axis=1)
    x_c_sorted   = jnp.mean(x_sorted, axis=(2, 3), keepdims=True)   # (B,T,1,1)
    y_c_sorted   = jnp.mean(y_sorted, axis=(2, 3), keepdims=True)   # (B,T,1,1)

    return yaw_sorted, tilt_sorted, x_sorted, y_sorted, x_c_sorted, y_c_sorted

def _make_constants_local(state: CCState):
    g   = _to_jax(state.grid)
    fld = _to_jax(state.flow)
    farm = state.farm

    z_coord = g.z_sorted
    z_c = jnp.mean(z_coord, axis=(2,3), keepdims=True)
    u_init  =  jnp.asarray(fld.u)
    dudz_init = jnp.asarray(fld.dudz)
    ambient_ti = jnp.asarray(fld.turbulence_intensities[:, None, None, None])
    return z_coord, z_c,  u_init, dudz_init, ambient_ti, _to_jax(farm.yaw_angles), _to_jax(farm.tilt_angles)

def make_layout_runner(state: CCState, grid_resolution: int = 3):
    """
    Build a compiled function that runs one full simulation given a shared farm layout:
        runner(layout_2d) -> CCResult
    where x_t, y_t are shape (T,), shared for all wind directions in the batch.

    Notes:
      * All constants are put on device and stop_gradient’ed once.
      * Rotation/sorting happens once per call (outside the time loop).
      * The time loop uses lax.fori_loop and only carries solver state.
    """
    # Model pieces
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)
    (params,
     enable_secondary_steering,
     enable_transverse_velocities,
     enable_yaw_added_recovery) = make_params(state)

    T_steps = int(params.T)
    B       = int(params.B)
    T_turbs = state.farm.yaw_angles.shape[1]  # number of turbines

    # Pull needed constants
    (z_coord, z_c,
     u_init, dudz_init, ambient_ti,
     
     yaw_angles, tilt_angles) = _make_constants_local(state)

    init = init_dynamic_state(state.grid, state.flow)

    theta = (state.flow.wind_directions - 3. * math.pi / 2.)[:, None]  # (B,1)

    # Build lateral offsets (dy_exp) once
    R = int(grid_resolution)
    span   = jnp.linspace(-1.0, 1.0, R)                         # (R,)
    radius = state.farm.turbine.rotor_diameter * 0.5 * 0.5
    dy     = span * radius                                      # (R,)
    dy_exp = jnp.broadcast_to(dy[None, None, :, None], (B, T_turbs, R, R))
    template = jnp.ones((B, T_turbs, R, R), dtype=u_init.dtype)

    # Freeze constants on device
    theta       = lax.stop_gradient(device_put(theta))
    dy_exp      = lax.stop_gradient(device_put(dy_exp))
    template    = lax.stop_gradient(device_put(template))
    yaw_angles  = lax.stop_gradient(device_put(yaw_angles))
    tilt_angles = lax.stop_gradient(device_put(tilt_angles))
    ambient_ti  = lax.stop_gradient(device_put(ambient_ti))
    u_init      = lax.stop_gradient(device_put(u_init))
    dudz_init   = lax.stop_gradient(device_put(dudz_init))
    z_coord     = lax.stop_gradient(device_put(z_coord))
    z_c         = lax.stop_gradient(device_put(z_c))
    init        = tree_map(lambda x: lax.stop_gradient(device_put(x)), init)
    params      = tree_map(lambda x: lax.stop_gradient(device_put(x)), params)

    # Plain Python bools
    enable_secondary_steering    = bool(enable_secondary_steering)
    enable_transverse_velocities = bool(enable_transverse_velocities)
    enable_yaw_added_recovery    = bool(enable_yaw_added_recovery)

    # Close over wake models
    velocity_model   = state.wake.velocity_model
    deflection_model = state.wake.deflection_model
    turbulence_model = state.wake.turbulence_model

    def _run_with_layout(layout_2d: jnp.ndarray):

        x_t = layout_2d[:,0]
        y_t = layout_2d[:,1]

        # Rotate, grid, sort once per call
        (yaw_sorted, tilt_sorted,
         x_sort, y_sort,
         x_c_sort, y_c_sort) = _rotate_and_sort_from_1d(
             x_t, y_t, yaw_angles, tilt_angles, theta, dy_exp, template
         )

        # Time loop — carry only solver state
        def body(ii, st):
            ii32 = lax.convert_element_type(ii, jnp.int32)
            st_next, _ = cc_solver_step(
                st, ii32, params,
                thrust_fn, axial_fn,
                velocity_model, deflection_model, turbulence_model,
                yaw_sorted, tilt_sorted,
                enable_secondary_steering,
                enable_transverse_velocities,
                enable_yaw_added_recovery,
                x_sort, y_sort, z_coord,
                x_c_sort, y_c_sort, z_c,
                u_init, dudz_init, ambient_ti,
            )
            return st_next

        final_state = lax.fori_loop(0, T_steps, body, init)
        return to_result(final_state)

    return jit(_run_with_layout)

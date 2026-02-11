from jax import lax, tree, device_put, jit
import jax.numpy as jnp
import jax
from .util_agnostic import (
    State,
    get_axial_induction_fn,
    get_thrust_fn, make_params,to_result,_to_jax, DynamicState
)
from .solver_agnostic import cc_solver_step
from dataclasses import replace as dc_replace


def runtime_dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def make_constants(state: State):
    g   = _to_jax(state.grid)
    fld = _to_jax(state.flow)
    farm = state.farm

    x, y, z = g.x_sorted, g.y_sorted, g.z_sorted
    x_c = jnp.mean(x, axis=(2,3), keepdims=True)
    y_c = jnp.mean(y, axis=(2,3), keepdims=True)
    z_c = jnp.mean(z, axis=(2,3), keepdims=True)

    const = dict(
        x_coord = x,  y_coord = y,  z_coord = z,
        x_c = x_c,    y_c = y_c,    z_c = z_c,
        u_init  = fld.u_initial_sorted.copy(),
        dudz_init = fld.dudz_initial_sorted.copy(),
    )

    return const, _to_jax(farm.yaw_angles_sorted), _to_jax(farm.tilt_angles_sorted)

#

def init_dynamic_state(grid, flow, batch_size = None) -> State:
    B, T, Ny, Nz = grid.x_sorted.shape
    if batch_size is None:
        batch_size = B
    zeros = jnp.zeros_like(_to_jax(grid.x_sorted[:batch_size]))
    ti = jnp.broadcast_to(flow.turbulence_intensities[:, None, None, None], (B, T, 3, 3))[:batch_size]
    return DynamicState(
        turb_u_wake = zeros.copy(),
        turb_inflow = _to_jax(flow.u_initial_sorted[:batch_size]).copy(),
        ti          = ti.copy(),
        v_sorted      = zeros.copy(),
        w_sorted      = zeros.copy(),
        Ctmp        = jnp.zeros((T, batch_size, T, Ny, Nz), zeros.dtype),
        ct_acc=jnp.zeros((batch_size, T, 1, 1), zeros.dtype),  # <— running CTs
    )



def make_par_runner(state: State):
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)

    (params,
     enable_secondary_steering,
     enable_transverse_velocities,
     enable_yaw_added_recovery) = make_params(state)

    const, yaw_angles, tilt_angles = make_constants(state)
    init = init_dynamic_state(state.grid, state.flow)

    # Put once on device & stop grads through constants
    const       = tree.map(lambda x: lax.stop_gradient(device_put(x)), const)

    tilt_angles = lax.stop_gradient(device_put(tilt_angles))
    yaw_angles = lax.stop_gradient(device_put(yaw_angles))

    init_template        = tree.map(lambda x: lax.stop_gradient(device_put(x)), init)
    params      = tree.map(lambda x: lax.stop_gradient(device_put(x)), params)

    # Make flags plain bools
    enable_secondary_steering    = bool(enable_secondary_steering)
    enable_transverse_velocities = bool(enable_transverse_velocities)
    enable_yaw_added_recovery    = bool(enable_yaw_added_recovery)

    velocity_model   = state.wake.velocity_model
    deflection_model = state.wake.deflection_model
    turbulence_model = state.wake.turbulence_model

    T = int(params.T)
    B = int(params.B)
    def _loop(ti_vec: jnp.ndarray):
        """
        Inputs are (B,) jnp arrays. We broadcast to:
        ambient_ti : (B,T,1,1)
        a_s, b_s   : (B,1,1,1)
        """
        DTYPE = runtime_dtype()

        # Ensure dtype and exact (B,) shapes
        ti_vec  = jnp.asarray(ti_vec,  DTYPE).reshape((B,))

        # Build ambient TI on turbine/time axes
        ambient_ti = jnp.broadcast_to(ti_vec[:, None, None, None], (B, T, 1, 1))   # (B,T,1,1)
        ti0_grid   = jnp.broadcast_to(ambient_ti, (B, T, 3, 3))                     # (B,T,3,3)

        init = dc_replace(init_template, ti=ti0_grid)

        def body(ii, st):
            ii32 = lax.convert_element_type(ii, jnp.int32)
            st_next, _ = cc_solver_step(
                st, ii32, params,
                thrust_fn, axial_fn,
                velocity_model, deflection_model, turbulence_model,
                yaw_angles, tilt_angles,
                **const,
                enable_secondary_steering=enable_secondary_steering,
                enable_transverse_velocities=enable_transverse_velocities,
                enable_yaw_added_recovery=enable_yaw_added_recovery,
                ambient_ti = ambient_ti
            )
            return st_next

        final_state = lax.fori_loop(0, T, body, init)
        return to_result(final_state)

    # Single compiled function taking only yaw
    return jit(_loop)



def make_sub_par_runner(state: State, batch_size = 512):
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)

    (params,
     enable_secondary_steering,
     enable_transverse_velocities,
     enable_yaw_added_recovery) = make_params(state)

    const, yaw_angles, tilt_angles = make_constants(state)

    init = init_dynamic_state(state.grid, state.flow, batch_size=batch_size)
    x_coord = const["x_coord"]; y_coord = const["y_coord"]; z_coord = const["z_coord"]
    x_c     = const["x_c"];     y_c     = const["y_c"];     z_c     = const["z_c"]
    u_init  = const["u_init"];  dudz_init = const["dudz_init"]

    # Put once on device & stop grads through constants
    const       = tree.map(lambda x: lax.stop_gradient(device_put(x)), const)
    tilt_angles = lax.stop_gradient(device_put(tilt_angles))
    yaw_angles = lax.stop_gradient(device_put(yaw_angles))

    init_template        = tree.map(lambda x: lax.stop_gradient(device_put(x)), init)
    params      = tree.map(lambda x: lax.stop_gradient(device_put(x)), params)

    # Make flags plain bools
    enable_secondary_steering    = bool(enable_secondary_steering)
    enable_transverse_velocities = bool(enable_transverse_velocities)
    enable_yaw_added_recovery    = bool(enable_yaw_added_recovery)

    velocity_model   = state.wake.velocity_model
    deflection_model = state.wake.deflection_model
    turbulence_model = state.wake.turbulence_model

    T = int(params.T)
    B = int(params.B)
    DTYPE = runtime_dtype()
    
    @jit
    def run_subset(ti_vec: jnp.ndarray, idx: jnp.ndarray):
        """
        ti_vec: [batch_size]  (ambient TI for this subset)
        idx   : [batch_size]  (int32 indices into the full batch dimension)
        """
        ti_vec  = jnp.asarray(ti_vec, DTYPE).reshape((batch_size,))
        idx     = jnp.asarray(idx, jnp.int32).reshape((batch_size,))

        # Build per-call initial state with the requested TI
        ambient_ti = jnp.broadcast_to(ti_vec[:, None, None, None], (batch_size, T, 1, 1))
        ti_grid    = jnp.broadcast_to(ambient_ti, (batch_size, T, 3, 3))
        st0        = dc_replace(init_template, ti=ti_grid)

        # Slice geometry & angles by idx
        yaw_b   = yaw_angles[idx]
        tilt_b  = tilt_angles[idx]
        x_b     = x_coord[idx]; y_b = y_coord[idx]; z_b = z_coord[idx]
        xcb     = x_c[idx];     ycb = y_c[idx];     zcb = z_c[idx]
        u0_b    = u_init[idx];  dudz0_b = dudz_init[idx]

        def body(ii, st):
            ii32 = lax.convert_element_type(ii, jnp.int32)
            st_next, _ = cc_solver_step(
                st, ii32, params,
                thrust_fn, axial_fn,
                velocity_model,
                deflection_model,
                turbulence_model,
                yaw_b, tilt_b,
                x_coord=x_b, y_coord=y_b, z_coord=z_b,
                x_c=xcb, y_c=ycb, z_c=zcb,
                u_init=u0_b, dudz_init=dudz0_b,
                enable_secondary_steering=enable_secondary_steering,
                enable_transverse_velocities=enable_transverse_velocities,
                enable_yaw_added_recovery=enable_yaw_added_recovery,
                ambient_ti=ambient_ti,
            )
            return st_next

        final_state = lax.fori_loop(0, T, body, st0)
        return to_result(final_state)  # has .u_sorted, etc.

    return run_subset
from jax         import lax, jit, tree, device_put, jit
import jax.numpy as jnp
from .util import (
    CCState,
    init_dynamic_state, get_axial_induction_fn,
    get_thrust_fn, make_params,
    make_constants,to_result
)
from .solver import cc_solver_step   


def make_yaw_runner(state: CCState):
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)

    (params,
     enable_secondary_steering,
     enable_transverse_velocities,
     enable_yaw_added_recovery) = make_params(state)

    const, _, tilt_angles = make_constants(state)
    init = init_dynamic_state(state.grid, state.flow)

    # Put once on device & stop grads through constants
    const       = tree.map(lambda x: lax.stop_gradient(device_put(x)), const)
    tilt_angles = lax.stop_gradient(device_put(tilt_angles))
    init        = tree.map(lambda x: lax.stop_gradient(device_put(x)), init)
    params      = tree.map(lambda x: lax.stop_gradient(device_put(x)), params)

    # Make flags plain bools
    enable_secondary_steering    = bool(enable_secondary_steering)
    enable_transverse_velocities = bool(enable_transverse_velocities)
    enable_yaw_added_recovery    = bool(enable_yaw_added_recovery)

    velocity_model   = state.wake.velocity_model
    deflection_model = state.wake.deflection_model
    turbulence_model = state.wake.turbulence_model

    T = int(params.T)

    def _loop_with_yaw(yaw_angles_sorted: jnp.ndarray):
        # fori_loop since we only need the final state
        def body(ii, st):
            ii32 = lax.convert_element_type(ii, jnp.int32)
            st_next, _ = cc_solver_step(
                st, ii32, params,
                thrust_fn, axial_fn,
                velocity_model, deflection_model, turbulence_model,
                yaw_angles_sorted, tilt_angles,
                **const,
                enable_secondary_steering=enable_secondary_steering,
                enable_transverse_velocities=enable_transverse_velocities,
                enable_yaw_added_recovery=enable_yaw_added_recovery,
            )
            return st_next

        final_state = lax.fori_loop(0, T, body, init)
        return to_result(final_state)

    # Single compiled function taking only yaw
    return jit(_loop_with_yaw)
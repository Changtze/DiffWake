from jax import lax, tree, device_put, jit
import jax.numpy as jnp
from .util import (
    State,
    init_dynamic_state, get_axial_induction_fn,
    get_thrust_fn, make_params,
    make_constants,to_result
)
from .solver import cc_solver_step, sequential_solve_step, turbopark_solver, empirical_gauss_solver


def make_yaw_runner(state: State):
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)

    (params,
     enable_secondary_steering,
     enable_transverse_velocities,
     enable_yaw_added_recovery) = make_params(state)

    const, _, tilt_angles = make_constants(state)
    init = init_dynamic_state(state.grid, state.flow, state.wake.model_strings['velocity_model'])

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
    velocity_model_name = state.wake.model_strings['velocity_model']
    deflection_model = state.wake.deflection_model
    turbulence_model = state.wake.turbulence_model
    combination_model = state.wake.combination_model

    T = int(params.T)

    if velocity_model_name == "cc":
        def _loop_with_yaw(yaw_angles_sorted: jnp.ndarray):
            def body(st, ii):
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

            # fori_loop since we only need the final state
            # final_state = lax.fori_loop(0, T, body, init)

            final_state, _ = lax.scan(body, init, jnp.arange(T))
            return to_result(final_state)

    elif velocity_model_name == "turbopark":
        # needs implementation
        pass

    elif velocity_model_name == "empirical_gauss":
        # needs implementation
        pass

    else:
        def _loop_with_yaw(yaw_angles_sorted: jnp.ndarray):
            def body(ii, st):
                ii32 = lax.convert_element_type(ii, jnp.int32)
                st_next, _ = sequential_solve_step(
                    st, ii32, params,
                    thrust_fn, axial_fn,
                    velocity_model, deflection_model,
                    turbulence_model, combination_model,
                    yaw_angles_sorted, tilt_angles,
                    **const,
                    enable_secondary_steering=enable_secondary_steering,
                    enable_yaw_added_recovery=enable_yaw_added_recovery,
                    enable_transverse_velocities=enable_transverse_velocities
                )
                return st_next#, None

            # fori_loop since we only need the final state. If enabling, swittch ii and st around in the body function
            final_state = lax.fori_loop(0, T, body, init)

            # Scan loop
            # final_state, _ = lax.scan(body, init, jnp.arange(T))
            return to_result(final_state)

    # Single compiled function taking only yaw
    return jit(_loop_with_yaw)
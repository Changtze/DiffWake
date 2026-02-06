# --- imports ---------------------------------------------------------------
from functools   import partial
from jax         import lax, jit
from typing import Callable
from jax         import lax, jit

import jax.numpy as jnp

from .util_agnostic import (
    init_dynamic_state, Params, DynamicState,
    Result, State,
    get_axial_induction_fn, get_thrust_fn,
    make_params, to_result,make_constants

)
from .solver import cc_solver_step           


def simulate(state: State) -> Result:
    """Forward run with *fixed* yaw angles (no grad wrt yaw)."""
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)

    (params,
    enable_secondary_steering,
    enable_transverse_velocities,
    enable_yaw_added_recovery) = make_params(state)

    const,yaw_angles, tilt_angles   = make_constants(state)
    init    = init_dynamic_state(state.grid, state.flow)
    T_int  = int(params.T)

    result_state = _simulate_scan(T_int,
                             params, 
                             thrust_fn,
                             axial_fn,
                             state.wake.velocity_model,
                             state.wake.deflection_model,
                             state.wake.turbulence_model, 
                             yaw_angles,
                             tilt_angles,
                             const, 
                             init,
                             enable_secondary_steering=bool(enable_secondary_steering),
                             enable_transverse_velocities=bool(enable_transverse_velocities),
                             enable_yaw_added_recovery=bool(enable_yaw_added_recovery))
    return to_result(result_state)


def simulate_simp(state: State) -> Result:
    """Forward run with *fixed* yaw angles (no grad wrt yaw)."""
    axial_fn  = get_axial_induction_fn(state.flow, state.farm, state.grid)
    thrust_fn = get_thrust_fn(state.flow, state.farm)
    (params,
    enable_secondary_steering,
    enable_transverse_velocities,
    enable_yaw_added_recovery) = make_params(state)
    const,yaw_angles, tilt_angles   = make_constants(state)
    init    = init_dynamic_state(state.grid, state.flow)
    T_int  = int(params.T)
    result_state = _simulate(T_int,
                             params, 
                             thrust_fn,
                             axial_fn,
                             state.wake.velocity_model,
                             state.wake.deflection_model,
                             state.wake.turbulence_model, 
                             yaw_angles,
                             tilt_angles,
                             const, 
                             init,
                             enable_secondary_steering=bool(enable_secondary_steering),
                             enable_transverse_velocities=bool(enable_transverse_velocities),
                             enable_yaw_added_recovery=bool(enable_yaw_added_recovery))
    return to_result(result_state)


def _simulate(T: int,
            params: Params,
            thrust_function: Callable,
            axial_induction_func:Callable,
            velocity_model:  Callable,
            deflection_model: Callable,
            turbulence_model: Callable, 
            yaw_angles:jnp.array,
            tilt_angles:jnp.array,
            const: dict,
            state: State,
            enable_secondary_steering: bool,
            enable_transverse_velocities: bool,
            enable_yaw_added_recovery: bool,):

    # What conditions are necessary for a specific solver to be used?
    if state.wake.model_strings['velocity_model'] == 'gauss':
        pass



    for i in range(T):
        state, _ = cc_solver_step(state, i, params,thrust_function,
                                    axial_induction_func,velocity_model, 
                                    deflection_model, turbulence_model,yaw_angles, tilt_angles,
                                    **const,
                                    enable_secondary_steering=enable_secondary_steering,
                                    enable_transverse_velocities=enable_transverse_velocities,
                                    enable_yaw_added_recovery=enable_yaw_added_recovery)

    return state
    

@partial(jit, static_argnames=["T", "velocity_model", "turbulence_model",
                               "enable_secondary_steering",
                                "enable_transverse_velocities",
                                "enable_yaw_added_recovery"],donate_argnames=("init_state",))

def _simulate_scan(  T:int,
            params: Params,
            thrust_fn: Callable,
            axial_fn:Callable,
            velocity_model:  Callable,
            deflection_model: Callable,
            turbulence_model: Callable, 
            yaw_angles_sorted:jnp.array,
            tilt_angles_sorted:jnp.array,
            const: dict,
            init_state: State,
            enable_secondary_steering: bool,
            enable_transverse_velocities: bool,
            enable_yaw_added_recovery: bool,):

    def body(ii, st):
        ii32 = lax.convert_element_type(ii, jnp.int32)
        st_next, _ = cc_solver_step(
            st, ii32, params,
            thrust_fn, axial_fn,
            velocity_model, deflection_model, turbulence_model,
            yaw_angles_sorted, tilt_angles_sorted,
            **const,
            enable_secondary_steering=enable_secondary_steering,
            enable_transverse_velocities=enable_transverse_velocities,
            enable_yaw_added_recovery=enable_yaw_added_recovery,
        )
        return st_next

    final_state = lax.fori_loop(0, T, body, init_state)
    return final_state



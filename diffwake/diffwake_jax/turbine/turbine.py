import jax.numpy as jnp
from flax import struct
from typing import Callable, Dict, Iterable, Union, Optional,Tuple
from .operation_models import cosine_loss_axial_induction, cosine_loss_thrust_coefficient
from jax import lax


def power(
    velocities: jnp.ndarray,                      # [n_findex, n_grid, n_grid]
    air_density: float,
    power_function: Callable,                     
    yaw_angles: jnp.ndarray,                      # [n_findex]
    tilt_angles: jnp.ndarray,                     # [n_findex]
    power_thrust_table: dict,                     # Dict with power curve and constants
    cubature_weights: Union[jnp.ndarray, None] = None,
) -> jnp.ndarray:
    """
    Compute power for a single turbine type (no type map).
    """
    # Construct full argument dictionary for the turbine power function
    power_model_kwargs = {
        "power_thrust_table": power_thrust_table,
        "velocities": velocities,
        "air_density": air_density,
        "yaw_angles": yaw_angles,
        "tilt_angles": tilt_angles,
        "cubature_weights": cubature_weights,
    }

    return power_function(**power_model_kwargs)

def thrust_coefficient(
    velocities: jnp.ndarray,                        # shape: (n_findex, n_turbines, H, W)
    yaw_angles: jnp.ndarray,                        # shape: (n_findex, n_turbines)
    tilt_angles: jnp.ndarray,                       # shape: (n_findex, n_turbines)
    thrust_fn: Callable,                            
    tilt_interp: Callable,                          
    power_thrust_table: dict,                       # dictionary with "wind_speed", "thrust_coefficient", etc.
    correct_cp_ct_for_tilt: bool = False,            # shape: (n_findex, n_turbines)
    ix_filter: Union[jnp.ndarray, None] = None,
    cubature_weights: Union[jnp.ndarray, None] = None,
) -> jnp.ndarray:

    if ix_filter is not None:
        velocities  = lax.dynamic_index_in_dim(velocities,  ix_filter, axis=1, keepdims=True)
        yaw_angles  = lax.dynamic_index_in_dim(yaw_angles,  ix_filter, axis=1, keepdims=True)
        tilt_angles = lax.dynamic_index_in_dim(tilt_angles, ix_filter, axis=1, keepdims=True)
    return thrust_fn(
        power_thrust_table=power_thrust_table,
        velocities=velocities,                          # (n_findex, n_turbines, H, W)
        yaw_angles=yaw_angles,                          # (n_findex, n_turbines)
        tilt_angles=tilt_angles,                        # (n_findex, n_turbines)
        tilt_interp=tilt_interp,
        cubature_weights=cubature_weights,
        correct_cp_ct_for_tilt=correct_cp_ct_for_tilt,  # (n_findex, n_turbines)
    )


def axial_induction(
    velocities: jnp.ndarray,                          # shape: (B, T, H, W)
    yaw_angles: jnp.ndarray,                          # shape: (B, T)
    tilt_angles: jnp.ndarray,                         # shape: (B, T)
    axial_induction_function: Callable,
    tilt_interp: Callable,
    turbine_power_thrust_table: Union[dict, Callable],
    correct_cp_ct_for_tilt: bool = False,
    ix_filter: Union[int, None] = None,
    cubature_weights: Union[jnp.ndarray, None] = None,
    multidim_condition: Union[Tuple, None] = None,
) -> jnp.ndarray:
    
    if ix_filter is not None:
        idx = jnp.asarray(ix_filter, jnp.int32)  # if ix_filter is traced, this stays dynamic

        velocities  = lax.dynamic_index_in_dim(velocities,  idx, axis=1, keepdims=True)  # (B,1,H,W)
        yaw_angles  = lax.dynamic_index_in_dim(yaw_angles,  idx, axis=1, keepdims=True)  # (B,1)
        tilt_angles = lax.dynamic_index_in_dim(tilt_angles, idx, axis=1, keepdims=True)  # (B,1)

    # Select the turbine's power/thrust curve table
    if "thrust_coefficient" in turbine_power_thrust_table:
        power_thrust_table = turbine_power_thrust_table
    else:
        # Select based on multidimensional conditions
        if multidim_condition is None:
            raise ValueError("multidim_condition must be specified when using multiple turbine models.")
        power_thrust_table = turbine_power_thrust_table[multidim_condition]

    # Prepare kwargs for the axial induction function
    axial_induction_model_kwargs = {
        "power_thrust_table": power_thrust_table,
        "velocities": velocities,
        "yaw_angles": yaw_angles,
        "tilt_angles": tilt_angles,
        "tilt_interp": tilt_interp,
        "cubature_weights": cubature_weights,
        "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
    }
    turb_axial_induction = axial_induction_function(**axial_induction_model_kwargs)

    # Output shape: (B, T)
    axial_induction = turb_axial_induction

    return axial_induction


def select_multidim_condition(
    condition: Union[dict, tuple],
    specified_conditions: Iterable[tuple]
) -> tuple:
    """
    Convert condition to the type expected by power_thrust_table and select
    nearest specified condition.

    Args:
        condition: A condition tuple or dict of values.
        specified_conditions: A list or iterable of condition tuples.

    Returns:
        tuple: The closest matching condition tuple from the specified_conditions.
    """
    if isinstance(condition, tuple):
        pass
    elif isinstance(condition, dict):
        condition = tuple(condition.values())
    else:
        raise TypeError("condition should be of type dict or tuple.")

    # Convert to jnp arrays
    specified_conditions_array = jnp.array(specified_conditions, dtype=jnp.float32)  # shape (N, D)
    condition_array = jnp.array(condition, dtype=jnp.float32)                         # shape (D,)

    # For each dimension, find nearest entry in that dimension among specified conditions
    nearest_condition = []
    for i in range(condition_array.shape[0]):
        diffs = jnp.abs(specified_conditions_array[:, i] - condition_array[i])
        nearest_idx = jnp.argmin(diffs)
        nearest_value = specified_conditions_array[nearest_idx, i]
        nearest_condition.append(float(nearest_value))  # ensure Python float for JSON compatibility etc.

    return tuple(nearest_condition)


def make_tilt_interp(wind_speeds: jnp.ndarray, tilt: jnp.ndarray):
    def interp_fn(ws: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(ws, wind_speeds, tilt)
    return interp_fn

@struct.dataclass
class Turbine:
    turbine_type: str
    operation_model: str
    rotor_diameter: float
    hub_height: float
    TSR: float
    power_thrust_table: Dict[str, jnp.ndarray]
    correct_cp_ct_for_tilt: bool = False
    floating_tilt_table: Optional[Dict[str, jnp.ndarray]] = None

    # Function handles (must be static and non-JITed)
    power_function: Optional[Callable] = struct.field(pytree_node=False, default=None)
    thrust_coefficient_function: Optional[Callable] = cosine_loss_thrust_coefficient
    axial_induction_function: Optional[Callable] = cosine_loss_axial_induction
    tilt_interp: Optional[Callable] = struct.field(pytree_node=False, default=None)

    @property
    def rotor_radius(self):
        return self.rotor_diameter / 2.0

    @property
    def rotor_area(self):
        return jnp.pi * self.rotor_radius ** 2

    def get_tilt(self, wind_speed: jnp.ndarray) -> jnp.ndarray:
        if self.correct_cp_ct_for_tilt and self.tilt_interp:
            return self.tilt_interp(wind_speed)
        return jnp.zeros_like(wind_speed)
 
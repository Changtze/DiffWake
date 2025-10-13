import torch
from typing import Callable, Dict, Iterable, Union
from .operation_models import CosineLossTurbine, cosine_loss_axial_induction,cosine_loss_thrust_coefficient  # replace with actual path
from typing import Callable, Optional, Dict

from ..util.interp1d import interp1d

TURBINE_MODEL_MAP = {
    "operation_model": {
        "cosine-loss": CosineLossTurbine,
    },
}

import torch
from typing import Callable, Union

def power(
    velocities: torch.Tensor,                      # [n_findex, n_grid, n_grid]
    air_density: float,
    power_function: Callable,                     # Single turbine model power function
    yaw_angles: torch.Tensor,                     # [n_findex]
    tilt_angles: torch.Tensor,                    # [n_findex]
    power_thrust_table: dict,                     # Dict with power curve and constants
    average_method: str = "cubic-mean",
    cubature_weights: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
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
        "average_method": average_method,
        "cubature_weights": cubature_weights,
    }

    # Direct call to the single power function
    return power_function(**power_model_kwargs)  # returns [n_findex] in watts

def thrust_coefficient(
    velocities: torch.Tensor,                       # shape: (n_findex, n_turbines, H, W)
    turbulence_intensities: torch.Tensor,           # shape: (n_findex, n_turbines)
    air_density: float,
    yaw_angles: torch.Tensor,                       # shape: (n_findex, n_turbines)
    tilt_angles: torch.Tensor,                      # shape: (n_findex, n_turbines)
    power_setpoints: torch.Tensor,                  # shape: (n_findex, n_turbines)
    thrust_fn: Callable,                            # the thrust coefficient function for your turbine
    tilt_interp: Callable,                          # a tilt interpolator (can be a placeholder)
    power_thrust_table: dict,                       # dictionary with "wind_speed", "thrust_coefficient", etc.
    correct_cp_ct_for_tilt: torch.Tensor,           # shape: (n_findex, n_turbines)
    ix_filter: torch.Tensor | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        turbulence_intensities = turbulence_intensities[:, ix_filter]
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]

    # Run the turbine-specific thrust coefficient model directly
    return thrust_fn(
        power_thrust_table=power_thrust_table,
        velocities=velocities,  #(n_findex, n_turbines, H, W)
        yaw_angles=yaw_angles,    # shape: (n_findex, n_turbines)
        tilt_angles=tilt_angles, # shape: (n_findex, n_turbines)
        tilt_interp=tilt_interp,
        average_method=average_method,
        cubature_weights=cubature_weights,
        correct_cp_ct_for_tilt=correct_cp_ct_for_tilt,  # shape: (n_findex, n_turbines)
    )


def axial_induction(
    velocities: torch.Tensor,
    turbulence_intensities: torch.Tensor,
    air_density: float,
    yaw_angles: torch.Tensor,
    tilt_angles: torch.Tensor,
    power_setpoints: torch.Tensor,
    axial_induction_function: Callable,
    tilt_interp: Callable,
    correct_cp_ct_for_tilt: torch.Tensor,
    turbine_power_thrust_table: Callable,
    ix_filter: int = None,
    average_method: str = "cubic-mean",
    cubature_weights: torch.Tensor = None,
    multidim_condition: tuple = None,
) -> torch.Tensor:

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:,ix_filter]#.unsqueeze(1)
        yaw_angles = yaw_angles[:, ix_filter]#.unsqueeze(1)
        tilt_angles = tilt_angles[:, ix_filter]#.unsqueeze(1)
        if isinstance(correct_cp_ct_for_tilt, bool):
            pass
        else:
            correct_cp_ct_for_tilt = correct_cp_ct_for_tilt

    # Initialize axial_induction tensor
    axial_induction = torch.zeros(velocities.shape[:2], dtype=velocities.dtype, device=velocities.device)

    # Unique turbine types


    if "thrust_coefficient" in turbine_power_thrust_table:
        power_thrust_table = turbine_power_thrust_table
    else:
        multidim_condition = select_multidim_condition(
            multidim_condition,
            list(turbine_power_thrust_table.keys())
        )
        power_thrust_table = turbine_power_thrust_table[multidim_condition]

    
    axial_induction_model_kwargs = {
        "power_thrust_table": power_thrust_table,
        "velocities": velocities,
        "yaw_angles": yaw_angles,
        "tilt_angles": tilt_angles,    
        "tilt_interp": tilt_interp,
        "average_method": average_method,
        "cubature_weights": cubature_weights,
        "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
    }

    turb_axial_induction = axial_induction_function(**axial_induction_model_kwargs)

    axial_induction += turb_axial_induction

    return axial_induction


import torch
def select_multidim_condition(
    condition: Union[dict, tuple],
    specified_conditions: Iterable[tuple]
) -> tuple:
    """
    Convert condition to the type expected by power_thrust_table and select
    nearest specified condition, using PyTorch.

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

    specified_conditions_tensor = torch.tensor(specified_conditions, dtype=torch.float32)
    condition_tensor = torch.tensor(condition, dtype=torch.float32)

    nearest_condition = []
    for i, c in enumerate(condition_tensor):
        diffs = torch.abs(specified_conditions_tensor[:, i] - c)
        nearest_value = specified_conditions_tensor[:, i][torch.argmin(diffs)].item()
        nearest_condition.append(nearest_value)

    return tuple(nearest_condition)




class Turbine:
    def __init__(
        self,
        turbine_type: str,
        operation_model: str,
        rotor_diameter: float,
        hub_height: float,
        TSR: float,
        power_thrust_table: Dict[str, torch.Tensor],
        correct_cp_ct_for_tilt: bool = False,
        floating_tilt_table: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.turbine_type = turbine_type
        self.operation_model = operation_model
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.TSR = TSR
        self.power_thrust_table = power_thrust_table
        self.correct_cp_ct_for_tilt = correct_cp_ct_for_tilt
        self.floating_tilt_table = floating_tilt_table

        # Computed values
        self.rotor_radius = self.rotor_diameter / 2.0
        self.rotor_area = torch.pi * self.rotor_radius ** 2

        # Function handles (set to default but replaceable)
        self.power_function = CosineLossTurbine.power
        self.thrust_coefficient_function = cosine_loss_thrust_coefficient
        


        self.axial_induction_function = cosine_loss_axial_induction
        self.tilt_interp = None

        if self.correct_cp_ct_for_tilt:
            self._initialize_tilt_interpolation()

    def _initialize_tilt_interpolation(self):
        if self.floating_tilt_table is None:
            raise ValueError("Tilt correction is enabled, but no tilt table provided.")

        wind_speed = self.floating_tilt_table["wind_speed"]
        tilt = self.floating_tilt_table["tilt"]

        if wind_speed.ndim != 1 or tilt.ndim != 1:
            raise ValueError("Tilt and wind_speed must be 1D tensors.")
        if wind_speed.shape != tilt.shape:
            raise ValueError("Tilt and wind_speed must be the same size.")

        def linear_interp(x: torch.Tensor) -> torch.Tensor:
            return torch.interp(x, wind_speed, tilt)  # PyTorch >= 2.0

        self.tilt_interp = linear_interp

    def get_tilt(self, wind_speed: torch.Tensor) -> torch.Tensor:
        if self.tilt_interp:
            return self.tilt_interp(wind_speed)
        return torch.zeros_like(wind_speed)

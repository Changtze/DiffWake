from .util import load_yaml
from pathlib import Path
from typing import Any, Dict, Optional

#from .simulator import simulate
from flax import struct
import jax.numpy as jnp
from .farm import Farm
from .grid import TurbineGrid
from .flow_field import FlowField
from .turbine.turbine import Turbine
from .turbine.turbine import power
from .wake_combination.sosfs import SOSFS
from .wake_deflection.gauss import GaussVelocityDeflection
from .wake_turbulence.crespo import CrespoHernandez
from .wake_vel.culm_gauss import CumulativeGaussCurlVelocityDeficit, CumulativeGaussCurlVelocityDeficitAsBs
from .wake_vel.gauss import GaussVelocityDeficit
from .util_agnostic import State, Config
from .wake import WakeModelManager

POWER_SETPOINT_DEFAULT = 1.e12

MODEL_MAP = {
    "combination_model": {
        "sosfs": SOSFS
    },
    "deflection_model": {
        "gauss": GaussVelocityDeflection
    },
    "turbulence_model": {
        "crespo_hernandez": CrespoHernandez
    },
    "velocity_model": {
        "cc": CumulativeGaussCurlVelocityDeficit,
        "gauss": GaussVelocityDeficit
    }
}


def load_input( farm_path: Path, generator_path: Path) -> Config:
    farm_dict = load_yaml(farm_path)
    generator_dict = load_yaml(generator_path)
    flow_field_dict = farm_dict["flow_field"]     

    wind_height = flow_field_dict['reference_wind_height']
    if wind_height < 0.:
        wind_height = farm_dict['farm']['turbine_type'][0]['hub_height']

    for key, value in flow_field_dict.items():
        if key == "reference_wind_height":
            flow_field_dict[key] = wind_height
        if isinstance(value, list):
            flow_field_dict[key] = jnp.array(value)

    layout_dict = farm_dict['farm']
    for key, value in layout_dict.items():
        if isinstance(value, list) and isinstance(value[0], float):
            layout_dict[key] = jnp.array(value)    
    return Config(generator_dict, farm_dict, flow_field_dict, layout_dict)


def create_wake(farm_dict):
    wake_dict = farm_dict['wake']

    # Get velocity model
    vel_model_string = wake_dict['model_strings']['velocity_model'].lower()
    vel_model = MODEL_MAP["velocity_model"][vel_model_string]
    if vel_model_string == "none":
        vel_model_parameters = None
    else:
        vel_model_parameters = wake_dict['wake_velocity_parameters'][vel_model_string]

    # Get deflection model
    def_model_string = wake_dict['model_strings']['deflection_model'].lower()
    def_model = MODEL_MAP["deflection_model"][def_model_string]
    if def_model_string == "none":
        def_model_parameters = None
    else:
        def_model_parameters = wake_dict['wake_deflection_parameters'][def_model_string]

    # Get turbulence model
    turb_model_string = wake_dict['model_strings']['turbulence_model'].lower()
    turb_model = MODEL_MAP["turbulence_model"][turb_model_string]
    if turb_model_string == "none":
        turb_model_parameters = None
    else:
        turb_model_parameters = wake_dict['wake_turbulence_parameters'][turb_model_string]

    # Get wake combination model
    combo_model_string = wake_dict['model_strings']['combination_model'].lower()
    combo_model = MODEL_MAP["combination_model"][combo_model_string]



    wake = WakeModelManager(velocity_model=vel_model(**vel_model_parameters),
                            deflection_model=def_model(**def_model_parameters),
                            turbulence_model=turb_model(**turb_model_parameters),
                            combination_model=combo_model(),
                            enable_secondary_steering = wake_dict['enable_secondary_steering'],
                            enable_yaw_added_recovery = wake_dict['enable_yaw_added_recovery'],
                            enable_active_wake_mixing = wake_dict['enable_active_wake_mixing'],
                            enable_transverse_velocities = wake_dict['enable_transverse_velocities'],
                            model_strings=wake_dict['model_strings'])


    # wake1 = WakeModelManager(velocity_model = CumulativeGaussCurlVelocityDeficit(**farm_dict['wake']['wake_velocity_parameters']['cc']),
    #                         deflection_model = GaussVelocityDeflection(**farm_dict['wake']['wake_deflection_parameters']['gauss']),
    #                         turbulence_model= CrespoHernandez(**farm_dict['wake']['wake_turbulence_parameters']['crespo_hernandez']),
    #                         #combination_model = SOSFS(),
    #                         enable_secondary_steering = farm_dict['wake']['enable_secondary_steering'],
    #                         enable_yaw_added_recovery = farm_dict['wake']['enable_yaw_added_recovery'],
    #                         #enable_active_wake_mixing = farm_dict['wake']['enable_active_wake_mixing'],
    #                         enable_transverse_velocities = farm_dict['wake']['enable_transverse_velocities'])


    return wake


def create_wake_asbs(farm_dict):
    wake_dict = farm_dict['wake']

    vel_model_string = wake_dict['model_strings']['velocity_model'].lower()
    vel_model = MODEL_MAP["velocity_model"][vel_model_string]
    vel_parameters = farm_dict.get('wake', {}).get('wake_velocity_parameters', {}).get(vel_model_string, {})
    vel_parameters.pop('a_s', None)
    vel_parameters.pop('b_s', None)

    # Get deflection model
    def_model_string = wake_dict['model_strings']['deflection_model'].lower()
    def_model = MODEL_MAP["deflection_model"][def_model_string]
    if def_model_string == "none":
        def_model_parameters = None
    else:
        def_model_parameters = wake_dict['wake_deflection_parameters'][def_model_string]

    # Get turbulence model
    turb_model_string = wake_dict['model_strings']['turbulence_model'].lower()
    turb_model = MODEL_MAP["turbulence_model"][turb_model_string]
    if turb_model_string == "none":
        turb_model_parameters = None
    else:
        turb_model_parameters = wake_dict['wake_turbulence_parameters'][turb_model_string]

    # Get wake combination model
    combo_model_string = wake_dict['model_strings']['combination_model'].lower()
    combo_model = MODEL_MAP["combination_model"][combo_model_string]


    wake = WakeModelManager(velocity_model = vel_model(**vel_parameters),
                            deflection_model=def_model(**def_model_parameters),
                            turbulence_model=turb_model(**turb_model_parameters),
                            combination_model=combo_model(),
                            enable_secondary_steering = wake_dict['enable_secondary_steering'],
                            enable_yaw_added_recovery = wake_dict['enable_yaw_added_recovery'],
                            enable_active_wake_mixing = wake_dict['enable_active_wake_mixing'],
                            enable_transverse_velocities = wake_dict['enable_transverse_velocities'])

    return wake


def create_grid(layout, generator, farm, flow):
    z_coords = jnp.full_like(layout['layout_x'], farm['farm']['turbine_type'][0]['hub_height'])
    coords = jnp.column_stack([layout['layout_x'],layout['layout_y'], z_coords])
    grid = TurbineGrid.create( turbine_coordinates=coords,
                                turbine_diameter=generator['rotor_diameter'],
                                wind_directions=flow['wind_directions'],
                                grid_resolution=3)
    return grid


def create_farm(layout, generator, sorted_indices):
    farm = Farm.create(layout_x = layout["layout_x"],
                        layout_y = layout["layout_y"], 
                        turbine_type_dict = generator, 
                        turbine_class =Turbine ).initialize(sorted_indices)    
    return farm


def create_flow_field(flow, grid):
    return FlowField(**flow).initialize_velocity_field(grid)

def create_state(cfg: Config, learn_as_bs = False):
    
    if learn_as_bs:
        wake = create_wake_asbs(cfg.farm)
    else:
        wake = create_wake(cfg.farm)
    grid = create_grid(cfg.layout, cfg.generator, cfg.farm, cfg.flow_field)
    farm = create_farm(cfg.layout, cfg.generator, grid.sorted_indices)
    flow = create_flow_field(cfg.flow_field, grid)

    return State(farm, grid, flow, wake)

def alter_yaw_angles(yaw_angles, state):
    idx = state.grid.sorted_indices[:, :, 0, 0]
    yaw_s   = jnp.take_along_axis(yaw_angles,  idx, axis=1)
    new_state = state.replace(
    farm=state.farm.replace(
    yaw_angles=yaw_angles,
    yaw_angles_sorted=yaw_s))
    return new_state


def turbine_powers(state: State) -> jnp.ndarray:
    return power(
        velocities=state.flow.u,
        air_density=state.flow.air_density,
        power_function=state.farm.power_function,
        yaw_angles=state.farm.yaw_angles,
        tilt_angles=state.farm.tilt_angles,
        power_thrust_table=state.farm.power_thrust_table,
    )


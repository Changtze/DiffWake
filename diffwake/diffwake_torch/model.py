import torch
from pathlib import Path
from typing import Optional
from torch import Tensor
from floris.utilities import load_yaml
from .grid import TurbineGrid
from .flow_field import FlowField
from .farm import Farm
from .turbine.turbine import Turbine
from .wake import WakeModelManager
from .wake_combination.sosfs import SOSFS
from .wake_deflection.gauss import GaussVelocityDeflection

from .wake_turbulence.crespo import CrespoHernandez
from .wake_vel.culm_gauss import CumulativeGaussCurlVelocityDeficit
from .solvers import cc_solver#, CCSolver#, cc_solver2
#from .solvers3 import CCSolver#, cc_solver2

from.turbine.turbine import power
class FlorisModel:
    def __init__(self, farm_path: Path, generator_path):


        self.farm_path = farm_path
        self.generator_path = generator_path
        self.farm_dict = load_yaml(farm_path)
        self.generator_dict = load_yaml(generator_path)
        flow_field_dict = self.farm_dict["flow_field"]
        for key, value in flow_field_dict.items():
            if isinstance(value, list):
                flow_field_dict[key] = torch.tensor(value)
        self.flow_field_dict =flow_field_dict
        farm_dict_farm = self.farm_dict['farm']
        for key, value in farm_dict_farm.items():
            if isinstance(value, list) and isinstance(value[0], float):
                farm_dict_farm[key] = torch.tensor(value)
        self.farm_dict_farm = farm_dict_farm

        self.create_core()
        #self.create_solver()
        self.cc_solver_func = cc_solver
        
    def create_core(self):

        self.x_coords = self.farm_dict_farm['layout_x']
        self.y_coords = self.farm_dict_farm['layout_y']
        self.z_coords = torch.full_like(self.x_coords, self.farm_dict['farm']['turbine_type'][0]['hub_height'])
        self.coords = torch.column_stack([self.farm_dict_farm['layout_x'],self.farm_dict_farm['layout_y'], self.z_coords]).clone()

        combination_model = SOSFS()
        deflection_model = GaussVelocityDeflection(**self.farm_dict['wake']['wake_deflection_parameters']['gauss'])
        turbulence_model = CrespoHernandez(**self.farm_dict['wake']['wake_turbulence_parameters']['crespo_hernandez'])
        velocity_model = CumulativeGaussCurlVelocityDeficit(**self.farm_dict['wake']['wake_velocity_parameters']['cc'])

        self.wake = WakeModelManager(combination_model=combination_model,
                                     
                        deflection_model = deflection_model,
                        turbulence_model = turbulence_model,
                        velocity_model=velocity_model, 
                        enable_secondary_steering = self.farm_dict['wake']['enable_secondary_steering'],
                        enable_yaw_added_recovery = self.farm_dict['wake']['enable_yaw_added_recovery'],
                        enable_transverse_velocities = self.farm_dict['wake']['enable_transverse_velocities'],
                        enable_active_wake_mixing = self.farm_dict['wake']['enable_active_wake_mixing'],)

        self.flow_field = FlowField(**self.flow_field_dict)

        if self.flow_field_dict['reference_wind_height'] < 0.:
            self.flow_field.reference_wind_height = self.farm_dict['farm']['turbine_type'][0]['hub_height']
        else:
            self.flow_field.reference_wind_height = self.flow_field['reference_wind_height']


        self.farm = Farm(layout_x = self.x_coords,
                layout_y = self.y_coords, 
                turbine_type_dict = self.generator_dict, 
                turbine_class =Turbine )
        
        self.turbine = Turbine(**self.generator_dict)

        self.grid = TurbineGrid(turbine_coordinates=self.coords,
                     turbine_diameter=self.generator_dict['rotor_diameter'],
                     wind_directions=self.flow_field_dict['wind_directions'],
                     grid_resolution=3)
        
    #def create_solver(self):
    #    self.solver = CCSolver(self.farm, self.grid,self.flow_field, self.wake)

    def initialize_domain(self):
        self.flow_field.initialize_velocity_field(self.grid)
        if self.farm.yaw_angles is None:
            self.farm.set_yaw_angles_to_ref_yaw(self.flow_field.n_findex)
        self.farm.set_tilt_to_ref_tilt(self.flow_field.n_findex)
        self.farm.set_power_setpoints_to_ref_power(self.flow_field.n_findex)
        self.farm.initialize(self.grid.sorted_indices)        

    def set(
        self,
        wind_speeds: Optional[Tensor] = None,
        wind_directions: Optional[Tensor] = None,
        turbulence_intensities: Optional[Tensor] = None,
        layout_x: Optional[Tensor] = None,
        layout_y: Optional[Tensor] = None,
        yaw_angles: Optional[Tensor] = None,
    ):
        if wind_speeds is not None:
            self.flow_field_dict["wind_speeds"] = wind_speeds.clone()
        if wind_directions is not None:
            self.flow_field_dict["wind_directions"] = wind_directions.clone()
        if turbulence_intensities is not None:
            self.flow_field_dict["turbulence_intensities"] = turbulence_intensities.clone()
        if layout_x is not None:
            self.farm_dict_farm["layout_x"] = layout_x.clone()
        if layout_y is not None:
            self.farm_dict_farm["layout_y"] = layout_y.clone()
        if (layout_x is not None) or (layout_y is not None):
            self.z_coords = torch.tensor([self.farm_dict['farm']['turbine_type'][0]['hub_height']]*16)
            self.coords = torch.column_stack([layout_x.clone(), layout_y.clone(), self.z_coords.clone()])
        self.create_core()
        if yaw_angles is not None:
            self.set_yaw_angles(yaw_angles)
        #self.create_solver()

    def set_yaw_angles(self, yaw_angles):
        self.farm.set_yaw_angles(yaw_angles)

    def run(self):
        self.run_func()

    def run_func(self):
        self.initialize_domain()
        cc_solver(self.farm, self.flow_field, self.grid, self.wake)
        self.finalize()
 
    def run_for_yaw_optim(self, yaw_angles):

        yaw_angles_sorted = torch.gather(yaw_angles, dim=1, index=self.grid.sorted_indices[:, :, 0, 0] )
        self.cc_solver.update_yaw_angles(yaw_angles_sorted)
        self.farm.yaw_angles = yaw_angles
        self.farm.yaw_angles_sorted = yaw_angles_sorted

        v_wake, w_wake, turb_u_wake, ti = self.cc_solver.solve_compile()

        self.flow_field.v_sorted = self.flow_field.v_sorted + v_wake
        self.flow_field.w_sorted = self.flow_field.w_sorted+  w_wake
        self.flow_field.u_sorted = turb_u_wake.clone()

        self.flow_field.turbulence_intensity_field_sorted = ti
        self.flow_field.turbulence_intensity_field_sorted_avg = torch.mean(
            ti,
            axis=(2,3)
        )[:, :, None, None]

        self.finalize()

    def finalize(self):
        self.flow_field.finalize(self.grid.unsorted_indices)
        self.farm.finalize(self.grid.unsorted_indices)
        self.state = "USED"

    def get_turbine_powers(self) -> Tensor:
        """Calculates the power at each turbine in the wind farm.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """

        if (self.flow_field.u < 0.0).any():
            print("Some velocities at the rotor are negative.")

        turbine_powers = power(
            velocities=self.flow_field.u.clone(),
            air_density=self.flow_field.air_density,
            power_function=self.farm.power_function,
            yaw_angles=self.farm.yaw_angles.clone(),
            tilt_angles=self.farm.tilt_angles.clone(),
            power_thrust_table=self.farm.power_thrust_table,
        )
        return turbine_powers
    
    def to(self, device: torch.device | str):
        """Move **every** tensor attribute (recursively) to *device*."""
        device = torch.device(device)
        for name, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, name, val.to(device, non_blocking=True))
        self.device = device

        self.farm.to(device)
        self.grid.to(device)
        self.flow_field.to(device)

        return self 
    
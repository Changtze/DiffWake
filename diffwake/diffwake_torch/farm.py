import torch
from typing import Callable, Dict

POWER_SETPOINT_DEFAULT = 1.e12

class Farm:
    def __init__(
        self,
        layout_x: torch.Tensor,
        layout_y: torch.Tensor,
        turbine_type_dict: Dict,
        turbine_class: Callable,
        yaw_angles: torch.Tensor = None,

    ):
        self.layout_x = layout_x
        self.layout_y = layout_y
        self.n_turbines = layout_x.shape[0]

        # Initialize a single turbine instance
        # 
        self.turbine = turbine_class(**turbine_type_dict)
        # Expand static turbine parameters to all turbines
        self.hub_height =  self.turbine.hub_height
        self.rotor_diameter = self.turbine.rotor_diameter
        self.TSR = self.turbine.TSR
        self.ref_tilt = self.turbine.power_thrust_table["ref_tilt"]
        self.correct_cp_ct_for_tilt = float(self.turbine.correct_cp_ct_for_tilt)

        self.power_function = self.turbine.power_function
        self.thrust_coefficient_function = self.turbine.thrust_coefficient_function
        self.axial_induction_function = self.turbine.axial_induction_function
        self.tilt_interp = self.turbine.tilt_interp
        self.power_thrust_table = self.turbine.power_thrust_table

        # Control variables
        self.yaw_angles = yaw_angles
        #if yaw_angles: 
        #    self.yaw_angles = torch.deg2rad(yaw_angles)
        self.tilt_angles = None
        self.power_setpoints = None
        self.awc_modes = None
        self.awc_amplitudes = None
        self.awc_frequencies = None
        self.device = "cpu"


    def initialize(self, sorted_indices: torch.Tensor):
        turbine_sort_idx = sorted_indices[:, :, 0, 0]  # shape: [B, T]
        self.yaw_angles_sorted = torch.gather(self.yaw_angles, dim=1, index=turbine_sort_idx)
        self.tilt_angles_sorted = torch.gather(self.tilt_angles, dim=1, index=turbine_sort_idx)
        self.power_setpoints_sorted = torch.gather(self.power_setpoints, dim=1, index=turbine_sort_idx)

        self.state = "INITIALIZED"



    def expand_farm_properties(self, n_findex: int, sorted_coord_indices: torch.Tensor):
        expand = lambda x: x.expand(n_findex, -1)
        self.tilt_angles_sorted = torch.gather(expand(self.ref_tilts), 1, sorted_coord_indices).to(self.device)

    def finalize(self, unsorted_indices: torch.Tensor):
        idx = unsorted_indices[:, :, 0, 0]
        self.yaw_angles = torch.gather(self.yaw_angles_sorted, 1, idx).to(self.device)
        self.tilt_angles = torch.gather(self.tilt_angles_sorted, 1, idx).to(self.device)

    def set_yaw_angles_to_ref_yaw(self, n_findex: int):
        self.yaw_angles = torch.zeros((n_findex, self.n_turbines), device = self.device)
        self.yaw_angles_sorted = self.yaw_angles
        
    def set_yaw_angles(self, yaw_angles: torch.Tensor):
        self.yaw_angles = yaw_angles.to(self.device)

    def set_tilt_to_ref_tilt(self, n_findex: int):
        self.tilt_angles = torch.ones((n_findex, self.n_turbines), device = self.device) * self.ref_tilt
        self.tilt_angles_sorted = self.tilt_angles.clone()

    def set_power_setpoints_to_ref_power(self, n_findex: int):
        self.power_setpoints = torch.ones((n_findex, self.n_turbines), device = self.device)*POWER_SETPOINT_DEFAULT
        self.power_setpoints_sorted = self.power_setpoints.clone()


    def calculate_tilt_for_eff_velocities(self, rotor_effective_velocities: torch.Tensor) -> torch.Tensor:
        if self.tilt_interp:
            return self.tilt_interp(rotor_effective_velocities)
        return torch.zeros_like(rotor_effective_velocities)

    @property
    def coordinates(self) -> torch.Tensor:
        return torch.stack([self.layout_x, self.layout_y, self.hub_heights], dim=-1)

    def to(self, device: torch.device | str):
        device = torch.device(device)
        for name, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, name, val.to(device, non_blocking=True))
        self.device = device
        return self 
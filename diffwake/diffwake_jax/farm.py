import jax.numpy as jnp
from flax import struct                
from typing import Callable, Dict, Optional, Any

POWER_SETPOINT_DEFAULT = 1.0e12

@struct.dataclass
class Farm:
    layout_x: jnp.ndarray                   
    layout_y: jnp.ndarray                    
    n_turbines: int
    turbine: Any                              

    hub_height: float
    rotor_diameter: float
    TSR: float
    ref_tilt: float
    correct_cp_ct_for_tilt: float
    power_function: Callable = struct.field(pytree_node=False)
    thrust_coefficient_function: Callable = struct.field(pytree_node=False)
    axial_induction_function: Callable = struct.field(pytree_node=False)
    tilt_interp: Optional[Callable] = struct.field(pytree_node=False)
    power_thrust_table: Dict = struct.field(pytree_node=False)

    yaw_angles: Optional[jnp.ndarray] = None      
    tilt_angles: Optional[jnp.ndarray] = None    
    power_setpoints: Optional[jnp.ndarray] = None  

    yaw_angles_sorted: Optional[jnp.ndarray] = None
    tilt_angles_sorted: Optional[jnp.ndarray] = None
    power_setpoints_sorted: Optional[jnp.ndarray] = None
    state: str = "UNINITIALIZED"

    @staticmethod
    def create(
        layout_x: jnp.ndarray,
        layout_y: jnp.ndarray,
        turbine_type_dict: Dict,
        turbine_class: Callable,
        yaw_angles: Optional[jnp.ndarray] = None,
    ) -> "Farm":
        turbine = turbine_class(**turbine_type_dict)
        return Farm(
            layout_x=layout_x,
            layout_y=layout_y,
            n_turbines=layout_x.shape[0],
            turbine=turbine,
            hub_height=turbine.hub_height,
            rotor_diameter=turbine.rotor_diameter,
            TSR=turbine.TSR,
            ref_tilt=turbine.power_thrust_table["ref_tilt"],
            correct_cp_ct_for_tilt=turbine.correct_cp_ct_for_tilt,
            power_function=turbine.power_function,
            thrust_coefficient_function=turbine.thrust_coefficient_function,
            axial_induction_function=turbine.axial_induction_function,
            tilt_interp=turbine.tilt_interp,
            power_thrust_table=turbine.power_thrust_table,
            yaw_angles=yaw_angles if yaw_angles is not None else None,
        )

    """
    def set_yaw_angles(self, yaw_angles: jnp.ndarray, sorted_indices: jnp.ndarray) -> "Farm":
        idx = sorted_indices[:, :, 0, 0]
        yaw_s   = jnp.take_along_axis(yaw_angles,  idx, axis=1)
        return self.replace(yaw_angles=yaw_angles,
                            yaw_angles_sorted=yaw_s)
    """
    def set_yaw_angles_to_ref_yaw(self, n_findex: int) -> "Farm":
        yaw_zero = jnp.zeros((n_findex, self.n_turbines))
        return self.replace(yaw_angles=yaw_zero, yaw_angles_sorted=yaw_zero)

    def set_tilt_to_ref_tilt(self, n_findex: int) -> "Farm":
        tilt_ref = jnp.ones((n_findex, self.n_turbines)) * self.ref_tilt
        return self.replace(tilt_angles=tilt_ref, tilt_angles_sorted=tilt_ref)

    def set_power_setpoints_to_ref_power(self, n_findex: int) -> "Farm":
        psp = jnp.ones((n_findex, self.n_turbines)) * POWER_SETPOINT_DEFAULT
        return self.replace(power_setpoints=psp, power_setpoints_sorted=psp)

    def initialize(self, sorted_indices: jnp.ndarray) -> "Farm":
        B,T,_,_ = sorted_indices.shape
        
        idx = sorted_indices[:, :, 0, 0]

        if self.yaw_angles is None:
            yaw = jnp.zeros((B,T))
        else:
            yaw = self.yaw_angles
        if self.tilt_angles is None:
            tilt = jnp.zeros((B,T))
        else:
            tilt = self.tilt_angles
        yaw_s   = jnp.take_along_axis(yaw,  idx, axis=1)
        tilt_s  = jnp.take_along_axis(tilt, idx, axis=1)
        if self.power_setpoints is not None:
            pset_s  = jnp.take_along_axis(self.power_setpoints, idx, axis=1)
        else:
            pset_s = None
        return self.replace(
            yaw_angles = yaw,
            tilt_angles = tilt,
            yaw_angles_sorted=yaw_s,
            tilt_angles_sorted=tilt_s,
            power_setpoints_sorted=pset_s,
            state="INITIALIZED",
        )

    def finalize(self, unsorted_indices: jnp.ndarray) -> "Farm":
        idx = unsorted_indices[:, :, 0, 0]
        yaw   = jnp.take_along_axis(self.yaw_angles_sorted,  idx, axis=1)
        tilt  = jnp.take_along_axis(self.tilt_angles_sorted, idx, axis=1)
        return self.replace(yaw_angles=yaw, tilt_angles=tilt)

    def expand_farm_properties(
        self, n_findex: int, sorted_coord_idx: jnp.ndarray
    ) -> "Farm":
        expand = lambda x: jnp.broadcast_to(x, (n_findex, x.shape[0]))
        tilt_sort = jnp.take_along_axis(
            expand(jnp.array(self.ref_tilt)), 1, sorted_coord_idx
        )
        return self.replace(tilt_angles_sorted=tilt_sort)

    def calculate_tilt_for_eff_velocities(
        self, v_eff: jnp.ndarray
    ) -> jnp.ndarray:
        if self.tilt_interp is not None:
            return self.tilt_interp(v_eff)
        return jnp.zeros_like(v_eff)

    @property
    def coordinates(self) -> jnp.ndarray:
        return jnp.stack(
            [self.layout_x, self.layout_y, jnp.full_like(self.layout_x, self.hub_height)], axis=-1
        )

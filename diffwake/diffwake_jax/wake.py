from flax import struct
from typing import Any


@struct.dataclass
class WakeModelManager:
    velocity_model: Any         
    deflection_model: Any
    turbulence_model: Any
    #combination_model: Any

    enable_secondary_steering: bool = False
    enable_yaw_added_recovery: bool = False
    enable_active_wake_mixing: bool = False
    enable_transverse_velocities: bool = False

    def deflection_function(self):
        return self.deflection_model.forward

    def velocity_function(self):
        return self.velocity_model.forward

    def turbulence_function(self):
        return self.turbulence_model.forward

    def combination_function(self):
        return self.combination_model.forward
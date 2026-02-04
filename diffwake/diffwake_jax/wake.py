from flax import struct
from typing import Any
import attrs
from attrs import define, field

from .wake_combination import SOSFS
from .wake_vel import GaussVelocityDeficit, CumulativeGaussCurlVelocityDeficit
from .wake_deflection import GaussVelocityDeflection
from .wake_turbulence import CrespoHernandez


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


@struct.dataclass
class WakeModelManager:
    velocity_model: Any         
    deflection_model: Any
    turbulence_model: Any
    # combination_model: Any  # Re-enable for other wake models. Future thing to do

    enable_secondary_steering: bool = field(converter=bool, default=False)
    enable_yaw_added_recovery: bool = field(converter=bool, default=False)
    enable_active_wake_mixing: bool = field(converter=bool, default=False)
    enable_transverse_velocities: bool = field(converter=bool, default=False)

    def deflection_function(self):
        return self.deflection_model.forward

    def velocity_function(self):
        return self.velocity_model.forward

    def turbulence_function(self):
        return self.turbulence_model.forward

    def combination_function(self):
        return self.combination_model.forward
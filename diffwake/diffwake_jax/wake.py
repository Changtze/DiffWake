from flax import struct
from typing import Any
import attrs
from attrs import define, field

from .wake_combination import SOSFS
from .wake_vel import GaussVelocityDeficit, CumulativeGaussCurlVelocityDeficit
from .wake_deflection import GaussVelocityDeflection
from .wake_turbulence import CrespoHernandez


# only really useful if using __attrs_post_init__ so no need for it
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
    combination_model: Any

    model_strings: dict = field(converter = dict)
    enable_secondary_steering: bool = field(converter=bool, default=False)
    enable_yaw_added_recovery: bool = field(converter=bool, default=False)
    enable_active_wake_mixing: bool = field(converter=bool, default=False)
    enable_transverse_velocities: bool = field(converter=bool, default=False)

    wake_deflection_parameters: dict = field(converter=dict)
    wake_turbulence_parameters: dict = field(converter=dict)
    wake_velocity_parameters: dict = field(converter=dict)

    # Not sure these really matter
    @property
    def deflection_function(self):
        return self.deflection_model

    @property
    def velocity_function(self):
        return self.velocity_model
    @property
    def turbulence_function(self):
        return self.turbulence_model

    @property
    def combination_function(self):
        return self.combination_model
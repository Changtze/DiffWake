import torch
from typing import Callable, Dict


class WakeModelManager:
    """
    WakeModelManager is a simplified container for fixed wake models:
    - Velocity: EmpiricalGaussVelocityDeficit
    - Deflection: EmpiricalGaussVelocityDeflection
    - Turbulence: CrespoHernandez
    - Combination: FLS

    All operations are performed using PyTorch.
    """

    def __init__(
        self,
        velocity_model: Callable,
        deflection_model: Callable,
        turbulence_model: Callable,
        combination_model: Callable,

        enable_secondary_steering: bool = False,
        enable_yaw_added_recovery: bool = False,
        enable_active_wake_mixing: bool = False,
        enable_transverse_velocities: bool = False,
    ):
        self.velocity_model = velocity_model
        self.deflection_model = deflection_model
        self.turbulence_model = turbulence_model
        self.combination_model = combination_model

        self.enable_secondary_steering = enable_secondary_steering
        self.enable_yaw_added_recovery = enable_yaw_added_recovery
        self.enable_active_wake_mixing = enable_active_wake_mixing
        self.enable_transverse_velocities = enable_transverse_velocities

    @property
    def deflection_function(self):
        return self.deflection_model.forward

    @property
    def velocity_function(self):
        return self.velocity_model.forward

    @property
    def turbulence_function(self):
        return self.turbulence_model.forward

    @property
    def combination_function(self):
        return self.combination_model.forward

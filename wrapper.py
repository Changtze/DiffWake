from __future__ import annotations
import os
from typing import Tuple, Optional
from pathlib import Path
import sys
from dataclasses import dataclass, replace
from jax.tree_util import tree_map
sys.path.append(os.path.abspath('..'))
import time

import jax.numpy as jnp
jnp.set_printoptions(precision=3, suppress=True)

# Wake model-agnostic edits to DiffWake
from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.util import average_velocity_jax, State, Result, set_cfg
from diffwake.diffwake_jax.turbine.operation_models import power as power_fn
from diffwake.diffwake_jax.simulator import simulate_simp, simulate


"""Wrapper file to run simple DiffWake simulations"""

@dataclass
class DiffWakeConfig:
    data_dir: Path
    turbine_file: str
    generator_file: str
    farm_path: Path = None
    generator_path: Path = None
    dtype: Optional = jnp.float32

    def __post_init__(self):
        if self.farm_path is None:
            self.farm_path: Path = self.data_dir / self.turbine_file
        if self.generator_path is None:
            self.generator_path: Path = self.data_dir / self.generator_file


@dataclass
class DiffWakeParams:
    layout_x: Optional[jnp.ndarray] = None
    layout_y: Optional[jnp.ndarray] = None
    wind_speed: Optional[jnp.ndarray] = None
    wind_directions: Optional[jnp.ndarray] = None
    turbulence_intensities: Optional[jnp.ndarray] = None


class DiffWakeSimulation:
    def __init__(self,
                 config: DiffWakeConfig,
                 params: DiffWakeParams,
                 ):
        self.config = config
        self.params = params
        self.dtype = config.dtype

        self.state = None
        self.result = None
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.params.layout_x is None and self.params.layout_y is None:
            self.sim = load_input(
                farm_path=self.config.farm_path,
                generator_path=self.config.generator_path,
            ).set(
                wind_speeds=self.params.wind_speed,
                wind_directions=self.params.wind_directions,
                turbulence_intensities=self.params.turbulence_intensities
            )
        else:
            self.sim = load_input(
                farm_path=self.config.farm_path,
                generator_path=self.config.generator_path,
            ).set(
                layout_x=self.params.layout_x,
                layout_y=self.params.layout_y,
                wind_speeds=self.params.wind_speed,
                wind_directions=self.params.wind_directions,
                turbulence_intensities=self.params.turbulence_intensities
            )

    def update_params(self, **updates) -> None:
        self.params = replace(self.params, **updates)

    def update_yaw_angles(self, new_yaw_angles: jnp.ndarray) -> None:
        idx = self.state.grid.sorted_indices[:, :, 0, 0]
        yaw_s = jnp.take_along_axis(new_yaw_angles, idx, axis=1)
        new_state = self.state.replace(
            farm=self.state.farm.replace(
                yaw_angles=new_yaw_angles,
                yaw_angles_sorted=yaw_s,
            )
        )
        return new_state


    def run(self, scan: bool = False) -> Tuple[State, Result, float]:
        self.state = create_state(self.sim)

        # cast state into dtype
        self.state = tree_map(
            lambda x: x.astype(self.dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, self.state
        )

        time_start = time.time()
        if scan:
            # Use lax
            self.result = simulate(self.state)
            tree_map(lambda x: x.block_until_ready(), self.result)
        else:
            self.result = simulate_simp(self.state)
            tree_map(lambda x: x.block_until_ready(), self.result)
        time_end = time.time()




        return self.state, self.result, time_end - time_start

    def get_turbine_powers(self) -> jnp.ndarray:
        # Simulation must be run first
        if self.state is None or self.result is None:
            self.state, self.result, _ = self.run()

        return power_fn(
            power_thrust_table=self.state.farm.power_thrust_table,
            velocities=self.result.u_sorted,
            air_density=self.state.flow.air_density,
            yaw_angles=self.state.farm.yaw_angles
        )

    def get_farm_power(self) -> float:
        return jnp.sum(self.get_turbine_powers()) / 1e6  # Convert watts to megawatts

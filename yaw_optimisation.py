"""
DiffWake/JAX: deterministic wind turbine yaw angle optimisation with LBFGS (and other optax optimisers)

"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# DiffWake imports
from diffwake.diffwake_jax.model_agnostic import load_input, create_state
from diffwake.diffwake_jax.yaw_runner_agnostic import make_yaw_runner
from diffwake.diffwake_jax.util_agnostic import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power


def setup_dtype(use_float64: bool = True) -> jnp.dtype:

    jax.config.update("jax_enable_x64", use_float64)
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


# Initialisation


# Runner
def build_state_runner(
        data_dir: Path,
        farm_yaml: str,
        turbine_yaml: str,
        wind_dir_rad: jax.Array,
        wind_speed: jax.Array,
        turb_intensity: jax.Array,
        dtype,
):
    cfg = load_input(
        str(data_dir / farm_yaml),
        str(data_dir / turbine_yaml),
    ).set(
        wind_directions=wind_dir_rad,
        wind_speeds=wind_speed,
        turbulence_intensities=turb_intensity,
    )
    state = create_state(cfg)
    runner = make_yaw_runner(state)

    # N from config
    x0 = jnp.asarray(cfg.layout["layout_x"], dtype=dtype)
    N = int(x0.shape[0])

    return state, runner, N

def make_losses():
    pass

def main():
    pass


if __name__ == "__main__":
    main()
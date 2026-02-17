import jax.numpy as jnp

import math
from flax import struct


@struct.dataclass
class TurbineGrid:
    turbine_coordinates: jnp.ndarray  
    turbine_diameter: float
    wind_directions: jnp.ndarray      

    B: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    x_center_of_rotation: float = struct.field(pytree_node=False)
    y_center_of_rotation: float = struct.field(pytree_node=False)

    x: jnp.ndarray = struct.field(pytree_node=False)
    y: jnp.ndarray = struct.field(pytree_node=False)
    z: jnp.ndarray = struct.field(pytree_node=False)

    x_sorted: jnp.ndarray = struct.field(pytree_node=False)
    y_sorted: jnp.ndarray = struct.field(pytree_node=False)
    z_sorted: jnp.ndarray = struct.field(pytree_node=False)

    sorted_indices: jnp.ndarray = struct.field(pytree_node=False)
    sorted_coord_indices: jnp.ndarray = struct.field(pytree_node=False)
    unsorted_indices: jnp.ndarray = struct.field(pytree_node=False)

    radius: jnp.ndarray = struct.field(pytree_node=False)
    x_sorted_inertial_frame: jnp.ndarray = struct.field(pytree_node=False)
    y_sorted_inertial_frame: jnp.ndarray = struct.field(pytree_node=False)
    z_sorted_inertial_frame: jnp.ndarray = struct.field(pytree_node=False)

    grid_resolution: int = 5
    wake_rad: float =0.
    average_method = "cubic-mean"
    cubature_weights = None


    @staticmethod
    def create(
        turbine_coordinates: jnp.ndarray,
        turbine_diameter: float,
        wind_directions: jnp.ndarray,
        grid_resolution: int = 5,
        wake_rad: float =0,
    ) -> "TurbineGrid":
        B = wind_directions.shape[0]
        T = turbine_coordinates.shape[0]

        xc = 0.5 * (turbine_coordinates[:, 0].min() + turbine_coordinates[:, 0].max())
        yc = 0.5 * (turbine_coordinates[:, 1].min() + turbine_coordinates[:, 1].max())

        x_rel = turbine_coordinates[:, 0] - xc
        y_rel = turbine_coordinates[:, 1] - yc
        z_rel = turbine_coordinates[:, 2]

        theta = (wind_directions - 270.0 * math.pi / 180.0)[:, None]  # (B,1)

        DTYPE = turbine_coordinates.dtype
        x_rot = (x_rel[None, :] * jnp.cos(theta) - y_rel[None, :] * jnp.sin(theta) + xc).astype(DTYPE)
        y_rot = (x_rel[None, :] * jnp.sin(theta) + y_rel[None, :] * jnp.cos(theta) + yc).astype(DTYPE)
        z_rot = jnp.broadcast_to(z_rel[None, :], (B, T)).astype(DTYPE)

        radius = turbine_diameter * 0.5 * 0.5 
        span = jnp.linspace(-1.0, 1.0, grid_resolution, dtype=DTYPE)  

        dy = span * radius 
        dz = span * radius 
        dy_exp = jnp.broadcast_to(dy[None, None, :, None], (B, T, grid_resolution, grid_resolution)).astype(DTYPE)
        dz_exp = jnp.broadcast_to(dz[None, None, None, :], (B, T, grid_resolution, grid_resolution)).astype(DTYPE)

        y_grid = (y_rot[:, :, None, None] + dy_exp).astype(DTYPE)
        z_grid = (z_rot[:, :, None, None] + dz_exp).astype(DTYPE)

        template = jnp.ones((B, T, grid_resolution, grid_resolution), dtype=DTYPE)
        x_grid = (x_rot[:, :, None, None] * template).astype(DTYPE)

        sorted_idx = jnp.argsort(x_grid, axis=1)
        sorted_coord = jnp.argsort(x_rot, axis=1)
        unsorted_idx = jnp.argsort(sorted_idx, axis=1)

        x_sorted = jnp.take_along_axis(x_grid, sorted_idx, axis=1).astype(DTYPE)
        y_sorted = jnp.take_along_axis(y_grid, sorted_idx, axis=1).astype(DTYPE)
        z_sorted = jnp.take_along_axis(z_grid, sorted_idx, axis=1).astype(DTYPE)

        th = (wind_directions - 270.0*jnp.pi / 180.0)[:, None, None, None]
        x_off = x_sorted - xc
        y_off = y_sorted - yc
        x_i_frame = (x_off * jnp.cos(th) + y_off * jnp.sin(th) + xc).astype(DTYPE)
        y_i_frame = (-x_off * jnp.sin(th) + y_off * jnp.cos(th) + yc).astype(DTYPE)
        z_i_frame = z_sorted.astype(DTYPE)

        return TurbineGrid(
            turbine_coordinates=turbine_coordinates,
            turbine_diameter=turbine_diameter,
            wind_directions=wind_directions,
            grid_resolution=grid_resolution,
            wake_rad=wake_rad,
            B=B, T=T,
            x_center_of_rotation=xc,
            y_center_of_rotation=yc,
            x=x_rot, y=y_rot, z=z_rot,
            x_sorted=x_sorted, y_sorted=y_sorted, z_sorted=z_sorted,
            sorted_indices=sorted_idx,
            sorted_coord_indices=sorted_coord,
            unsorted_indices=unsorted_idx,
            radius=jnp.full((T,), radius),
            x_sorted_inertial_frame=x_i_frame,
            y_sorted_inertial_frame=y_i_frame,
            z_sorted_inertial_frame=z_i_frame,
        )

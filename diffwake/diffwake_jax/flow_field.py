import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional, Any


@struct.dataclass
class FlowField:
    wind_speeds: jnp.ndarray        
    wind_directions: jnp.ndarray    
    wind_shear: float
    wind_veer: float
    air_density: float
    turbulence_intensities: jnp.ndarray 
    reference_wind_height: float

    u_initial_sorted: Optional[jnp.ndarray] = None
    v_initial_sorted: Optional[jnp.ndarray] = None
    w_initial_sorted: Optional[jnp.ndarray] = None
    dudz_initial_sorted: Optional[jnp.ndarray] = None

    u_sorted:  Optional[jnp.ndarray] = None
    v_sorted:  Optional[jnp.ndarray] = None
    w_sorted:  Optional[jnp.ndarray] = None

    u: Optional[jnp.ndarray] = None
    v: Optional[jnp.ndarray] = None
    w: Optional[jnp.ndarray] = None
    dudz: Optional[jnp.ndarray] = None

    turbulence_intensity_field_sorted: Optional[jnp.ndarray] = None
    turbulence_intensity_field_sorted_avg: Optional[jnp.ndarray] = None

    turbulence_intensity_field: Optional[jnp.ndarray] = None
    grid_resolution: Optional[int] = None
    n_turbines:     Optional[int] = None
    n_findex:       Optional[int] = None
    state: str = "UNINITIALIZED"

    def initialize_velocity_field(self, grid: Any) -> "FlowField":
        """
        grid must expose:
            .z_sorted        : (B,T,Ny,Nz)
            .unsorted_indices: (B,T,Ny,Nz)  or (B,T,1,1) – same shape you used
        Returns a **new** FlowField with all *_sorted fields filled.
        """
        z_sorted = grid.z_sorted
        B, T, Ny, Nz = z_sorted.shape
        safe_z = jnp.clip(z_sorted, min=1e-7)

        wind_profile_plane = (safe_z / self.reference_wind_height) ** self.wind_shear
        dudz_profile = (
            self.wind_shear *
            (1.0 / self.reference_wind_height) ** self.wind_shear *
            safe_z ** (self.wind_shear - 1)
        )

        wind_speeds_exp = self.wind_speeds[:, None, None, None].astype(self.wind_speeds.dtype)  # (B,1,1,1)

        u_init = wind_speeds_exp * wind_profile_plane
        u_init = u_init.astype(self.wind_speeds.dtype)
        dudz   = wind_speeds_exp * dudz_profile
        dudz   = dudz.astype(self.wind_speeds.dtype)
        zeros  = jnp.zeros_like(u_init, dtype=self.wind_speeds.dtype)

        idxer  = grid.unsorted_indices
        u_uns  = jnp.take_along_axis(u_init, idxer, axis=1)
        v_uns  = jnp.take_along_axis(zeros,   idxer, axis=1)
        w_uns  = jnp.take_along_axis(zeros,   idxer, axis=1)
        dudz_uns = jnp.take_along_axis(dudz,   idxer, axis=1)
        # Expand TI field
        turb_exp = self.turbulence_intensities[:, None, None, None]  # (B,1,1,1)
        turb_exp = jnp.broadcast_to(turb_exp, (B, T, 1, 1))

        new_flow = self.replace(
            u_initial_sorted=u_init,
            v_initial_sorted=zeros,
            w_initial_sorted=zeros,
            dudz_initial_sorted=dudz,
            u_sorted=u_init,
            v_sorted=zeros,
            w_sorted=zeros,
            dudz = dudz_uns,
            u=u_uns, v=v_uns, w=w_uns,
            turbulence_intensity_field_sorted=turb_exp,
            turbulence_intensity_field=turb_exp,
            grid_resolution=Ny,
            n_turbines=T,
            n_findex=B,
            state="INITIALIZED",
        )

        return new_flow

    def finalize(self, unsorted_indices: jnp.ndarray) -> "FlowField":
        idx = unsorted_indices
        u_uns = jnp.take_along_axis(self.u_sorted, idx, axis=1)
        v_uns = jnp.take_along_axis(self.v_sorted, idx, axis=1)
        w_uns = jnp.take_along_axis(self.w_sorted, idx, axis=1)

        ti_uns = jnp.take_along_axis(
            self.turbulence_intensity_field_sorted, idx, axis=1
        )
        ti_field = jnp.mean(ti_uns, axis=(2, 3))

        return self.replace(u=u_uns, v=v_uns, w=w_uns,
                            turbulence_intensity_field=ti_field,
                            state="FINALIZED")

    def to_device(self, device) -> "FlowField":
        """
        device: 'cpu', 'gpu', 'tpu', or jax.Device
        Returns a copy with every ndarray placed on *device*.
        """
        def _move(x):
            return jax.device_put(x, device) if isinstance(x, jnp.ndarray) else x
        return self.map(_move)             # flax.struct dataclass utility

    @property
    def coordinates(self) -> jnp.ndarray:
        return jnp.stack(
            [self.layout_x, self.layout_y,
             jnp.full_like(self.layout_x, self.reference_wind_height)],
            axis=-1
        )

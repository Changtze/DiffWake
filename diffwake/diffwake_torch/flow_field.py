import torch


class FlowField:
    def __init__(
        self,
        wind_speeds: torch.Tensor,            # [B]
        wind_directions: torch.Tensor,        # [B]
        wind_shear: float,
        wind_veer: float,
        air_density: float,
        turbulence_intensities: torch.Tensor, # [B]
        reference_wind_height: float,
    ):
        assert wind_speeds.ndim == 1
        assert wind_directions.ndim == 1
        assert turbulence_intensities.ndim == 1
        assert wind_speeds.shape == wind_directions.shape == turbulence_intensities.shape

        self.wind_speeds = wind_speeds
        self.wind_directions = wind_directions
        self.wind_shear = wind_shear
        self.wind_veer = wind_veer
        self.air_density = air_density
        self.turbulence_intensities = turbulence_intensities
        self.reference_wind_height = reference_wind_height
        self.n_findex = wind_speeds.shape[0]

        # Runtime-initialized fields
        self.u_initial_sorted = None
        self.v_initial_sorted = None
        self.w_initial_sorted = None
        self.dudz_initial_sorted = None

        self.u_sorted = None
        self.v_sorted = None
        self.w_sorted = None

        self.u = None
        self.v = None
        self.w = None

        self.turbulence_intensity_field = None
        self.turbulence_intensity_field_sorted = None
        self.turbulence_intensity_field_sorted_avg = None

    def initialize_velocity_field(self, grid):
        """
        grid: a TurbineGrid object with attributes
              .z_sorted of shape [B, T, Ny, Nz]
        """
        z_sorted = grid.z_sorted
        B, T, Ny, Nz = z_sorted.shape

        self.grid_resolution = Ny  # Assumes square rotor grid
        self.n_turbines = T

        safe_z = z_sorted.clamp(min=1e-6)

        # Velocity profile using shear law
        wind_profile_plane = (safe_z / self.reference_wind_height) ** self.wind_shear
        dudz_profile = (
            self.wind_shear *
            (1.0 / self.reference_wind_height) ** self.wind_shear *
            safe_z.pow(self.wind_shear - 1)
        )

        # Expand wind speeds
        wind_speeds_exp = self.wind_speeds.view(-1, 1, 1, 1)

        self.u_initial_sorted = wind_speeds_exp * wind_profile_plane
        self.dudz_initial_sorted = wind_speeds_exp * dudz_profile

        self.v_initial_sorted = torch.zeros_like(self.u_initial_sorted)
        self.w_initial_sorted = torch.zeros_like(self.u_initial_sorted)

        self.u_sorted = self.u_initial_sorted.clone()
        self.v_sorted = self.v_initial_sorted.clone()
        self.w_sorted = self.w_initial_sorted.clone()

        indexer = grid.unsorted_indices

        self.u = torch.gather(self.u_sorted, dim=1, index=indexer).clone()
        self.v = torch.gather(self.v_sorted, dim=1, index=indexer).clone()
        self.w = torch.gather(self.w_sorted, dim=1, index=indexer).clone()

        # Expand TI to match [B, T, Ny, Nz]
        turb_exp = self.turbulence_intensities[:, None, None, None]  # [B, 1, 1, 1]
        turb_exp = turb_exp.expand(B, T, 1, 1)                     # [B, T, Ny, Nz]

        self.turbulence_intensity_field_sorted = turb_exp.clone()
        self.turbulence_intensity_field = turb_exp.clone()

    def finalize(self, unsorted_indices: torch.Tensor):
        """
        unsorted_indices: [B, T] - indices to reorder turbines back to original order.
        """
        indexer = unsorted_indices
        self.u = torch.gather(self.u_sorted, dim=1, index=indexer)
        self.v = torch.gather(self.v_sorted, dim=1, index=indexer)
        self.w = torch.gather(self.w_sorted, dim=1, index=indexer)

        ti_unsorted = torch.gather(
            self.turbulence_intensity_field_sorted, dim=1, index=indexer
        )
        self.turbulence_intensity_field = ti_unsorted.mean(dim=(2, 3))
    def to(self, device: torch.device | str):
        """Move **every** tensor attribute (recursively) to *device*."""
        device = torch.device(device)
        for name, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, name, val.to(device, non_blocking=True))
        self.device = device
        return self 
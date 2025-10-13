import torch
from .utils import wake_levels_from_mask, wake_mask_all, mask_non_influencers
import math

class TurbineGrid:
    def __init__(self, turbine_coordinates, turbine_diameter, wind_directions, grid_resolution=5, wake_rad = 0.):
        self.turbine_coordinates = turbine_coordinates  # [T, 3]
        self.turbine_diameter = turbine_diameter        # scalar
        self.wind_directions = wind_directions          # [B]
        self.grid_resolution = grid_resolution

        self.B = wind_directions.shape[0]
        self.T = turbine_coordinates.shape[0]
        self.device = turbine_coordinates.device

        self.average_method = "cubic-mean"
        self.cubature_weights = None
        self.wake_rad = wake_rad

        self.set_grid()

    def rotate_coordinates(self, wind_directions, coordinates):
        B, T = wind_directions.shape[0], coordinates.shape[0]

        x_center = 0.5 * (coordinates[:, 0].min() + coordinates[:, 0].max())
        y_center = 0.5 * (coordinates[:, 1].min() + coordinates[:, 1].max())

        x = coordinates[:, 0] - x_center
        y = coordinates[:, 1] - y_center
        z = coordinates[:, 2]

        theta = (wind_directions - 270.0*math.pi/180)[:, None]  # (B,1)

        x = x.view(1, T).expand(B, T)
        y = y.view(1, T).expand(B, T)
        z = z.view(1, T).expand(B, T)

        x_rot = x * torch.cos(theta) - y * torch.sin(theta) + x_center
        y_rot = x * torch.sin(theta) + y * torch.cos(theta) + y_center
        z_rot = z

        return x_rot, y_rot, z_rot, x_center, y_center

    def reverse_rotate_coordinates_rel_west(self, wind_directions, grid_x, grid_y, grid_z, x_center, y_center):
        B = wind_directions.shape[0]
        theta = (wind_directions - 270.0*math.pi/180).view(B, *[1]*(grid_x.ndim - 1))
        x_off = grid_x - x_center
        y_off = grid_y - y_center


        x_rev = x_off * torch.cos(theta) + y_off * torch.sin(theta) + x_center
        y_rev = -x_off * torch.sin(theta) + y_off * torch.cos(theta) + y_center
        return x_rev, y_rev, grid_z
    
    def set_wake_levels(self):
        x_ = self.x_sorted[:,:,1,1]
        y_ = self.y_sorted[:,:,1,1]
        rad = (self.wake_rad/2)
        mask = wake_mask_all(x_, y_,rad, L= self.turbine_diameter*1.1*20,d= self.turbine_diameter*1.1)
        self.wake_levels_sorted = wake_levels_from_mask(mask)
        self.non_influencers_sorted = mask_non_influencers(mask)

    def set_grid(self):
        # Step 1: rotate turbine coordinates
        x, y, z, x_center, y_center = self.rotate_coordinates(self.wind_directions, self.turbine_coordinates)
        self.x_center_of_rotation = x_center
        self.y_center_of_rotation = y_center

        # Step 2: build local rotor grid
        radius_ratio = 0.5
        radius = self.turbine_diameter * radius_ratio / 2.0
        radius_tensor = torch.full((self.T,), radius, device=self.device)

        span = torch.linspace(-1.0, 1.0, self.grid_resolution, device=self.device)  # [R]
        dy = span.view(1, -1) * radius_tensor.view(-1, 1)  # [T, R]
        dz = span.view(1, -1) * radius_tensor.view(-1, 1)  # [T, R]
        dy_exp = dy[None, :, :, None].expand(self.B, self.T, self.grid_resolution, self.grid_resolution)  # y varies along axis -2
        dz_exp = dz[None, :, None, :].expand(self.B, self.T, self.grid_resolution, self.grid_resolution)  # z varies along axis -1

        y_grid = y[:, :, None, None] + dy_exp  # [B, T, R, R]
        z_grid = z[:, :, None, None] + dz_exp  # [B, T, R, R]solution)d


        # Step 1: Create template grid
        template_grid = torch.ones((self.B, self.T, self.grid_resolution, self.grid_resolution), device=self.device)

        # Step 2: Create full x grid
        _x = x[:, :, None, None] * template_grid  # [B, T, R, R]

        # Step 3: Sort turbines by their center x-location
        self.sorted_indices = torch.argsort(_x, dim=1)
        self.sorted_coord_indices = torch.argsort(x, dim=1)
        self.unsorted_indices = torch.argsort(self.sorted_indices, dim=1)

        # Step 4: Gather grid values
        self.x_sorted = torch.gather(_x, 1, self.sorted_indices)  # [B, T, R, R]
        self.y_sorted = torch.gather(y_grid, 1, self.sorted_indices)  # [B, T, R, R]
        self.z_sorted = torch.gather(z_grid, 1, self.sorted_indices)  # [B, T, R, R]

        self.x = x
        self.y = y
        self.z = z
        self.radius = radius_tensor

        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            self.reverse_rotate_coordinates_rel_west(
                self.wind_directions,
                self.x_sorted,
                self.y_sorted,
                self.z_sorted,
                self.x_center_of_rotation,
                self.y_center_of_rotation
            )
        self.set_wake_levels()
    def to(self, device: torch.device | str):
        """Move **every** tensor attribute (recursively) to *device*."""
        device = torch.device(device)
        for name, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, name, val.to(device, non_blocking=True))
        self.device = device
        return self 
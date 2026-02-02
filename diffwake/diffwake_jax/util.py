import yaml
import jax.numpy as jnp
import os
from flax import struct
from .farm import Farm
from .grid import TurbineGrid
from .flow_field import FlowField
from .wake import WakeModelManager
from typing import Callable, Any, Optional,NamedTuple,Dict
from .turbine.turbine import thrust_coefficient, axial_induction
from jax import lax, config
from jax.tree_util import tree_map
from .interp1d import interp1d

# ---------- helpers --------------------------------------------------------
def _to_jax(pytree):
    """Convert any lingering torch.Tensor to jnp.ndarray."""
    return tree_map(lambda x: jnp.asarray(x) if hasattr(x, "detach") else x, pytree)

def index_dtype():
    return jnp.int64 if config.x64_enabled else jnp.int32

@struct.dataclass
class GCHConfig:
    generator: Dict
    farm: Dict
    flow_field: Dict
    layout: Dict
    def set(self, **kwargs):
        return set_cfg(self, **kwargs)

@struct.dataclass
class GCHSTate:
    farm: Farm
    grid: TurbineGrid
    flow: FlowField
    wake: WakeModelManager

@struct.dataclass
class CCConfig:
    generator: Dict
    farm: Dict
    flow_field: Dict
    layout: Dict
    def set(self, **kwargs):
        return set_cfg(self, **kwargs)

@struct.dataclass
class CCState:
    farm: Farm            # already immutable
    grid: TurbineGrid     # immutable
    flow: FlowField       # uses .replace() for updates
    wake: WakeModelManager


def average_velocity_jax(v, method="cubic-mean"):
    if method == "simple-mean":
        return jnp.mean(v, axis=(-2, -1), keepdims=True)
    if method == "cubic-mean":
        m3 = jnp.mean(v**3, axis=(-2, -1), keepdims=True)
        return jnp.cbrt(m3)                 # exact libm cbrt, matches Torch
    raise ValueError

def make_rotor_masks_old(x_coord, y_coord, x_c, y_c, D, tol=0.01):
    """
    Returns a Boolean mask of shape (B, T, Ny, Nz) that is True on the
    rotor disc of each turbine. Compute once in __init__ and later use
    rotor_masks[:, i] inside loops.
    """
    B, T, Ny, Nz = x_coord.shape
    # Broadcast centers
    xc = jnp.broadcast_to(x_c, (B, T, Ny, Nz))
    yc = jnp.broadcast_to(y_c, (B, T, Ny, Nz))
    rad = 0.51 * D  # scalar or (1, 1, 1, 1)

    rotor_mask = (
        (x_coord > xc - tol) & (x_coord < xc + tol) &
        (y_coord > yc - rad) & (y_coord < yc + rad)
    )
    return rotor_mask

def make_rotor_masks(x_coord, y_coord, x_c, y_c, D, tol=0.01):
    B, T, Ny, Nz = x_coord.shape
    rad = 0.51 * D

    # (B, T, 1, 1, 1)  vs  (B, 1, T, Ny, Nz)
    xi = x_c[:, :, None, None, None]
    yi = y_c[:, :, None, None, None]
    xg = x_coord[:, None, :, :, :]
    yg = y_coord[:, None, :, :, :]

    mask = ((jnp.abs(xg - xi) < tol) &
            (jnp.abs(yg - yi) < rad))
    # mask[b, i, j, :, :] = rotor-mask för turbin i på grid j
    return mask            # shape (B, T, T, Ny, Nz)

def smooth_step(x, edge, width=1.0):
    return 1.0 / (1.0 + jnp.exp(-(x - edge) / width))

def smooth_box(x, centre, half, inv_w=2.0):
    s1 = lax.sigmoid((x - (centre - half)) * inv_w)
    s2 = lax.sigmoid((x - (centre + half)) * inv_w)
    return s1 * (1.0 - s2)

@struct.dataclass
class GCHParams:
    B: int
    T: int
    rotor_diameter: float
    hub_height: float
    TSR: float
    wind_shear: float
    wind_veer: float
    gr_square: float  # Ny * Nz
    enable_secondary_steering: bool = False
    enable_transverse_velocities: bool = False
    enable_yaw_added_recovery: bool = False

class GCHResult:
    turb_u_wake: jnp.ndarray
    u_sorted: jnp.ndarray

@struct.dataclass
class CCParams:
    # Grid/turbine constants
    B: int
    T: int
    rotor_diameter: float
    hub_height: float
    TSR: float
    wind_shear: float
    wind_veer: float
    gr_square: float  # Ny * Nz
    enable_secondary_steering: bool = False
    enable_transverse_velocities: bool = False
    enable_yaw_added_recovery: bool = False

class CCResult(NamedTuple):
    turb_u_wake: jnp.ndarray
    u_sorted: jnp.ndarray
    ti: jnp.ndarray
    v_sorted: jnp.ndarray
    w_sorted: jnp.ndarray

@struct.dataclass
class CCDynamicState:
    turb_u_wake: jnp.ndarray     # (B,T,Ny,Nz)
    turb_inflow: jnp.ndarray     # (B,T,Ny,Nz)
    ti:          jnp.ndarray     # (B,T,1,1)
    v_sorted:      jnp.ndarray
    w_sorted:      jnp.ndarray
    Ctmp:        jnp.ndarray     # (T,B,T,Ny,Nz)
    ct_acc: jnp.ndarray         # (B, T, 1, 1)
    
@struct.dataclass
class Thrust:
    ti: jnp.ndarray
    rho: float

    thrust_fn: Callable= struct.field(pytree_node=False)
    tilt_interp: Callable= struct.field(pytree_node=False)
    correct_cp_ct_for_tilt: bool
    power_thrust_table: dict = struct.field(pytree_node=False)

    def __call__(self, velocities: jnp.ndarray,
                 yaw_angles: jnp.ndarray,
                 tilt_angles: jnp.ndarray) -> jnp.ndarray:
        
        return thrust_coefficient(
            velocities=velocities,
            yaw_angles=yaw_angles,
            tilt_angles=tilt_angles,
            thrust_fn=self.thrust_fn,
            tilt_interp=self.tilt_interp,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
            power_thrust_table=self.power_thrust_table,
        )

@struct.dataclass
class AxialInduction:
    ti: jnp.ndarray
    rho: float
    pset: jnp.ndarray

    axial_induction_function: Callable= struct.field(pytree_node=False)
    tilt_interp: Callable= struct.field(pytree_node=False)
    correct_cp_ct_for_tilt: bool

    power_thrust_table: dict = struct.field(pytree_node=False)
    cubature_weights: Optional[jnp.ndarray]
    multidim_condition: Optional[Any] = None  # can be jnp.ndarray or None

    def __call__(self, velocities: jnp.ndarray, 
                 yaw_angles: jnp.ndarray, 
                 tilt_angles: jnp.ndarray, 
                 ix_filter: Optional[int] = None) -> jnp.ndarray:

        return axial_induction(
            velocities=velocities,
            ix_filter=ix_filter,
            yaw_angles=yaw_angles,
            tilt_angles=tilt_angles,
            axial_induction_function=self.axial_induction_function,
            tilt_interp=self.tilt_interp,
            correct_cp_ct_for_tilt=self.correct_cp_ct_for_tilt,
            turbine_power_thrust_table=self.power_thrust_table,
            cubature_weights=self.cubature_weights,
            multidim_condition=self.multidim_condition,
        )
    


def init_dynamic_state(grid, flow) -> CCState:
    B, T, Ny, Nz = grid.x_sorted.shape
    zeros = jnp.zeros_like(_to_jax(grid.x_sorted))
    ti = jnp.broadcast_to(flow.turbulence_intensities[:, None, None, None], (B, T, 3, 3))
    return CCDynamicState(
        turb_u_wake = zeros.copy(),
        turb_inflow = _to_jax(flow.u_initial_sorted).copy(),
        ti          = ti.copy(),
        v_sorted      = zeros.copy(),
        w_sorted      = zeros.copy(),
        Ctmp        = jnp.zeros((T, B, T, Ny, Nz), zeros.dtype),
        ct_acc=jnp.zeros((B, T, 1, 1), zeros.dtype),  # <— running CTs
    )

def get_axial_induction_fn(flow, farm, grid):

    return AxialInduction(
        ti      = _to_jax(flow.turbulence_intensity_field_sorted),
        rho     = flow.air_density,
        pset    = _to_jax(farm.power_setpoints_sorted),
        axial_induction_function = farm.axial_induction_function,
        tilt_interp              = farm.tilt_interp,
        correct_cp_ct_for_tilt   = farm.correct_cp_ct_for_tilt,
        power_thrust_table       = farm.power_thrust_table,
        cubature_weights         = grid.cubature_weights,
        multidim_condition       = None,
    )


def get_thrust_fn(flow, farm):
    return Thrust(
        ti      = _to_jax(flow.turbulence_intensity_field_sorted),
        rho     = flow.air_density,
        thrust_fn              = farm.turbine.thrust_coefficient_function,
        tilt_interp            = farm.tilt_interp,
        correct_cp_ct_for_tilt = farm.correct_cp_ct_for_tilt,
        power_thrust_table     = farm.power_thrust_table,
    )



def make_constants(state: CCState):
    g   = _to_jax(state.grid)
    fld = _to_jax(state.flow)
    farm = state.farm

    x, y, z = g.x_sorted, g.y_sorted, g.z_sorted
    x_c = jnp.mean(x, axis=(2,3), keepdims=True)
    y_c = jnp.mean(y, axis=(2,3), keepdims=True)
    z_c = jnp.mean(z, axis=(2,3), keepdims=True)

    const = dict(
        x_coord = x,  y_coord = y,  z_coord = z,
        x_c = x_c,    y_c = y_c,    z_c = z_c,
        u_init  = fld.u_initial_sorted.copy(),
        dudz_init = fld.dudz_initial_sorted.copy(),
        ambient_ti = fld.turbulence_intensities[:, None, None, None].copy(),
    )

    return const, _to_jax(farm.yaw_angles_sorted), _to_jax(farm.tilt_angles_sorted)

def make_params(state: CCState) -> CCParams:
    B, T, _, _ = state.grid.x_sorted.shape

    params = CCParams(
        B = B, T = T,
        rotor_diameter = state.farm.rotor_diameter,
        hub_height     = state.farm.hub_height,
        TSR            = state.farm.TSR,
        wind_shear     = state.flow.wind_shear,
        wind_veer      = state.flow.wind_veer,
        gr_square      = state.grid.grid_resolution ** 2)


    return (params, 
        state.wake.enable_secondary_steering,
        state.wake.enable_transverse_velocities,
        state.wake.enable_yaw_added_recovery)


def to_result(st: CCState) -> CCResult:
    return CCResult(                # keep field order explicit
        turb_u_wake = st.turb_u_wake,
        u_sorted = st.turb_inflow,
        ti          = st.ti,
        v_sorted      = st.v_sorted,
        w_sorted      = st.w_sorted,
    )

# --- Loader definition ---
class LoaderWithInclude(yaml.SafeLoader):
    def __init__(self, stream):
        self.name = stream.name  # required by _include()
        super().__init__(stream)

# --- !include handler ---
def _include(loader, node):
    rel_path = loader.construct_scalar(node)
    base_dir = os.path.dirname(loader.name)
    full_path = os.path.join(base_dir, rel_path)
    with open(full_path, "r") as inc:
        return yaml.load(inc, LoaderWithInclude)

# REGISTER after defining the class
LoaderWithInclude.add_constructor("!include", _include)

# --- YAML loader with conversion ---
def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, LoaderWithInclude)

    def _to_jax(obj):
        if isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
            return jnp.array(obj)
        if isinstance(obj, dict):
            return {k: _to_jax(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jax(x) for x in obj]
        return obj

    return _to_jax(data)



def set_cfg(cfg,
    wind_speeds=None,
    wind_directions=None,
    turbulence_intensities=None,
    layout_x=None,
    layout_y=None,
    yaw_angles=None,
):
    # Make shallow copies of the dicts
    flow_field = cfg.flow_field.copy()
    layout     = cfg.layout.copy()
    farm       = cfg.farm.copy()

    if wind_speeds is not None:
        flow_field["wind_speeds"] = wind_speeds
    if wind_directions is not None:
        flow_field["wind_directions"] = wind_directions
    if turbulence_intensities is not None:
        flow_field["turbulence_intensities"] = turbulence_intensities

    if layout_x is not None:
        layout["layout_x"] = layout_x
    if layout_y is not None:
        layout["layout_y"] = layout_y

    if yaw_angles is not None:
        farm["yaw_angles"] = yaw_angles

    # Return a new CCConfig with updated dicts
    return CCConfig(
        generator=cfg.generator,  # unchanged
        farm=farm,
        flow_field=flow_field,
        layout=layout,
    )


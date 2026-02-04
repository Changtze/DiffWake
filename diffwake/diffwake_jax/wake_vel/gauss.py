import jax
import jax.numpy as jnp
from flax import struct                 # Lightweight immutable dataclass
from jax import lax


def cosd(angle):
    return jnp.cos(jnp.radians(angle))

def sind(angle):
    return jnp.sin(jnp.radians(angle))

def tand(angle):
    return jnp.tan(jnp.radians(angle))

def gaussian_function(C, r, n, sigma):
    return C * jnp.exp(-1 * r ** n / (2 * sigma ** 2))

def rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D):
    wind_veer = jnp.radians(wind_veer)
    
    a = jnp.cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + jnp.sin(wind_veer) ** 2 / (2 * sigma_z ** 2)
    b = -jnp.sin(2 * wind_veer) / (4 * sigma_y ** 2) + jnp.sin(2 * wind_veer) / (4 * sigma_z ** 2)
    c = jnp.sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + jnp.cos(wind_veer) ** 2 / (2 * sigma_z ** 2)
    
    r = a * ((y - y_i - delta) ** 2) - 2 * b * (y - y_i - delta) * (z - HH) + c * ((z - HH) ** 2)
    d = jnp.clip(1 - (Ct * cosd(yaw) / ( 8.0 * sigma_y * sigma_z / (D * D) )), 0.0, 1.0)
    C = 1 - jnp.sqrt(d)
    return r, C


def safe_sqrt(x: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Numerically-safe sqrt (same logic as PyTorch clamp-then-sqrt)."""
    return jnp.sqrt(jnp.clip(x, min=eps))


def wake_expansion(delta_x, ct, ti, D, a_s, b_s, c_s1, c_s2):
    beta   = 0.5 * (1.0 + safe_sqrt(1.0 - ct)) / safe_sqrt(1.0 - ct + 1e-8)
    k      = a_s * ti + b_s
    eps    = (c_s1 * ct + c_s2) * safe_sqrt(beta)
    x_tilde = jnp.abs(delta_x) / D
    sigma_y = k * x_tilde + eps
    return sigma_y

@struct.dataclass
class GaussVelocityDeficit:
    alpha: float = 0.58
    beta: float = 0.077
    ka: float = 0.38
    kb: float = 0.004

    def __call__(
        self,
        x_i: jnp.ndarray,
        y_i: jnp.ndarray,
        z_i: jnp.ndarray,
        axial_induction_i: jnp.ndarray,
        deflection_field_i: jnp.ndarray,
        yaw_angle_i: jnp.ndarray,
        turbulence_intensity_i: jnp.ndarray,
        ct_i: jnp.ndarray,
        hub_height_i: float,
        rotor_diameter_i: float,
        *,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
        u_initial: jnp.ndarray,
        wind_veer: float,
    ) -> jnp.ndarray:

        # Opposite sign convention in this model
        yaw_angle = -1 * yaw_angle_i

        # Initialize the velocity deficit
        uR = u_initial * ct_i / (2.0 * (1 - jnp.sqrt(1 - ct_i)))
        u0 = u_initial * jnp.sqrt(1 - ct_i)

        # Initial lateral bounds
        sigma_z0 = rotor_diameter_i * 0.5 * jnp.sqrt(uR / (u_initial + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(wind_veer)

        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = jnp.ones_like(u_initial)
        x0 *= rotor_diameter_i * cosd(yaw_angle) * (1 + jnp.sqrt(1 - ct_i) )
        x0 /= jnp.sqrt(2) * (
            4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - jnp.sqrt(1 - ct_i) )
        )
        x0 += x_i

        # Initialise velocity deficit array
        velocity_deficit = jnp.zeros_like(u_initial)

        # Masks
        # When there is an inequality, the current turbine may be applied in its own wake where
        # numerical precision causes an incorrect comparison.
        # Apply a small numerical bump to avoid this. "0.1" is arbitrary but it is still a small,
        # non-zero value.

        # This mask defines the near wake, keeps the areas downstream of xR and upstream x0)
        near_wake_mask = (x > xR + 0.1) * (x < x0)
        far_wake_mask = (x >= x0)

        # NEAR WAKE region
        if jnp.sum(near_wake_mask):
            near_wake_ramp_up = (x - xR) / (x0 - xR)
            near_wake_ramp_down = (x0 - x) / (x0 - xR)

            sigma_y_near = near_wake_ramp_down * 0.501 * rotor_diameter_i * jnp.sqrt(ct_i / 2.0)
            sigma_y_near += near_wake_ramp_up * sigma_y0
            sigma_y_near = jnp.where(x >= xR, sigma_y_near, 0.5 * rotor_diameter_i)

            sigma_z_near = near_wake_ramp_down * 0.501 * rotor_diameter_i * jnp.sqrt(ct_i / 2.0)
            sigma_z_near += near_wake_ramp_up * sigma_z0
            sigma_z_near = jnp.where(x >= xR, sigma_z_near, 0.5 * rotor_diameter_i)

            r_near, C_near = rC(
                wind_veer,
                sigma_y_near,
                sigma_z_near,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i,
            )

            near_wake_deficit = gaussian_function(C_near, r_near, 1, jnp.sqrt(0.5))
            near_wake_deficit *= near_wake_mask

            velocity_deficit += near_wake_deficit

        # FAR WAKE region
        # Wake expansion in the lateral (y) and the vertical (z)
        if jnp.sum(far_wake_mask):
            ky = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            kz = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            sigma_y_far = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * (x < x0)
            sigma_z_far = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * (x < x0)

            r_far, C_far = rC(
                wind_veer,
                sigma_y_far,
                sigma_z_far,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i,
            )

            far_wake_deficit = gaussian_function(C_far, r_far, 1, jnp.sqrt(0.5))
            far_wake_deficit *= far_wake_mask

            velocity_deficit += far_wake_mask

        return velocity_deficit



import jax.numpy as jnp
import math
from flax import struct


@struct.dataclass
class GaussVelocityDeflection:
    ad : float = 0.0;  bd : float = 0.0
    alpha: float = 0.58; beta: float = 0.077
    ka : float = 0.38;  kb : float = 0.004
    dm : float = 1.0
    eps_gain: float = 0.2
    use_secondary_steering: bool = True
    _sqrt2: float = math.sqrt(2.0)

    # --------------------------------------------------------------------
    def __call__(self,
        x_i: jnp.ndarray,
        yaw_i: jnp.ndarray,
        turb_I_i: jnp.ndarray,
        ct_i: jnp.ndarray,
        D: jnp.ndarray,
        x: jnp.ndarray, 
        U_free: jnp.ndarray,
        wind_veer: float
    ) -> jnp.ndarray:

        dtype = x.dtype                   
        ad     = jnp.asarray(self.ad,   dtype)
        bd     = jnp.asarray(self.bd,   dtype)
        sqrt2  = jnp.asarray(self._sqrt2, dtype)
        c16    = jnp.asarray(1.6 ,  dtype)
        f52    = jnp.asarray(5.2 ,  dtype)
        k03    = jnp.asarray(0.3 ,  dtype)
        exp_1_12 = jnp.exp(jnp.asarray(1/12, dtype))
        exp_1_3  = jnp.exp(jnp.asarray(1/3 , dtype))
        veer   = jnp.rad2deg(jnp.asarray(wind_veer, dtype))

        yaw   = -yaw_i
        cos_y = jnp.cos(yaw)
        one   = jnp.ones_like(cos_y)

        uR = U_free * ct_i * cos_y / (2. * (one - jnp.sqrt(one - ct_i * cos_y)))
        u0 = U_free * jnp.sqrt(one - ct_i)

        denom = sqrt2 * (4 * self.alpha * turb_I_i +
                         2 * self.beta  * (one - jnp.sqrt(one - ct_i)))
        x0 = D * cos_y * (1 + jnp.sqrt(one - ct_i * cos_y)) / denom + x_i

        k = self.ka * turb_I_i + self.kb

        sigma_z0 = D * 0.5 * jnp.sqrt(uR / (U_free + u0))
        sigma_y0 = sigma_z0 * cos_y * jnp.cos(veer)

        theta_c0 = self.dm * k03 * yaw / cos_y
        theta_c0 *= (1 - jnp.sqrt(one - ct_i * cos_y))
        delta0   = jnp.tan(theta_c0) * (x0 - x_i)

        mask_near = (x >= x_i) & (x <= x0)
        mask_far  = x > x0

        delta_near = ((x - x_i) / (x0 - x_i)) * delta0 + (ad + bd * (x - x_i))
        delta_near = jnp.where(mask_near, delta_near, jnp.zeros_like(delta_near))

        sigma_y = jnp.where(mask_far, k * (x - x0) + sigma_y0, sigma_y0)
        sigma_z = jnp.where(mask_far, k * (x - x0) + sigma_z0, sigma_z0)

        C0   = 1 - u0 / U_free
        M0   = C0 * (2 - C0)
        E0   = C0**2 - 3 * exp_1_12 * C0 + 3 * exp_1_3
        M0_s = jnp.sqrt(M0)

        mid    = jnp.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0))
        ln_num = (c16 + M0_s) * (c16 * mid - M0_s)
        ln_den = (c16 - M0_s) * (c16 * mid + M0_s)
        log_t  = jnp.log(ln_num / ln_den)

        mult = theta_c0 * E0 / f52 * jnp.sqrt(
            sigma_y0 * sigma_z0 / (k * k * M0)
        )

        delta_far = delta0 + mult * log_t + (ad + bd * (x - x_i))
        delta_far = jnp.where(mask_far, delta_far, jnp.zeros_like(delta_far))

        return delta_near + delta_far


def gamma(D, velocity, Uinf, Ct, scale=1.0):
    return scale * (jnp.pi / 8) * D * velocity * Uinf * Ct

NUM_EPS = 0.001
NUM_EPS_2 = 1e-7
EPS_GAIN = 0.2

def gamma(D: float,
          velocity: jnp.ndarray,
          Uinf:    jnp.ndarray,
          Ct:      jnp.ndarray,
          scale: float = 1.0) -> jnp.ndarray:
    return scale * (jnp.pi / 8.0) * D * velocity * Uinf * Ct


def cbrt(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)


# ───────────────────────────────────────── main ───────────────────────────────
def calculate_transverse_velocity(
    u_i: jnp.ndarray,                # (B,1,Ny,Nz)
    u_initial: jnp.ndarray,          # (B,1,Ny,Nz)
    dudz_initial: jnp.ndarray,       # (B,1,Ny,Nz)
    delta_x: jnp.ndarray,            # (B,1,Ny,Nz)
    delta_y: jnp.ndarray,            # (B,1,Ny,Nz)
    z: jnp.ndarray,                  # (B,1,Ny,Nz)
    rotor_diameter: float,
    hub_height: float,
    yaw: jnp.ndarray,                # (B,1,1,1) 
    ct_i: jnp.ndarray,               # (B,1,1,1)
    tsr_i: jnp.ndarray,              # (B,1,1,1)
    axial_induction_i: jnp.ndarray,  # (B,1,1,1)
    wind_shear: float,
    scale: float = 1.0,
):
    dtype = u_i.dtype
    NUM_EPS  = jnp.asarray(1e-10, dtype)
    EPS_GAIN = jnp.asarray(0.2,   dtype)
    D   = jnp.asarray(rotor_diameter, dtype)
    HH  = jnp.asarray(hub_height,     dtype)
    TSR = tsr_i                       # redan tensor
    Ct  = ct_i
    aI  = axial_induction_i

    Uinf = jnp.mean(u_initial, axis=(1, 2, 3), keepdims=True)        # (B,1,1,1)
    eps  = EPS_GAIN * D
    ones = jnp.ones_like(Ct)                                         # (B,1,1,1)

    s_c = jnp.sin(yaw) * jnp.cos(yaw)
    vel_top    = ((HH + 0.5 * D) / HH) ** wind_shear * ones
    vel_bottom = ((HH - 0.5 * D) / HH) ** wind_shear * ones

    Gamma_top    =  s_c * gamma(D, vel_top,    Uinf, Ct, scale)
    Gamma_bottom = -s_c * gamma(D, vel_bottom, Uinf, Ct, scale)

    mean_cubed    = jnp.mean(u_i ** 3, axis=(2, 3), keepdims=True)
    turbine_avg_u = cbrt(mean_cubed)
    Gamma_core = 0.5 * jnp.pi * D * (aI - aI ** 2) * turbine_avg_u / TSR

    lmda  = D / 8.0
    kappa = 0.41
    lm = kappa * z / (1.0 + kappa * z / lmda)
    nu = lm**2 * jnp.abs(dudz_initial)

    decay = eps**2 / (4.0 * nu * delta_x / Uinf + eps**2)
    y = delta_y + NUM_EPS

    def vortex(Gamma: jnp.ndarray, z_shift: float):
        z_  = z - z_shift + NUM_EPS
        r2  = y**2 + z_**2
        core = 1.0 - jnp.exp(-r2 / eps**2)
        V =  (Gamma * z_)  / (2.0 * jnp.pi * r2) * core * decay
        W = (-Gamma * y)   / (2.0 * jnp.pi * r2) * core * decay
        return V, W

    V1, W1 = vortex( Gamma_top,            HH + 0.5 * D)
    V2, W2 = vortex( Gamma_bottom,         HH - 0.5 * D)
    V5, W5 = vortex( Gamma_core,           HH)

    V3, W3 = vortex(-Gamma_top,          -(HH + 0.5 * D))
    V4, W4 = vortex(-Gamma_bottom,       -(HH - 0.5 * D))
    V6, W6 = vortex(-Gamma_core,         -HH)

    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6

    V = jnp.where(delta_x + NUM_EPS_2 >= 0.0, V, jnp.zeros_like(V))
    W = jnp.where((delta_x + NUM_EPS_2 >= 0.0) & (W + NUM_EPS_2 >= 0.0),
                  W, jnp.zeros_like(W))

    return V, W

def yaw_added_turbulence_mixing(u_i, I_i, v_i, w_i, turb_v_i, turb_w_i):
    I_i = I_i[:, 0, 0, 0]

    average_u_i = jnp.sign(u_i) * jnp.abs(u_i)**3
    average_u_i = jnp.mean(average_u_i, axis=(1, 2, 3))**(1 / 3)

    k = (average_u_i * I_i) ** 2 / (2.0 / 3.0)
    u_term = jnp.sqrt(2.0 * k)
    v_term = jnp.mean(v_i + turb_v_i, axis=(1, 2, 3))
    w_term = jnp.mean(w_i + turb_w_i, axis=(1, 2, 3))

    k_total = 0.5 * (u_term**2 + v_term**2 + w_term**2)
    I_total = jnp.sqrt((2.0 / 3.0) * k_total) / average_u_i

    I_mixing = I_total - I_i

    return I_mixing[:, None, None, None]

def gamma(D, velocity, Uinf, Ct, scale=1.0):
    return scale * (jnp.pi / 8) * D * velocity * Uinf * Ct

def wake_added_yaw(
    u_i, v_i, u_initial, delta_y, z_i,
    rotor_diameter, hub_height, ct_i,
    tip_speed_ratio, axial_induction_i, wind_shear, scale=1.0
):
    NUM_EPS = 0.001
    D = rotor_diameter
    HH = hub_height
    Ct = ct_i
    TSR = tip_speed_ratio
    aI = axial_induction_i

    avg_v = jnp.mean(v_i, axis=(2, 3))
    Uinf = jnp.mean(u_initial, axis=(1, 2, 3), keepdims=True)
    eps_gain = 0.2
    eps = eps_gain * D

    vel_top = ((HH + D / 2) / HH) ** wind_shear * jnp.ones_like(Ct)
    Gamma_top = gamma(D, vel_top, Uinf, Ct, scale)

    vel_bottom = ((HH - D / 2) / HH) ** wind_shear * jnp.ones_like(Ct)
    Gamma_bottom = -gamma(D, vel_bottom, Uinf, Ct, scale)

    mean_cubed = jnp.mean(u_i**3, axis=(2, 3), keepdims=True)

    turbine_average_velocity = (
        jnp.sign(mean_cubed) * jnp.abs(mean_cubed) ** (1/3)   
    )

    Gamma_wake_rotation = 0.25 * 2 * jnp.pi * D * (aI - aI**2) * turbine_average_velocity / TSR
    yLocs = delta_y + NUM_EPS

    def vortex_velocity(Gamma, z_shift):
        z_ = z_i - z_shift + NUM_EPS
        r_ = yLocs**2 + z_**2
        core_shape = 1 - jnp.exp(-r_ / eps**2)
        return (Gamma * z_) / (2 * jnp.pi * r_) * core_shape

    v_top = jnp.mean(vortex_velocity(Gamma_top, HH + D / 2), axis=(2, 3))
    v_bottom = jnp.mean(vortex_velocity(Gamma_bottom, HH - D / 2), axis=(2, 3))
    v_core = jnp.mean(vortex_velocity(Gamma_wake_rotation, HH), axis=(2, 3))

    val = 2.0 * (avg_v - v_core) / (v_top + v_bottom)
    val = jnp.clip(val, -1.0, 1.0)
    y = 0.5 * jnp.arcsin(val)

    return y[:, :, None, None]

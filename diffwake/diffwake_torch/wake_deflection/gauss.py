import torch
import math
from torch import nn, Tensor


class GaussVelocityDeflection(nn.Module):
    """
    Torch–friendly implementation of the original analyt-ics model.
    Everything that is a scalar stays a scalar buffer so it lives on
    the correct device and participates in torch.compile fusion.
    """

    def __init__(
        self,
        ad: float = 0.0, bd: float = 0.0,
        alpha: float = 0.58, beta: float = 0.077,
        ka: float = 0.38, kb: float = 0.004,
        dm: float = 1.0, eps_gain: float = 0.2,
        use_secondary_steering: bool = True
    ):
        super().__init__()
        # ───────────────────────── constants as buffers ──────────────────────────
        for name, value in dict(
            ad=ad, bd=bd, alpha=alpha, beta=beta,
            ka=ka, kb=kb, dm=dm, eps_gain=eps_gain
        ).items():
            self.register_buffer(name, torch.as_tensor(float(value)))

        self.use_secondary_steering = bool(use_secondary_steering)
        # reusable constant
        self.register_buffer("_sqrt2", torch.tensor(math.sqrt(2.0)))

    @staticmethod
    def _cos(x: Tensor) -> Tensor:
        return torch.cos(x)

    def forward(
        self,
        x_i: Tensor, y_i: Tensor,                   # (B,1,1,1)
        yaw_i: Tensor,                              # radians, sign + = CW
        turb_I_i: Tensor,                           # turbulence intensity
        ct_i: Tensor,
        D: Tensor,                                  # rotor diameter
        x: Tensor, y: Tensor, z: Tensor,            # global grids
        U_free: Tensor,                             # free-stream velocity
        wind_veer: float                            # scalar in rad
    ) -> Tensor:

        # ───────── small helpers ─────────
        yaw = -yaw_i.clone()
        cos_yaw = self._cos(yaw)                 # flip sign once
        one = torch.ones_like(cos_yaw)

        # (0) basic terms ---------------------------------------------------------
        uR = U_free * ct_i * cos_yaw / (
            2. * (one - torch.sqrt(one - ct_i * cos_yaw))
        )
        u0 = U_free * torch.sqrt(one - ct_i)

        # (1) near-wake start -----------------------------------------------------
        denom = self._sqrt2 * (
            4 * self.alpha * turb_I_i +
            2 * self.beta  * (one - torch.sqrt(one - ct_i))
        )
        x0 = D * cos_yaw * (1 + torch.sqrt(one - ct_i * cos_yaw)) / denom + x_i

        # (2) lateral / vertical spread coefficients -----------------------------
        k   = self.ka * turb_I_i + self.kb        # same for y & z here

        # (3) init sigma ----------------------------------------------------------
        sigma_z0 = D * 0.5 * torch.sqrt(uR / (U_free + u0))
        veer = torch.deg2rad(torch.as_tensor(wind_veer, dtype=cos_yaw.dtype,
                       device=cos_yaw.device))
        # Wind veer is still in degree.
        sigma_y0 = sigma_z0 * cos_yaw *torch.cos(veer)

        # (4) centre-line deflection at x0 ---------------------------------------
        theta_c0 = self.dm * 0.3 * yaw / cos_yaw
        theta_c0 *= (1 - torch.sqrt(1 - ct_i * cos_yaw))
        delta0   = torch.tan(theta_c0) * (x0 - x_i)

        # (5) masks ---------------------------------------------------------------
        mask_near = (x >= x_i) & (x <= x0)
        mask_far  = x > x0

        # (6) near-wake deflection ----------------------------------------------
        delta_near = ((x - x_i) / (x0 - x_i)) * delta0 + (self.ad + self.bd*(x - x_i))
        delta_near = torch.where(mask_near, delta_near, torch.zeros_like(delta_near))

        # (7) sigmas downstream ---------------------------------------------------
        sigma_y = torch.where(mask_far, k * (x - x0) + sigma_y0, sigma_y0)
        sigma_z = torch.where(mask_far, k * (x - x0) + sigma_z0, sigma_z0)

        # (8) far-wake deflection -----------------------------------------------
        C0   = 1 - u0 / U_free
        M0   = C0 * (2 - C0)
        E0   = C0**2 - 3*math.exp(1/12) * C0 + 3*math.exp(1/3)
        M0_s = torch.sqrt(M0)

        mid  = torch.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0))
        ln_num = (1.6 + M0_s) * (1.6 * mid - M0_s)
        ln_den = (1.6 - M0_s) * (1.6 * mid + M0_s)
        log_t  = torch.log(ln_num / ln_den)
        mult  = theta_c0 * E0 / 5.2 * torch.sqrt(sigma_y0 * sigma_z0 / (k * k * M0))
        delta_far = delta0 + mult * log_t + (self.ad + self.bd * (x - x_i))
        delta_far = torch.where(mask_far, delta_far, torch.zeros_like(delta_far))

        # (9) combine ------------------------------------------------------------
        return (delta_near + delta_far)

def gamma(D, velocity, Uinf, Ct, scale=1.0):
    return scale * (math.pi / 8) * D * velocity * Uinf * Ct




def cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x).pow(1/3)


NUM_EPS = 0.001
NUM_EPS_2 =1e-7

EPS_GAIN = 0.2

eps = 1e-8

def cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x).pow(1/3)

def gamma(
    D: float,
    velocity: Tensor,      # or ndarray
    Uinf: Tensor,
    Ct: Tensor,
    scale: float = 1.0,
):
    """Yaw‑vortex circulation strength (same algebra as NumPy)."""
    return scale * (math.pi / 8.0) * D * velocity * Uinf * Ct


# ------------------------------------------------------------------
def calculate_transverse_velocity(
    u_i: Tensor, u_initial: Tensor, dudz_initial: Tensor,
    delta_x: Tensor, delta_y: Tensor, z: Tensor,
    rotor_diameter: float, hub_height: float,
    yaw_rad: Tensor,                      # radians
    ct_i: Tensor, tsr_i: Tensor, axial_induction_i: Tensor,
    wind_shear: float, scale: float = 1.0,
):


    D, HH = rotor_diameter, hub_height
    Ct, TSR, aI = ct_i, tsr_i, axial_induction_i

    Uinf = u_initial.mean(dim=(1, 2, 3), keepdim=True)
    eps  = EPS_GAIN * D
    ones = torch.ones_like(Ct)

    s_c = torch.sin(yaw_rad) * torch.cos(yaw_rad)       # yaw in **radians**
    vel_top    = ((HH + 0.5 * D) / HH) ** wind_shear * ones
    vel_bottom = ((HH - 0.5 * D) / HH) ** wind_shear * ones

    Γ_top    =  s_c * gamma(D, vel_top,    Uinf, Ct, scale)
    Γ_bottom = -s_c * gamma(D, vel_bottom, Uinf, Ct, scale)

    mean_cubed    = (u_i ** 3).mean(dim=(2, 3), keepdim=True)
    turbine_avg_u = cbrt(mean_cubed)
    Γ_core = 0.5 * math.pi * D * (aI - aI ** 2) * turbine_avg_u / TSR  # 0.5 π = 0.25 · 2π

    lmda  = D / 8.0
    kappa = 0.41
    lm    = kappa * z / (1.0 + kappa * z / lmda)
    nu    = lm ** 2 * torch.abs(dudz_initial)
    decay = eps ** 2 / (4.0 * nu * delta_x / Uinf + eps ** 2)
    yLocs = delta_y + NUM_EPS
    def vortex(Γ: Tensor, z_shift: float):
                      # exactly as NumPy
        z_    = z - z_shift + NUM_EPS
        r2    = yLocs ** 2 + z_ ** 2
        core  = 1.0 - torch.exp(-r2 / eps ** 2)
        V =  (Γ * z_)    / (2.0 * math.pi * r2) * core * decay
        W = (-Γ * yLocs) / (2.0 * math.pi * r2) * core * decay
        return V, W

    V1, W1 = vortex( Γ_top,             HH + 0.5 * D)
    V2, W2 = vortex( Γ_bottom,          HH - 0.5 * D)
    V5, W5 = vortex( Γ_core,            HH)

    V3, W3 = vortex(-Γ_top,            -(HH + 0.5 * D))
    V4, W4 = vortex(-Γ_bottom,         -(HH - 0.5 * D))
    V6, W6 = vortex(-Γ_core,           -HH)

    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6


    V = torch.where(delta_x + NUM_EPS_2 >= 0.0, V, torch.zeros_like(V))
    W = torch.where(delta_x + NUM_EPS_2 >= 0.0, W, torch.zeros_like(W))
    W = torch.where(W  +NUM_EPS_2  >= 0.0, W, torch.zeros_like(W))  # no downward W

    return V, W

def yaw_added_turbulence_mixing(
    u_i,
    I_i,
    v_i,
    w_i,
    turb_v_i,
    turb_w_i
):
    I_i = I_i[:, 0, 0, 0].clone()

    average_u_i = torch.sign(u_i).mul(u_i.abs().pow(3)).mean(dim=(1, 2, 3)).pow(1/3)

    k = (average_u_i * I_i) ** 2 / (2.0 / 3.0)

    u_term = (2.0 * k).sqrt()
    v_term = (v_i + turb_v_i).mean(dim=(1, 2, 3))
    w_term = (w_i + turb_w_i).mean(dim=(1, 2, 3))

    k_total = 0.5 * (u_term ** 2 + v_term ** 2 + w_term ** 2)

    I_total = ((2.0 / 3.0) * k_total).sqrt() / average_u_i

    I_mixing = I_total - I_i

    return I_mixing[:, None, None, None]
def gamma(D, velocity, Uinf, Ct, scale=1.0):
    return scale * (math.pi / 8) * D * velocity * Uinf * Ct


def cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x).pow(1/3)


def wake_added_yaw(
    u_i: Tensor,                     # (B,1,Ny,Nz)
    v_i: Tensor,                     # (B,1,Ny,Nz)
    u_initial: Tensor,               # (B,1,Ny,Nz)
    delta_y: Tensor,                 # (B,1,Ny,Nz)
    z_i: Tensor,                     # (B,1,Ny,Nz)
    rotor_diameter: float,
    hub_height: float,
    ct_i: Tensor,                    # (B,1,1,1)
    tip_speed_ratio: float,
    axial_induction_i: Tensor,       # (B,1,1,1)
    wind_shear: float,
    scale: float = 1.0,
) -> Tensor:
    """
    Beräkna yaw-vinkeln (radianer) som skulle ge samma medel­sidvind
    som observeras i v_i-fältet. Följer originalets logik men utan numexpr.
    """
    NUM_EPS = 0.001                  # samma värde överallt
    D, HH = rotor_diameter, hub_height
    Ct, TSR = ct_i, tip_speed_ratio
    aI = axial_induction_i

    avg_v = v_i.mean(dim=(2, 3))                     # (B,1)
    Uinf  = u_initial.mean(dim=(1, 2, 3), keepdim=True)  # (B,1,1,1)

    eps = 0.2 * D
    ones = torch.ones_like(Ct)                       # (B,1,1,1)

    vel_top    = ((HH + 0.5 * D) / HH) ** wind_shear * ones
    vel_bottom = ((HH - 0.5 * D) / HH) ** wind_shear * ones

    Gamma_top    =  gamma(D, vel_top,    Uinf, Ct, scale)
    Gamma_bottom = -gamma(D, vel_bottom, Uinf, Ct, scale)

    mean_cubed = (u_i ** 3).mean(dim=(2, 3), keepdim=True)
    turbine_avg_u = cbrt(mean_cubed)
    Gamma_core = 0.5 * math.pi * D * (aI - aI ** 2) * turbine_avg_u / TSR

    yLocs = delta_y + NUM_EPS
    zT = z_i - (HH + 0.5 * D) + NUM_EPS
    zB = z_i - (HH - 0.5 * D) + NUM_EPS
    zC = z_i - HH             + NUM_EPS

    rT2 = yLocs**2 + zT**2
    rB2 = yLocs**2 + zB**2
    rC2 = yLocs**2 + zC**2

    core_T = 1.0 - torch.exp(-rT2 / eps**2)
    core_B = 1.0 - torch.exp(-rB2 / eps**2)
    core_C = 1.0 - torch.exp(-rC2 / eps**2)

    v_top    = (Gamma_top   * zT) / (2.0 * math.pi * rT2) * core_T
    v_bottom = (Gamma_bottom* zB) / (2.0 * math.pi * rB2) * core_B
    v_core   = (Gamma_core  * zC) / (2.0 * math.pi * rC2) * core_C

    v_top    = v_top.mean(dim=(2, 3))    # (B,1)
    v_bottom = v_bottom.mean(dim=(2, 3))
    v_core   = v_core.mean(dim=(2, 3))

    val = 2.0 * (avg_v - v_core) / (v_top + v_bottom)
    val = torch.clamp(val, -1.0, 1.0)
    yaw_rad = 0.5 * torch.arcsin(val)    # radianer

    return yaw_rad[:, :, None, None]     # behåller (B,1,1,1)
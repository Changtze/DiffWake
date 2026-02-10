import jax
import jax.numpy as jnp
from flax import struct                    # Lightweight immutable dataclass
from jax import lax
from typing import Tuple


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
class CumulativeGaussCurlVelocityDeficit:
    a_s: float = 0.179367259
    b_s: float = 0.0118889215
    c_s1: float = 0.0563691592
    c_s2: float = 0.13290157
    a_f: float = 3.11
    b_f: float = -0.68
    c_f: float = 2.41
    alpha_mod: float = 1.0          
    
    def _vec_sum_lbda(self, ii, Ctmp, u_initial, x,
                    x_coord, y_coord, z_coord,
                    ct, ti, D, sigma_n_sq,
                    y_i_loc, z_i_loc, deflection, a_s,b_s, c_s1, c_s2):
        """
        Vectorised λ-sum with **static shapes**.
        Works inside jit/scan because it never creates
        slices whose length depends on the tracer `ii`.
        """

        B, T, _, _ = ct.shape
        mask_1d = jnp.arange(T) < ii                 # True för j < ii

        mask_4d    = mask_1d[None, :, None, None]    # (1,T,1,1)

        mask_5d = mask_1d[None, :, None, None, None] # for broadcast

        def m(arr):
            return jnp.where(mask_4d, arr, 0.0)      # keep shape, zero out

        ct_m = m(ct)             # (B,T,1,1)
        ti_m = m(ti)
        x_m  = m(x_coord)
        y_m  = m(y_coord)
        z_m  = m(z_coord)


        Ctmp_full = Ctmp_full = jnp.moveaxis(Ctmp, 0, 1)
        Ctmp_m    = jnp.where(mask_5d, Ctmp_full, 0.0)     # (B, T, T, Ny, Nz)

        # ------------------------------------------------------------------
        dx_m = jnp.expand_dims(x, 1) - jnp.expand_dims(x_m, 2)    # (B,T,…)

        sigma_i = wake_expansion(
            dx_m,
            jnp.expand_dims(ct_m, 2),
            jnp.expand_dims(ti_m, 2),
            D, a_s, b_s, c_s1, c_s2
        )
        sigma_i_sq = sigma_i ** 2
        S_i        = jnp.expand_dims(sigma_n_sq, 1) + sigma_i_sq

        defl_m = deflection[:, None, ...]                 
        Y_i = ((jnp.expand_dims(y_i_loc, 1) - jnp.expand_dims(y_m, 2) - defl_m) ** 2) / (2 * S_i)
        Z_i = ((jnp.expand_dims(z_i_loc, 1) - jnp.expand_dims(z_m, 2)) ** 2) / (2 * S_i)

        lbda = sigma_i_sq / S_i * jnp.exp(-Y_i - Z_i)
        lbda = jnp.where(mask_5d, lbda, 0.0)         # nolla “senare” turbiner
        # Sum over turbine-axis (axis=1), keep static shapes
        term = lbda * (Ctmp_m / jnp.expand_dims(u_initial, 1))
        return jnp.sum(term, axis=1)
    # ─────────────────────────────────────────────────────────────────────
    #  Forward call  (identical API & tensor algebra to PyTorch)
    # ─────────────────────────────────────────────────────────────────────
    def __call__(self, ii,
                 x_i, y_i, z_i, u_i,
                 deflection_field, yaw_i,
                 ti, ct, D,
                 turb_u_wake, Ctmp,
                 x, y, z, u_initial):
        mean_cubed = jnp.mean(u_i ** 3, axis=(2, 3), keepdims=True)
        turb_avg_vels = jnp.sign(mean_cubed) * jnp.abs(mean_cubed) ** (1 / 3)
        delta_x = x - x_i
        dtype = x.dtype
        a_s  = jnp.asarray(self.a_s,  dtype)
        b_s  = jnp.asarray(self.b_s,  dtype)
        c_s1 = jnp.asarray(self.c_s1, dtype)
        c_s2 = jnp.asarray(self.c_s2, dtype)


        D    = jnp.asarray(D,         dtype)   

        ct_i = lax.dynamic_index_in_dim(ct, ii, axis=1, keepdims=True)
        ti_i = lax.dynamic_index_in_dim(ti, ii, axis=1, keepdims=True)
        sigma_n = wake_expansion(delta_x, ct_i, ti_i, D,
                                 a_s, b_s, c_s1, c_s2)

        y_i_loc = jnp.mean(y_i, axis=(2, 3), keepdims=True)
        z_i_loc = jnp.mean(z_i, axis=(2, 3), keepdims=True)

        x_coord = jnp.mean(x, axis=(2, 3), keepdims=True)
        y_coord = jnp.mean(y, axis=(2, 3), keepdims=True)
        z_coord = jnp.mean(z, axis=(2, 3), keepdims=True)

        sigma_n_sq = sigma_n ** 2

        sum_lbda = self._vec_sum_lbda(ii, Ctmp, u_initial, x,
                                      x_coord, y_coord, z_coord,
                                      ct, ti, D, sigma_n_sq,
                                      y_i_loc, z_i_loc, deflection_field,
                                      a_s,b_s, c_s1, c_s2)
        

        x_tilde = jnp.abs(delta_x) / D

        inside = (y - y_i_loc - deflection_field) ** 2 + (z - z_i_loc) ** 2
        r_tilde = safe_sqrt(inside) / D

        n = self.a_f * jnp.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)

        gamma_val = jnp.exp(jax.lax.lgamma(2 / n))

        tmp = a2 - ((n * ct_i) * jnp.cos(yaw_i) /
                    (16.0 * gamma_val * jnp.sign(sigma_n) *
                     (jnp.abs(sigma_n) ** (4 / n)) * (1 - sum_lbda) ** 2))

        C_new = (a1 - safe_sqrt(tmp)) * (1 - sum_lbda)

        C_new5 = jnp.expand_dims(C_new, axis=0)       # (1,B,T,Ny,Nz)
        Ctmp_out = lax.dynamic_update_slice_in_dim(Ctmp, C_new5, ii, axis=0)
        yR = y - y_i_loc
        xR = yR * jnp.tan(yaw_i) + x_i

        velDef = C_new * jnp.exp(-(jnp.abs(r_tilde) ** n) / (2 * sigma_n_sq))
        velDef = jnp.where((x - xR) >= 0.1, velDef, 0.0)

        # functional update of turb_u_wake
        turb_u_wake_updated = turb_u_wake + velDef * turb_avg_vels

        return turb_u_wake_updated, Ctmp_out


@struct.dataclass
class CumulativeGaussCurlVelocityDeficitAsBs:
    c_s1: float = 0.0563691592
    c_s2: float = 0.13290157
    a_f: float = 3.11
    b_f: float = -0.68
    c_f: float = 2.41
    alpha_mod: float = 1.0          

    
    def _vec_sum_lbda(self, ii, Ctmp, u_initial, x,
                    x_coord, y_coord, z_coord,
                    ct, ti, D, sigma_n_sq,
                    y_i_loc, z_i_loc, deflection, a_s,b_s, c_s1, c_s2):
        """
        Vectorised λ-sum with **static shapes**.
        Works inside jit/scan because it never creates
        slices whose length depends on the tracer `ii`.
        """

        B, T, _, _ = ct.shape
        mask_1d = jnp.arange(T) < ii                 # True för j < ii

        mask_4d    = mask_1d[None, :, None, None]    # (1,T,1,1)

        mask_5d = mask_1d[None, :, None, None, None] # for broadcast

        def m(arr):
            return jnp.where(mask_4d, arr, 0.0)      # keep shape, zero out

        ct_m = m(ct)             # (B,T,1,1)
        ti_m = m(ti)
        x_m  = m(x_coord)
        y_m  = m(y_coord)
        z_m  = m(z_coord)


        Ctmp_full = Ctmp_full = jnp.moveaxis(Ctmp, 0, 1)#jnp.transpose(Ctmp, (1, 0, 2, 3, 4))   # (B, T, T, Ny, Nz)
        Ctmp_m    = jnp.where(mask_5d, Ctmp_full, 0.0)     # (B, T, T, Ny, Nz)

        # ------------------------------------------------------------------
        dx_m = jnp.expand_dims(x, 1) - jnp.expand_dims(x_m, 2)    # (B,T,…)

        sigma_i = wake_expansion(
            dx_m,
            jnp.expand_dims(ct_m, 2),
            jnp.expand_dims(ti_m, 2),
            D, jnp.expand_dims(a_s, 2), jnp.expand_dims(b_s, 2), c_s1, c_s2
        )
        sigma_i_sq = sigma_i ** 2
        S_i        = jnp.expand_dims(sigma_n_sq, 1) + sigma_i_sq

        defl_m = deflection[:, None, ...]                 
        Y_i = ((jnp.expand_dims(y_i_loc, 1) - jnp.expand_dims(y_m, 2) - defl_m) ** 2) / (2 * S_i)
        Z_i = ((jnp.expand_dims(z_i_loc, 1) - jnp.expand_dims(z_m, 2)) ** 2) / (2 * S_i)

        lbda = sigma_i_sq / S_i * jnp.exp(-Y_i - Z_i)
        lbda = jnp.where(mask_5d, lbda, 0.0)         # nolla “senare” turbiner
        # Sum over turbine-axis (axis=1), keep static shapes
        term = lbda * (Ctmp_m / jnp.expand_dims(u_initial, 1))
        return jnp.sum(term, axis=1)

    def __call__(self, ii,
                 x_i, y_i, z_i, u_i,
                 deflection_field, yaw_i,
                 ti, ct, D,
                 turb_u_wake, Ctmp,
                 x, y, z, u_initial, a_s, b_s):
        mean_cubed = jnp.mean(u_i ** 3, axis=(2, 3), keepdims=True)
        turb_avg_vels = jnp.sign(mean_cubed) * jnp.abs(mean_cubed) ** (1 / 3)
        delta_x = x - x_i
        dtype = x.dtype
        c_s1 = jnp.asarray(self.c_s1, dtype)
        c_s2 = jnp.asarray(self.c_s2, dtype)

        D    = jnp.asarray(D,         dtype)   # promote the argument as well

        ct_i = lax.dynamic_index_in_dim(ct, ii, axis=1, keepdims=True)
        ti_i = lax.dynamic_index_in_dim(ti, ii, axis=1, keepdims=True)

        B = x.shape[0]                                         # CHANGE
        a_s = a_s.reshape(B, 1, 1, 1) # CHANGE
        b_s = b_s.reshape(B, 1, 1, 1) # CHANGE

        sigma_n = wake_expansion(delta_x, ct_i, ti_i, D,
                                 a_s, b_s, c_s1, c_s2)

        y_i_loc = jnp.mean(y_i, axis=(2, 3), keepdims=True)
        z_i_loc = jnp.mean(z_i, axis=(2, 3), keepdims=True)

        x_coord = jnp.mean(x, axis=(2, 3), keepdims=True)
        y_coord = jnp.mean(y, axis=(2, 3), keepdims=True)
        z_coord = jnp.mean(z, axis=(2, 3), keepdims=True)

        sigma_n_sq = sigma_n ** 2

        sum_lbda = self._vec_sum_lbda(ii, Ctmp, u_initial, x,
                                      x_coord, y_coord, z_coord,
                                      ct, ti, D, sigma_n_sq,
                                      y_i_loc, z_i_loc, deflection_field,
                                      a_s,b_s, c_s1, c_s2)
        

        x_tilde = jnp.abs(delta_x) / D

        inside = (y - y_i_loc - deflection_field) ** 2 + (z - z_i_loc) ** 2
        r_tilde = safe_sqrt(inside) / D

        n = self.a_f * jnp.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)

        gamma_val = jnp.exp(jax.lax.lgamma(2 / n))

        tmp = a2 - ((n * ct_i) * jnp.cos(yaw_i) /
                    (16.0 * gamma_val * jnp.sign(sigma_n) *
                     (jnp.abs(sigma_n) ** (4 / n)) * (1 - sum_lbda) ** 2))

        C_new = (a1 - safe_sqrt(tmp)) * (1 - sum_lbda)

        C_new5 = jnp.expand_dims(C_new, axis=0)       # (1,B,T,Ny,Nz)
        Ctmp_out = lax.dynamic_update_slice_in_dim(Ctmp, C_new5, ii, axis=0)
        yR = y - y_i_loc
        xR = yR * jnp.tan(yaw_i) + x_i

        velDef = C_new * jnp.exp(-(jnp.abs(r_tilde) ** n) / (2 * sigma_n_sq))
        velDef = jnp.where((x - xR) >= 0.1, velDef, 0.0)

        turb_u_wake_updated = turb_u_wake + velDef * turb_avg_vels

        return turb_u_wake_updated, Ctmp_out
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffWake – Turbulence Intensity (TI) Model Training Script
Trains a neural model that predicts turbulence-intensity (TI) distributions
using a differentiable wind-farm simulator (DiffWake JAX backend).

Author: Maria Bånkestad (2025)
License: BSD-3-Clause (compatible with FLORIS license)
"""

from __future__ import annotations
import time
from typing import Any
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
import optax
import flax.linen as nn
import flax.serialization as ser
from flax.training.train_state import TrainState

# -----------------------------------------------------------------------------
# Optional persistent cache setup
# -----------------------------------------------------------------------------
try:
    from setup_cashe import init_jax_persistent_cache
    init_jax_persistent_cache()
except ImportError:
    pass

# -----------------------------------------------------------------------------
# DiffWake imports
# -----------------------------------------------------------------------------
from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.par_runner import make_sub_par_runner
from diffwake.diffwake_jax.util import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DTYPE = jnp.float64
jax.config.update("jax_enable_x64", True)

CUT_IN_MS = 3.8
CUT_OUT_MS = 25.0
P_THR_MW = 1e-3
TI_LO, TI_HI = 0.001, 0.40
BATCH_SIZE = 1024

# -----------------------------------------------------------------------------
# Kumaraswamy utilities
# -----------------------------------------------------------------------------
def softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)

def ti_ab_from_raw(ti_raw: jnp.ndarray, min_ab=0.5, max_ab=20.0):
    a = jnp.minimum(softplus(ti_raw[:, 0]) + min_ab, max_ab)
    b = jnp.minimum(softplus(ti_raw[:, 1]) + min_ab, max_ab)
    return a, b

def kumar_sample(key, a, b, shape, eps=1e-7):
    u = jax.random.uniform(key, shape=shape, minval=eps, maxval=1.0 - eps)
    return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)

def kumar_log_prob(x, a, b, eps=1e-12):
    x = jnp.clip(x, eps, 1.0 - eps)
    xa = x ** a
    one_minus_xa = jnp.clip(1.0 - xa, eps, 1.0)
    return (
        jnp.log(a)
        + jnp.log(b)
        + (a - 1.0) * jnp.log(x)
        + (b - 1.0) * jnp.log(one_minus_xa)
    )

def map01_to_ti(x01, lo, hi): 
    return lo + (hi - lo) * x01

def kumar_mean01(a, b): 
    return b * jnp.exp(gammaln(1 + 1/a) + gammaln(b) - gammaln(1 + 1/a + b))

def kumar_var01(a, b):
    m1 = kumar_mean01(a, b)
    m2 = b * jnp.exp(gammaln(1 + 2/a) + gammaln(b) - gammaln(1 + 2/a + b))
    return jnp.maximum(0.0, m2 - m1 * m1)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_BN(arr: jnp.ndarray) -> jnp.ndarray:
    """Ensure consistent (B,N) output shape from simulator."""
    if arr.ndim == 4 and arr.shape[-1] == 1 and arr.shape[-2] == 1:
        return arr[..., 0, 0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected shape {arr.shape}")

def save_ckpt(path: str, *, params, X_mu, X_std, TI_lo, TI_hi, model_cfg, step: int | None = None):
    payload = {
        "params": params,
        "norm": {"X_mu": np.asarray(X_mu), "X_std": np.asarray(X_std)},
        "config": {"TI_LO": float(TI_lo), "TI_HI": float(TI_hi), "model": dict(model_cfg)},
        "step": int(step) if step is not None else None,
        "format_version": 1,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(flax.serialization.msgpack_serialize(payload))
    print(f"[saved] {p}")

def load_ckpt(path: str) -> dict[str, Any]:
    p = Path(path)
    return flax.serialization.msgpack_restore(p.read_bytes())

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
KINIT = nn.initializers.lecun_normal()
BINIT = nn.initializers.zeros

class Trunk(nn.Module):
    hidden: int = 32
    @nn.compact
    def __call__(self, x):
        x = nn.gelu(nn.Dense(self.hidden, kernel_init=KINIT, bias_init=BINIT)(x))
        x = nn.gelu(nn.Dense(self.hidden, kernel_init=KINIT, bias_init=BINIT)(x))
        return x

class TIHeads(nn.Module):
    hidden: int = 32
    @nn.compact
    def __call__(self, x):
        h = Trunk(self.hidden)(x)
        return nn.Dense(2, kernel_init=KINIT, bias_init=BINIT)(h)  # (a,b) logits

# -----------------------------------------------------------------------------
# Loss (fully documented, self-contained)
# -----------------------------------------------------------------------------
def compute_loss_minibatch(
    params: dict,
    apply_fn,
    rng: jax.Array,
    idx_batch: jnp.ndarray,
    *,
    Xn_all: jnp.ndarray,            # (B,D)
    P_sort_all: jnp.ndarray,        # (B,N) MW
    bool_mask_sort: jnp.ndarray,    # (B,N) in {0,1}
    runner,                         # (ti_vec[BS], idx_batch[BS]) -> sim out
    state,                          # simulator state
    K: int = 1,
    ti_lo: float = TI_LO,
    ti_hi: float = TI_HI,
    w_kl: float = 0.1,
    sigma_MW: float | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Monte-Carlo marginal-likelihood over latent TI with KL regularization."""
    idx_batch = jnp.asarray(idx_batch, jnp.int32).reshape((-1,))
    Xn    = Xn_all[idx_batch, :]
    P_obs = P_sort_all[idx_batch, :]
    wmask = bool_mask_sort[idx_batch, :].astype(DTYPE)

    ti_raw = apply_fn({'params': params["net"]}, Xn)
    a, b   = ti_ab_from_raw(ti_raw)

    (k_samp,) = jax.random.split(rng, 1)
    x01_KB = kumar_sample(k_samp, a[None, :], b[None, :], (K, Xn.shape[0]))
    TI_KB  = jnp.clip(map01_to_ti(x01_KB, ti_lo, ti_hi), ti_lo + 1e-4, ti_hi - 1e-4)

    def _simulate_one(ti_vec: jnp.ndarray):
        out = runner(ti_vec, idx_batch)
        vel = average_velocity_jax(out.u_sorted)
        pow_W = power(
            state.farm.power_thrust_table,
            vel,
            state.flow.air_density,
            yaw_angles=state.farm.yaw_angles[idx_batch],
        )
        pow_MW = to_BN(pow_W) / DTYPE(1e6)
        pow_MW = jnp.clip(jnp.nan_to_num(pow_MW, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 5.0)
        vel_BN = to_BN(vel)
        op_mask = (vel_BN >= CUT_IN_MS) & (vel_BN < CUT_OUT_MS) & (pow_MW >= P_THR_MW)
        return pow_MW, op_mask

    P_pred_KBN, op_mask_KBN = jax.vmap(_simulate_one, in_axes=0)(TI_KB)

    if sigma_MW is None:
        sigma = jnp.exp(params["log_sigma"]).astype(DTYPE) if ("log_sigma" in params) else DTYPE(0.03)
    else:
        sigma = DTYPE(sigma_MW)

    diff_KBN = P_pred_KBN - P_obs[None, :, :]
    log_norm_const = -0.5 * (2.0 * jnp.log(sigma) + jnp.log(2.0 * jnp.pi))
    log_lik_KBN = -0.5 * (diff_KBN / sigma) ** 2 + log_norm_const
    log_marg_BN = logsumexp(log_lik_KBN, axis=0) - jnp.log(K)

    op_mask_any_BN = jnp.any(op_mask_KBN, axis=0).astype(DTYPE)
    mask_BN = wmask * op_mask_any_BN
    denom = jnp.maximum(jnp.sum(mask_BN), 1.0)
    nll = -jnp.sum(mask_BN * log_marg_BN) / denom

    a0, b0 = DTYPE(1.0), DTYPE(3.75)
    logq = kumar_log_prob(x01_KB, jnp.clip(a[None, :], 1e-3, 1e3), jnp.clip(b[None, :], 1e-3, 1e3))
    logp = kumar_log_prob(x01_KB, a0, b0)
    kl = jnp.mean(logq - logp)

    loss = jnp.nan_to_num(nll + w_kl * kl, nan=1e6)
    return loss, (nll, kl, sigma)

# -----------------------------------------------------------------------------
# Batching utilities
# -----------------------------------------------------------------------------
def make_epoch_batches(key, n_samples: int, batch_size: int) -> jnp.ndarray:
    perm = jax.random.permutation(key, n_samples)
    n_full = (n_samples + batch_size - 1) // batch_size * batch_size
    pad = n_full - n_samples
    if pad > 0:
        perm = jnp.concatenate([perm, perm[:pad]], axis=0)
    return perm.reshape((-1, batch_size)).astype(jnp.int32)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train_ti_model(
    data_dir="data/smarteole",
    epochs=800,
    lr=1e-4,
    seed=0,
    K=1,
    checkpoint_path="checkpoints/ti_model.msgpack",
):
    """Train the turbulence-intensity model."""
    key = jax.random.PRNGKey(seed)

    # --- Load data ---
    X_np = np.load(f"{data_dir}/input_data.npy").astype(np.float32)
    P_np = np.load(f"{data_dir}/output_data.npy").astype(np.float32)[:, :7] / 1e3  # MW
    bool_mask = np.load(f"{data_dir}/bool_mask.npy")
    assert X_np.shape[0] == P_np.shape[0]

    X_raw = jnp.asarray(X_np, DTYPE)
    P_all = jnp.asarray(P_np, DTYPE)
    B_full, _ = X_np.shape

    # --- Build simulator state ---
    wd_deg, ws = X_raw[:, 0], X_raw[:, 1]
    ti = jnp.full_like(ws, 0.10)
    wd_rad = jnp.where(jnp.max(wd_deg) < 2 * jnp.pi, wd_deg, jnp.deg2rad(wd_deg))
    cfg = (
        load_input(f"{data_dir}/cc.yaml", f"{data_dir}/senvion_MM82.yaml")
        .set(wind_directions=wd_rad, wind_speeds=ws, turbulence_intensities=ti)
    )
    state = create_state(cfg)
    runner = make_sub_par_runner(state, batch_size=BATCH_SIZE)

    # --- Align P/Mask to solver turbine order ---
    idx_sort = state.grid.sorted_indices[:, :, 0, 0]  # (B_full, N)
    P_sort_all = jnp.take_along_axis(P_all, idx_sort, axis=1)
    bool_mask_sort = jnp.take_along_axis(jnp.asarray(bool_mask, dtype=DTYPE), idx_sort, axis=1)

    # --- Features ---
    wd_sin, wd_cos = jnp.sin(wd_rad), jnp.cos(wd_rad)
    X_feats = jnp.stack([wd_sin, wd_cos, ws, *X_raw[:, 2:].T], axis=1)  # (B_full, 7)
    X_mu, X_std = jnp.mean(X_feats, axis=0), jnp.std(X_feats, axis=0) + 1e-6
    Xn_all = (X_feats - X_mu) / X_std

    # --- Model & optimizer ---
    model = TIHeads(hidden=64)
    init_vars = model.init(key, Xn_all[:2])
    # optional learned sigma: uncomment to learn obs noise
    # init_vars = {**init_vars, "log_sigma": jnp.array(np.log(0.03), DTYPE)}
    tx = optax.adam(lr)
    state_model = TrainState.create(apply_fn=model.apply, params=init_vars, tx=tx)

    # Loss wrapper with closed-over dependencies
    def loss_wrap(p, rng, idx):
        return compute_loss_minibatch(
            p, model.apply, rng, idx,
            Xn_all=Xn_all,
            P_sort_all=P_sort_all,
            bool_mask_sort=bool_mask_sort,
            runner=runner,
            state=state,
            K=K,
            ti_lo=TI_LO,
            ti_hi=TI_HI,
            w_kl=0.1,
            # sigma_MW=None  # use learned if "log_sigma" present, else 0.03
        )

    loss_grad = jax.jit(jax.value_and_grad(loss_wrap, has_aux=True))

    # --- Training ---
    for ep in range(1, epochs + 1):
        key, k_ep = jax.random.split(key)
        batches = make_epoch_batches(k_ep, B_full, BATCH_SIZE)
        losses = []
        for idx in batches:
            key, k_step = jax.random.split(key)
            (loss_val, _aux), grads = loss_grad(state_model.params, k_step, idx)
            state_model = state_model.apply_gradients(grads=grads)
            losses.append(float(loss_val))
        if ep % 20 == 0:
            print(f"Epoch {ep:04d}: mean loss = {np.mean(losses):.6f}")

    # --- Save checkpoint ---
    model_cfg = {"hidden": 64, "feat_dim": int(Xn_all.shape[1])}
    payload = {
        "params": state_model.params,
        "norm": {"X_mu": np.asarray(X_mu), "X_std": np.asarray(X_std)},
        "config": {"TI_LO": float(TI_LO), "TI_HI": float(TI_HI), "model": model_cfg},
        "step": int(state_model.step),
        "format_version": 1,
    }
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(ser.msgpack_serialize(payload))
    print(f"Training complete. Model saved to {p}")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_ti_model()

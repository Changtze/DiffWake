from setup_cashe import init_jax_persistent_cache
init_jax_persistent_cache()

import time
import numpy as np
import jax, jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct
from jax.scipy.special import gammaln

from diffwake.diffwake_jax.model import load_input, create_state
from diffwake.diffwake_jax.par_runner import make_sub_par_runner  # <-- subset runner
from diffwake.diffwake_jax.util import average_velocity_jax
from diffwake.diffwake_jax.turbine.operation_models import power
from collections import deque
import flax.serialization as ser
from pathlib import Path
from jax.scipy.special import logsumexp  # add near your imports
bool_mask = np.load("bool_mask.npy")

def save_ckpt(path: str, *, params, X_mu, X_std, TI_lo, TI_hi, model_cfg, step: int | None = None):
    """
    params  : PyTree of model params (e.g., full.model.params)
    X_mu/X_std: feature normalization vectors
    TI_lo/TI_hi: TI bounds used in training
    model_cfg: dict, e.g. {"hidden": 64, "feat_dim": 7}
    step    : optional global step for bookkeeping
    """
    payload = {
        "params": params,
        "norm": {"X_mu": np.asarray(X_mu), "X_std": np.asarray(X_std)},
        "config": {"TI_LO": float(TI_lo), "TI_HI": float(TI_hi), "model": dict(model_cfg)},
        "step": int(step) if step is not None else None,
        "format_version": 1,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(ser.msgpack_serialize(payload))
    print(f"Saved -> {p}")

def load_ckpt(path: str):
    """Returns a dict with keys: params, norm{X_mu,X_std}, config{TI_LO,TI_HI,model}, step."""
    p = Path(path)
    payload = ser.msgpack_restore(p.read_bytes())
    print(f"Loaded <- {p}")
    return payload
# --------------------------
# Config / constants
# --------------------------
DTYPE = jnp.float64
jax.config.update("jax_enable_x64", True)

CUT_IN_MS   = 3.8
CUT_OUT_MS  = 25.0
P_THR_MW    = 1e-3
TI_LO, TI_HI = 0.001, 0.40

# --------------------------
# Utils (Kumar)
# --------------------------
def softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.)

def ti_ab_from_raw(ti_raw, min_ab=0.5, max_ab=20.0):
    a = jnp.minimum(softplus(ti_raw[:, 0]) + min_ab, max_ab)
    b = jnp.minimum(softplus(ti_raw[:, 1]) + min_ab, max_ab)
    return a, b

def kumar_sample(key, a, b, shape, eps=1e-7):
    u = jax.random.uniform(key, shape=shape, minval=eps, maxval=1.0 - eps)
    t = (1.0 - u) ** (1.0 / b)
    return (1.0 - t) ** (1.0 / a)

def kumar_log_prob(x, a, b, eps=1e-12):
    x = jnp.clip(x, eps, 1.0 - eps)
    xa = x ** a
    one_minus_xa = jnp.clip(1.0 - xa, eps, 1.0)
    return jnp.log(a) + jnp.log(b) + (a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log(one_minus_xa)

def map01_to_ti(x01, lo, hi):
    return lo + (hi - lo) * x01

def kumar_mean01(a, b):
    return b * jnp.exp(gammaln(1.0 + 1.0/a) + gammaln(b) - gammaln(1.0 + 1.0/a + b))

def kumar_moment01(a, b, r):
    return b * jnp.exp(gammaln(1.0 + r/a) + gammaln(b) - gammaln(1.0 + r/a + b))

def kumar_var01(a, b):
    m1 = kumar_mean01(a, b)
    m2 = kumar_moment01(a, b, r=2.0)
    return jnp.maximum(0.0, m2 - m1*m1)

# --------------------------
# Small helpers
# --------------------------
def to_BN(arr):
    if arr.ndim == 4 and arr.shape[-1] == 1 and arr.shape[-2] == 1:
        return arr[..., 0, 0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected shape {arr.shape}")

# --------------------------
# Data (full set)
# --------------------------
X_np = np.load("data/smarteole/input_data.npy").astype(np.float32)  
P_np = np.load("data/smarteole/output_data.npy").astype(np.float32)[:, :7] / 1.0e3
assert X_np.shape[0] == P_np.shape[0]
B_full, F = X_np.shape
assert F == 6

X_raw = jnp.asarray(X_np, DTYPE)
P_all = jnp.asarray(P_np, DTYPE)
N = P_all.shape[1]

# Build state on full set (runner will subset internally)
def build_state():
    wd_deg = X_raw[:, 0]
    ws     = X_raw[:, 1]
    ti     = jnp.full_like(ws, 0.10)
    wd_rad = jnp.where(jnp.max(wd_deg) < (2*jnp.pi + 0.1), wd_deg, jnp.deg2rad(wd_deg))
    cfg = (load_input("data/smarteole/cc.yaml", "data/smarteole/senvion_MM82.yaml")
           .set(wind_directions=wd_rad, wind_speeds=ws, turbulence_intensities=ti))
    return create_state(cfg)

state = build_state()

# Align targets to solver's sorted turbine order
idx_sort = state.grid.sorted_indices[:, :, 0, 0]      # (B_full, T)
P_sort_all = jnp.take_along_axis(P_all, idx_sort, axis=1)  # (B_full, N)
bool_mask_sort = jnp.take_along_axis(bool_mask, idx_sort, axis=1)  # (B_full, N)
# --------------------------
# MLP features (wd -> sin/cos)
# --------------------------
wd_deg = X_raw[:, 0]
wd_rad = jnp.where(jnp.max(wd_deg) < (2*jnp.pi + 0.1), wd_deg, jnp.deg2rad(wd_deg))
wd_sin = jnp.sin(wd_rad)
wd_cos = jnp.cos(wd_rad)
ws     = X_raw[:, 1]
hour_sin, hour_cos = X_raw[:, 2], X_raw[:, 3]
doy_sin,  doy_cos  = X_raw[:, 4], X_raw[:, 5]

X_feats = jnp.stack([wd_sin, wd_cos, ws, hour_sin, hour_cos, doy_sin, doy_cos], axis=1)  # (B_full,7)
X_mu  = jnp.mean(X_feats, axis=0)
X_std = jnp.std(X_feats, axis=0) + 1e-6
Xn_all = (X_feats - X_mu) / X_std

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
        return nn.Dense(2, kernel_init=KINIT, bias_init=BINIT)(h)  # (B,2)

BS = 1024  # mini-batch size (static for jit)
runner = make_sub_par_runner(state, batch_size=BS)   # signature: (ti_vec[BS], idx[BS]) -> result

def compute_loss_minibatch(params, apply_fn, rng, idx_batch, K=1, ti_lo=TI_LO, ti_hi=TI_HI, w_kl=3e-3):
    # Slice features/targets
    Xn = Xn_all[idx_batch, :]
    P_sort = P_sort_all[idx_batch, :]

    wake_mask = bool_mask_sort[idx_batch, :].astype(DTYPE) # (BS, N)

    # --- NN forward (note the 'net' leaf) ---
    ti_raw = apply_fn({'params': params["net"]}, Xn)  # (BS,2)
    a, b = ti_ab_from_raw(ti_raw)

    # Sample TI in [lo, hi], shape (K, BS)
    (k1,) = jax.random.split(rng, 1)
    x01_KB = kumar_sample(k1, a[None, :], b[None, :], (K, BS))
    TI_KB  = jnp.clip(map01_to_ti(x01_KB, ti_lo, ti_hi), ti_lo + 1e-4, ti_hi - 1e-4)

    # Simulator for each TI sample
    def sim_one(ti_vec):
        out = runner(ti_vec, idx_batch)
        vel = average_velocity_jax(out.u_sorted)
        pow_W = power(state.farm.power_thrust_table, vel, state.flow.air_density,
                      yaw_angles=state.farm.yaw_angles[idx_batch])
        pow_MW = to_BN(pow_W) / 1.0e6
        pow_MW = jnp.clip(jnp.nan_to_num(pow_MW, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 5.0)

        vel_BN = to_BN(vel)
        mask = (vel_BN >= CUT_IN_MS) & (vel_BN < CUT_OUT_MS) & (pow_MW >= P_THR_MW)
        return pow_MW, mask

    P_pred_KBN, mask_KBN = jax.vmap(sim_one, in_axes=0)(TI_KB)  # (K,BS,N), (K,BS,N)

    sigma = jnp.asarray(0.03, dtype=DTYPE)

    # --- Monte-Carlo marginal likelihood over K samples ---
    diff = P_pred_KBN - P_sort[None, :, :]  # (K,BS,N)
    log_norm = -0.5 * ((diff / sigma)**2)  # + (- jnp.log(sigma) - 0.5*jnp.log(2*jnp.pi))

    log_marg_BN = logsumexp(log_norm, axis=0) - jnp.log(K)  # (BS,N)

    w_BN = (P_sort >= P_THR_MW).astype(DTYPE)  # (BS,N)
    mask_ = wake_mask * w_BN
    count = jnp.sum(w_BN)
    nll = jnp.where(
        count > 0,
        -jnp.sum(mask_ * log_marg_BN) / count,
        0.0,
    )

    a0, b0 = 1.0, 3.75
    logq = kumar_log_prob(
        x01_KB,
        jnp.clip(a[None, :], 1e-3, 1e3),
        jnp.clip(b[None, :], 1e-3, 1e3),
    )
    logp = kumar_log_prob(x01_KB, a0, b0)
    Lkl = jnp.mean(logq - logp)

    # --- Total loss ---
    loss = jnp.nan_to_num(nll + w_kl * Lkl, nan=1e6)
    return loss, (nll, Lkl, sigma)

def make_loss_grad(apply_fn, K, ti_lo, ti_hi, w_kl):
    def loss_wrap(model_params, rng, idx_batch):
        idx_batch = jnp.asarray(idx_batch, jnp.int32).reshape((-1,))
        return compute_loss_minibatch(model_params, apply_fn, rng, idx_batch,
                                      K=K, ti_lo=ti_lo, ti_hi=ti_hi, w_kl=w_kl)
    return jax.jit(jax.value_and_grad(loss_wrap, argnums=0, has_aux=True))


def make_tx(lr_or_schedule, clip_norm: float = 1.0):
    chain = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr_or_schedule),
    )
    return optax.apply_if_finite(chain, max_consecutive_errors=5)

@struct.dataclass
class FullState:
    model: TrainState

def make_epoch_batches(key, B_full: int, BS: int) -> jnp.ndarray:
    """
    Return int32 array of shape [num_batches, BS] with a random permutation of
    0..B_full-1, padded (wrap-around) to a multiple of BS to keep shapes static.
    """
    perm = jax.random.permutation(key, B_full, independent=True)  # (B_full,)
    n_full = (B_full + BS - 1) // BS * BS                         # ceil to multiple of BS
    # pad by wrapping to the start of perm (no replacement, just reuse)
    pad = n_full - B_full
    if pad > 0:
        perm = jnp.concatenate([perm, perm[:pad]], axis=0)        # (n_full,)
    # reshape to [num_batches, BS]
    batches = perm.reshape((-1, BS)).astype(jnp.int32)
    return batches  # [num_batches, BS]

def main(epochs=800, lr=1e-4, seed=0, K=1):
    key = jax.random.PRNGKey(seed)

    # --- model & init ---
    model = TIHeads(hidden=64)
    vars_init = model.init(key, Xn_all[:2])  # shape probe

    log_sigma0 = jnp.array(np.log(0.03), DTYPE)

    num_batches_per_epoch = int(np.ceil(B_full / BS))
    total_steps = epochs * num_batches_per_epoch
    lr_min = 3e-7
    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=total_steps,
        alpha=lr_min / lr,  # final multiplier so lr(final) ~= lr_min
    )
    tx = make_tx(schedule, clip_norm=1.0)

    init_params = {
        "net": vars_init["params"],        
        "log_sigma": log_sigma0,         
    }



    model_state = TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=tx
    )
    full = FullState(model=model_state)
    loss_grad = make_loss_grad(model.apply, K=K, ti_lo=TI_LO, ti_hi=TI_HI, w_kl=0.1)

    @jax.jit
    def step(full, rng, idx_batch):
        idx_batch = jnp.asarray(idx_batch, jnp.int32).reshape((BS,))
        (loss_val, parts), grads = loss_grad(full.model.params, rng, idx_batch)
        nll, Lkl, sigma = parts  # unpack once here
        new_model = full.model.apply_gradients(grads=grads)
        return full.replace(model=new_model), loss_val, nll, Lkl, sigma

    # --- warmup (deterministic first batch to trigger compilation) ---
    key, k_ep, k_step = jax.random.split(key, 3)
    batches = make_epoch_batches(k_ep, B_full, BS)   # [num_batches, BS]
    full, loss0, nll0, Lkl0, sigma0 = step(full, k_step, batches[0])

    jax.block_until_ready(loss0)
    print(f"warmup loss: {float(loss0):.6f}  (NLL={float(nll0):.6f}  KL={float(Lkl0):.6f}  sigma={float(sigma0):.4f} MW)")
    rm_losses = deque(maxlen=20)
    rm_Lkls   = deque(maxlen=20)
    rm_NLLs   = deque(maxlen=20)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        key, k_ep = jax.random.split(key)
        batches = make_epoch_batches(k_ep, B_full, BS)
        num_batches = int(batches.shape[0])

        ep_loss_sum = 0.0
        ep_Lkl_sum  = 0.0
        ep_Nll_sum  = 0.0

        for bi in range(num_batches):
            key, k_step = jax.random.split(key)
            idx_batch = batches[bi]
            full, loss_val, nll, Lkl, _sigma = step(full, k_step, idx_batch)
            jax.block_until_ready(loss_val)
            ep_loss_sum += float(loss_val)
            ep_Lkl_sum  += float(Lkl)
            ep_Nll_sum  += float(nll)

        ep_loss = ep_loss_sum / num_batches
        ep_Lkl  = ep_Lkl_sum  / num_batches
        ep_Nll  = ep_Nll_sum  / num_batches

        rm_losses.append(ep_loss)
        rm_Lkls.append(ep_Lkl)
        rm_NLLs.append(ep_Nll)

        if ep % 20 == 0:
            cur_lr = float(schedule(int(full.model.step)))
            print(
                f"epoch {ep:03d}  lr={cur_lr:.2e}  "
                f"20-ep mean: loss={np.mean(rm_losses):.6f}  "
                f"NLL={np.mean(rm_NLLs):.6f}  KL={np.mean(rm_Lkls):.6f}"
            )



    print(f"done in {time.time()-t0:.1f}s")
    model_cfg = {"hidden": 64, "feat_dim": int(Xn_all.shape[1])}
    save_ckpt(
        "checkpoints/ti_model2.msgpack",
        params=full.model.params,
        X_mu=X_mu, X_std=X_std,
        TI_lo=TI_LO, TI_hi=TI_HI,
        model_cfg=model_cfg,
        step=int(full.model.step),
    )

    # ---- TI stats on full set ----
    ti_raw_all = full.model.apply_fn({'params': full.model.params["net"]}, Xn_all)
    a_all, b_all = ti_ab_from_raw(ti_raw_all)

    mean01 = kumar_mean01(a_all, b_all)
    var01  = kumar_var01(a_all, b_all)
    std01  = jnp.sqrt(var01)
    p10_01 = jnp.clip((1.0 - (1.0 - 0.10)**(1.0/b_all))**(1.0/a_all), 0, 1)
    p50_01 = jnp.clip((1.0 - (1.0 - 0.50)**(1.0/b_all))**(1.0/a_all), 0, 1)
    p90_01 = jnp.clip((1.0 - (1.0 - 0.90)**(1.0/b_all))**(1.0/a_all), 0, 1)

    mean_ti = map01_to_ti(mean01, TI_LO, TI_HI)
    std_ti  = (TI_HI - TI_LO) * std01
    p10_ti  = map01_to_ti(p10_01, TI_LO, TI_HI)
    p50_ti  = map01_to_ti(p50_01, TI_LO, TI_HI)
    p90_ti  = map01_to_ti(p90_01, TI_LO, TI_HI)

    print("TI mean  -> min {:.3f}  med {:.3f}  max {:.3f}".format(
        float(jnp.min(mean_ti)), float(jnp.median(mean_ti)), float(jnp.max(mean_ti))))
    print("TI std   -> min {:.3f}  med {:.3f}  max {:.3f}".format(
        float(jnp.min(std_ti)),  float(jnp.median(std_ti)),  float(jnp.max(std_ti))))
    print("TI p10   -> min {:.3f}  med {:.3f}  max {:.3f}".format(
        float(jnp.min(p10_ti)),  float(jnp.median(p10_ti)),  float(jnp.max(p10_ti))))
    print("TI p50   -> min {:.3f}  med {:.3f}  max {:.3f}".format(
        float(jnp.min(p50_ti)),  float(jnp.median(p50_ti)),  float(jnp.max(p50_ti))))
    print("TI p90   -> min {:.3f}  med {:.3f}  max {:.3f}".format(
        float(jnp.min(p90_ti)),  float(jnp.median(p90_ti)),  float(jnp.max(p90_ti))))
    return full


if __name__ == "__main__":
    main() 

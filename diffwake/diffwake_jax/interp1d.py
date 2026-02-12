import jax.numpy as jnp


def interp1d(x, y, xnew):
    indices = jnp.searchsorted(x, xnew, side="right")
    indices = jnp.clip(indices, 1, x.shape[0] - 1)

    x0 = x[indices - 1]
    x1 = x[indices]
    y0 = y[indices - 1]
    y1 = y[indices]

    slope = (y1 - y0) / (x1 - x0 + 1e-16)
    ynew = y0 + slope * (xnew - x0)

    # Clamp xnew outside x-range
    ynew = jnp.where(xnew < x[0], y[0], ynew)
    ynew = jnp.where(xnew > x[-1], y[-1], ynew)

    return ynew


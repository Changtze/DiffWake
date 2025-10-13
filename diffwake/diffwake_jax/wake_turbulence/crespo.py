import jax
import jax.numpy as jnp
from flax import struct
from jax import lax


@struct.dataclass
class CrespoHernandez:
    initial:    float = struct.field(pytree_node=False, default=0.1)
    constant:   float = struct.field(pytree_node=False, default=0.9)
    ai:         float = struct.field(pytree_node=False, default=0.8)
    downstream: float = struct.field(pytree_node=False, default=-0.32)

    def prepare_function(self):                       # noqa: D401
        """Nothing to prepare – kept for API compatibility."""
        return {}


    def __call__(self,
                 ambient_TI:    jnp.ndarray,   # (B,1,1,1) 
                 x:             jnp.ndarray,   # (B,T,H,W)
                 x_i:           jnp.ndarray,   # (B,1,1,1)
                 rotor_diameter: float,        # scalar
                 axial_induction: jnp.ndarray  # (B,1,1,1) 
                 ) -> jnp.ndarray:

        delta_x      = x - x_i

        delta_x_safe = lax.select(delta_x > 0.1, delta_x, jnp.ones_like(delta_x))

        ti_add = ( self.constant
                 * jnp.power(axial_induction, self.ai)
                 * jnp.power(ambient_TI,      self.initial)
                 * jnp.power(delta_x_safe / rotor_diameter,
                             self.downstream) )

        ti_add = ti_add * (delta_x > -0.1).astype(ti_add.dtype)

        ti_add = jnp.nan_to_num(ti_add, nan=0.0, posinf=0.0)
        return ti_add


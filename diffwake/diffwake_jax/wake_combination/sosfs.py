import jax.numpy as jnp
from typing import Any, Dict
from flax import struct
from typing import Tuple


# i dont think this should be a dataclass?
#@struct.dataclass
class SOSFS:
    def prepare_function(self) -> Dict[str, Any]:
        """
        Nothing to do here, just for API compatibility
        """
        return {}

    def __call__(self, wake_field: jnp.ndarray,
                 velocity_field: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        """
        Combines the base flow field with the velocity deficits using sum of squares.

        Args:
            wake_field (jnp.ndarray): the wake to apply to the base flow field.
            velocity_field (jnp.ndarray): base flow field.
        Returns:
            jnp.ndarray: Resulting flow field after applying the wake to the base
            NOTE: according to JAX documentation, jnp.hypot is a more numerically stable way of computing
            jnp.sqrt(x1 ** 2 + x2 ** 2)
        """

        return jnp.hypot(wake_field, velocity_field), None
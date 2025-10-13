import jax.numpy as jnp
from typing import Any, Dict

class SOSFS:

    def __init__(self):
        pass  
    def prepare_function(self) -> Dict[str, Any]:
        return {}

    def __call__(self, wake_field: jnp.ndarray, velocity_field: jnp.ndarray) -> jnp.ndarray:

        return jnp.sqrt(wake_field**2 + velocity_field**2)
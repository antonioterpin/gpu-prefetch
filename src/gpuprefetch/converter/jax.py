"""Utils for converting between cupy and jax arrays."""
import jax.numpy as jnp
import cupy as cp
from jax import dlpack


def cupy_to_jax(cp_arr: cp.ndarray) -> jnp.ndarray:
    """Convert a cupy array to a jax array.

    TODO: This is at the moment a bottleneck, needs to be optimized.
    """
    return jnp.from_dlpack(cp_arr, copy=False)


def jax_to_cupy(jax_arr: jnp.ndarray) -> cp.ndarray:
    """Convert a jax array to a cupy array with zero-copy.

    Args:
        jax_arr (jnp.ndarray): The jax array to convert.

    Returns:
        cp.ndarray: The converted cupy array.
    """
    return cp.from_dlpack(dlpack.to_dlpack(jax_arr))

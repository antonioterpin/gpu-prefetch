"""Module for converting between different array types."""

# Try to import jax related utils if jax is installed
try:
    from .jax import cupy_to_jax, jax_to_cupy

    __all__ = ["cupy_to_jax", "jax_to_cupy"]
except ImportError:
    pass

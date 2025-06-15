import timeit

import cupy as cp
import pytest

JAX_AVAILABLE = True
try:
    import jax
    import jax.numpy as jnp
    from gpuprefetch.converter import cupy_to_jax, jax_to_cupy

    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None

    class JnpFallback:
        int32 = None
        float32 = None
        float64 = None

    jnp = JnpFallback()
    JAX_AVAILABLE = False


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="jax is not available, skipping jax-related tests.",
)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((100_000_000,), jnp.int32),
        ((100_000_000,), jnp.float32),
        ((100_000_000,), jnp.float64),
    ],
)
def test_jax_to_cupy_timing(shape, dtype):
    jax_arr = jnp.arange(jnp.prod(jnp.array(shape)), dtype=dtype).reshape(shape)

    # warm-up
    _ = jax_to_cupy(jax_arr).data.ptr

    runs = 1_000
    threshold_ms = 0.006

    # measure total seconds for `runs` calls
    t_sec = timeit.timeit(lambda: jax_to_cupy(jax_arr).data.ptr, number=runs)

    avg_ms = (t_sec / runs) * 1_000
    assert (
        avg_ms < threshold_ms
    ), f"Average transfer time {avg_ms:.4f} ms exceeded {threshold_ms} ms"


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="jax is not available, skipping jax-related tests.",
)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((100_000_000,), jnp.int32),
        ((100_000_000,), jnp.float32),
        ((100_000_000,), jnp.float64),
    ],
)
def test_jax_to_cupy_no_dlpack_timing(shape, dtype):
    jax_arr = jnp.arange(jnp.prod(jnp.array(shape)), dtype=dtype).reshape(shape)

    # warm-up
    _ = jax_to_cupy(jax_arr).data.ptr

    runs = 1_000
    threshold_ms = 0.009

    t_sec = timeit.timeit(lambda: cp.array(jax_arr).data.ptr, number=runs)
    avg_ms = (t_sec / runs) * 1_000
    assert avg_ms > threshold_ms, (
        f"Average direct cp.array time {avg_ms:.4f} ms did not exceed "
        f"{threshold_ms} ms; you could skip the DLPack path."
    )


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="jax is not available, skipping jax-related tests.",
)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((100_000_000,), cp.int32),
        ((100_000_000,), cp.float32),
        ((100_000_000,), cp.float64),
    ],
)
def test_cupy_to_jax_no_dlpack_timing(shape, dtype):
    cp_arr = cp.arange(cp.prod(cp.array(shape)), dtype=dtype).reshape(shape)

    # warm-up
    _ = cupy_to_jax(cp_arr).block_until_ready()

    runs = 1_000
    threshold_ms = 0.055

    t_sec = timeit.timeit(lambda: cupy_to_jax(cp_arr).block_until_ready(), number=runs)
    avg_ms = (t_sec / runs) * 1_000
    assert (
        avg_ms < threshold_ms
    ), f"Average transfer time {avg_ms:.4f} ms exceeds {threshold_ms} ms"

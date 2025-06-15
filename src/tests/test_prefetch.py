import time
import numpy as np
import pytest

from gpuprefetch import Prefetcher


# Simple loader that simulates disk I/O by sleeping
def slow_loader(shape=(100_000,), sleep_time=0.1):
    # Simulate loading data from disk by sleeping for `sleep_time` seconds.
    # Returns a numpy array of the given shape.
    time.sleep(sleep_time)
    return np.ones(shape, dtype=np.float32)


class SeqLoader:
    def __init__(self, dtype, shape=(10,)):
        self.dtype = dtype
        self.shape = shape
        self.seq = 1

    def __call__(self):
        # draw from the internal RNG
        val = np.ones(self.shape, dtype=self.dtype) * self.seq
        self.seq += 1
        return val


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("capacity", [1, 2, 5, 10])
@pytest.mark.parametrize("N", [1, 2, 5, 10])
@pytest.mark.parametrize("shape", [(10,), (100, 100), (1000, 1000)])
def test_prefetcher_correctness(dtype, capacity, N, shape):
    seq_loader = SeqLoader(dtype, shape=shape)
    with Prefetcher(
        loader=SeqLoader(dtype, shape=shape),
        capacity=capacity,
        dtype=dtype,
        shape=shape,
        device="cuda:0",
        post=None,
        timeout=2,
    ) as prefetch:
        # Collect several batches
        ref = seq_loader()
        res = next(prefetch).get()
        np.testing.assert_allclose(res, ref, atol=1e-4)


@pytest.mark.parametrize("capacity", [100])
def test_prefetcher_performance(capacity):
    # Baseline: direct loading
    N = 1000
    baseline_time = 0.1 * N

    # Prefetched loading
    with Prefetcher(
        loader=slow_loader,
        capacity=capacity,
        dtype=np.float32,
        shape=(100_000,),
        device="cuda:0",
        post=None,
        nworkers=32,
    ) as prefetch:
        start_pf = time.time()
        for _ in range(N):
            next(prefetch)
        pf_time = time.time() - start_pf

    # We expect prefetching to be at least 20x faster than baseline
    assert pf_time < baseline_time * 0.05, (
        f"Prefetching with capacity={capacity} took {pf_time:.3f}s, "
        f"which is not significantly faster than baseline {baseline_time:.3f}s."
    )

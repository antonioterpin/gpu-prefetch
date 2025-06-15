# GPU prefetching
A minimal package for GPU prefetching from disk.

## Installation 🚀
```bash
pip install --upgrade pip uv
uv pip install gpuprefetch
```

## Usage 🔥
The snippet below show how to easily instantiate our prefetching.
```python
from gpuprefetch import Prefetcher

# By default, we do not install jax,
# you need to install it separately if you want to use our converter.
from gpuprefetch.converter import cupy_to_jax

def my_loader():
    # Replace with your real disk I/O
    return np.random.rand(64, 64, 3).astype(np.float32)

with Prefetcher(
    loader=my_loader,   # The callable loading the data
                        # (e.g., from disk)
    dtype=cp.float32,   # Type of data.
    shape=(64,64,3),    # Shape of the data for pre-allocation.
    capacity=100,       # Buffer size.
                        # The larger, the higher the memory
                        # consumption but the lower the latency.
    device="cuda:0",    # Where you want your data to be loaded.
                        # We currently support only cuda devices.
    post=cupy_to_jax,   # By default, we return a cupy array.
                        # You can use one of our efficient converters
                        # on the returned value, pass it as `post`
                        # argument, or pass a custom callable.
                        # We use it as: return post(data)
    nworkers=32,        # We spawn nworkers processes to concurrently
                        # fill the queue.
) as loader_with_prefetching:
    # Inside this block, the workers are active and they are safely
    # killed on exit.
    # You can access the pre-fetched data with:
    prefetched_data = next(loader_with_prefetching)
```

## Citation ☕️
If this was useful in your research software, you can cite this work as:
```bibtex
@software{terpin2025gpuprefetching,
  author       = {Terpin, Antonio},
  title        = {A minimal GPU prefetching package.},
  year         = {2025},
  version      = {1.0.0},
  url          = {https://github.com/antonioterpin/gpu-prefetch},
}
```

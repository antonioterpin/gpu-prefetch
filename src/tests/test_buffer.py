import pytest
import cupy as cp
from gpuprefetch.memory.buffer import SPSCBuffer, BufferClosedError, BufferFullError


@pytest.fixture
def buffer():
    # Initialize buffer with capacity 2, element shape (2,), dtype float32
    buf = SPSCBuffer(capacity=2, dtype=cp.float32, shape=(2,))
    buf.open()
    yield buf
    buf.cleanup()


def test_capacity(buffer):
    assert buffer.capacity == 2


def test_put_get(buffer):
    arr = cp.array([1.0, 2.0], dtype=cp.float32)
    buffer.put(arr, timeout=0.1)
    result = buffer.get(timeout=0.1)
    assert result is not None
    assert cp.allclose(result, arr)


def test_buffer_full(buffer):
    arr = cp.zeros((2,), dtype=cp.float32)
    # Fill buffer
    buffer.put(arr, timeout=0.1)
    buffer.put(arr, timeout=0.1)
    # Should raise after timeout
    with pytest.raises(BufferFullError):
        buffer.put(arr, timeout=0.1)


def test_get_timeout(buffer):
    # Nothing in buffer, should return None
    result = buffer.get(timeout=0.1)
    assert result is None


def test_type_mismatch(buffer):
    wrong_shape = cp.zeros((3,), dtype=cp.float32)
    with pytest.raises(TypeError):
        buffer.put(wrong_shape)

    wrong_dtype = cp.zeros((2,), dtype=cp.int32)
    with pytest.raises(TypeError):
        buffer.put(wrong_dtype)


def test_wraparound():
    a1 = cp.array([1.0, 1.0], dtype=cp.float32)
    a2 = cp.array([2.0, 2.0], dtype=cp.float32)
    a3 = cp.array([3.0, 3.0], dtype=cp.float32)
    buf = SPSCBuffer(capacity=2, dtype=cp.float32, shape=(2,))
    buf.open()

    print("Putting items in buffer")
    buf.put(a1.copy(), timeout=1)
    buf.put(a2.copy(), timeout=1)

    # Read one to advance tail
    val = buf.get(timeout=1)  # Remove the extra .get().copy()
    print(f"First get: {val}")

    # Now head wraps
    buf.put(a3.copy(), timeout=1)
    val2 = buf.get(timeout=1)  # Remove the extra .get().copy()
    print(f"Second get: {val2}")

    val3 = buf.get(timeout=1)  # Remove the extra .get().copy()
    print(f"Third get: {val3}")

    buf.cleanup()

    # Items should come in order: a1, a2, a3
    assert cp.allclose(val, a1)
    assert cp.allclose(val2, a2, atol=1e-4)
    assert cp.allclose(val3, a3, atol=1e-4)


def test_cleanup_closes_buffer(buffer):
    buffer.cleanup()
    arr = cp.zeros((2,), dtype=cp.float32)
    with pytest.raises(BufferClosedError):
        buffer.put(arr, timeout=0.1)
    with pytest.raises(BufferClosedError):
        buffer.get(timeout=0.1)

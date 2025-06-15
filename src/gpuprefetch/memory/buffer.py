"""Module for a process-safe ring buffer using shared memory."""

import ctypes
import os
import cupy as cp
from multiprocessing import get_context
from typing import Optional, Tuple, Any

from .sharray import CupySharray


class BufferClosedError(Exception):
    """Custom exception for broken pipe errors in the buffer."""

    def __init__(self, message: str):
        """Initialize the BufferClosedError with a message.

        Args:
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.message = message


class BufferFullError(Exception):
    """Custom exception for buffer full errors."""

    def __init__(self, message: str):
        """Initialize the BufferFullError with a message.

        Args:
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.message = message


class SPSCBuffer:
    """Single-producer, single-consumer ring buffer using shared memory and semaphores.

    The producer (writer) calls put(item, timeout=None),
        blocking if the buffer is full or until timeout expires.
    The consumer (reader) calls get(timeout=None),
        blocking until an item is available or timeout expires.
    """

    def __init__(self, capacity: int, dtype: Any, shape: Tuple[int, ...], ctx=None):
        """Initialize the SPSCBuffer.

        Args:
            capacity (int): The maximum number of items the buffer can hold.
            dtype (Any): The data type of the items in the buffer.
            shape (Tuple[int, ...]): The shape of the items in the buffer.
            ctx: Optional context for multiprocessing (default is None).
        """
        self._ctx = ctx or get_context("spawn")
        self._capacity = capacity
        self._shape = shape
        self._dtype = dtype
        self._owner_pid = os.getpid()

        # Shared pointers to CupySharray segments
        self._buffer_ptr = None
        # Local views: producer writes to _in_buffer; consumer reads from _out_buffer
        self._in_buffer = None
        self._out_buffer = None

        # Pointers
        # Head pointer (locked) for multiple producers
        self._head = self._ctx.Value(ctypes.c_int32, 0)  # next write index
        self._head_lock = self._ctx.Lock()
        self._tail = self._ctx.RawValue(ctypes.c_int32, 0)  # next read index

        # Semaphores: track free slots and filled items
        self._space_avail = self._ctx.Semaphore(capacity)
        self._items_avail = self._ctx.Semaphore(0)

        # Allocate and link shared-memory arrays
        self._allocate_buffer()

    def _allocate_buffer(self):
        # Create Cupy arrays and wrap them for shared memory
        self.arr = cp.zeros((self._capacity,) + self._shape, dtype=self._dtype)
        self._buffer_ptr = CupySharray.from_array(self.arr)

    def open(self, mode: str = "both"):
        """Open the shared-memory buffers in this process (producer or consumer).

        If mode=="producer", only opens the buffers for writing.
        If mode=="consumer", only opens the buffers for reading.
        Default "both" opens buffers for both reading and writing.
        """
        if self._buffer_ptr is None:
            raise BufferClosedError("Buffer has been destroyed")

        if mode == "producer" or mode == "both":
            self._in_buffer = self._buffer_ptr.open()

        if mode == "consumer" or mode == "both":
            self._out_buffer = self._buffer_ptr.open()

    @property
    def capacity(self) -> int:
        """Return the capacity of the buffer.

        Returns:
            int: The maximum number of items the buffer can hold.
        """
        return self._capacity

    def is_closed(self) -> bool:
        """Check if the buffer is closed.

        Returns:
            bool: True if the buffer is closed, False otherwise.
        """
        return self._buffer_ptr is None

    def put(self, item: cp.ndarray, timeout: Optional[float] = None) -> None:
        """Write a single item into the buffer, blocking if full.

        Args:
            item (cp.ndarray): The item to write into the buffer.
            timeout (Optional[float]): Maximum time to wait for space in seconds.
        """
        # Validate shape/dtype
        if item.shape != self._shape or item.dtype != self._dtype:
            raise TypeError(
                f"Expected array of shape {self._shape} and "
                f"dtype {self._dtype}, got {item.shape} and {item.dtype}"
            )

        if self.is_closed():
            raise BufferClosedError("Buffer has been closed")

        # Check if buffers are opened for writing
        if self._in_buffer is None:
            raise BufferClosedError(
                "Buffer not opened for writing. "
                "Call open('producer') or open('both') first."
            )

        # Wait for a free slot
        if not self._space_avail.acquire(timeout=timeout):
            raise BufferFullError("Buffer is full and timeout expired")

        # Reserve exactly one slot and advance HEAD once
        with self._head_lock:
            idx = self._head.value
            self._head.value = (idx + 1) % self._capacity

        # Copy into that slot using direct assignment
        self._in_buffer[idx, ...] = item

        # Signal that an item is now available
        self._items_avail.release()

    def get(self, timeout: Optional[float] = None) -> Optional[cp.ndarray]:
        """Read a single item from the buffer.

        Args:
            timeout (Optional[float]): Maximum time to wait for an item in seconds.

        Returns:
            Optional[cp.ndarray]: The item read from the buffer,
                or None if timeout expires.
        """
        if self.is_closed():
            raise BufferClosedError("Buffer has been closed")

        # Wait for item availability
        if not self._items_avail.acquire(timeout=timeout):
            return None

        idx = self._tail.value

        # Correctly copy the data
        buf = self._out_buffer[idx, ...].copy()

        # Immediately advance the tail
        self._tail.value = (idx + 1) % self._capacity

        # Signal space availability after reading
        self._space_avail.release()

        return buf

    def close(self):
        """Close this process's references to the shared buffers."""
        self._in_buffer = None
        self._out_buffer = None

    def unlink(self):
        """Unlink and destroy shared-memory segments (only owner)."""
        self._buffer_ptr = None

    def cleanup(self):
        """Close and unlink the shared memory."""
        self.close()
        self.unlink()

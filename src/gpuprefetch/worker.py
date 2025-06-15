"""Worker module."""

import os
import abc
from typing import Callable
from multiprocessing import get_context
import cupy as cp
from .memory.buffer import SPSCBuffer, BufferFullError

ctx = get_context(os.getenv("PREFETCH_MP_CONTEXT", "spawn"))
_ctx = ctx
_SpawnThread = _ctx.Process
_SpawnEvent = _ctx.Event


class Worker(_SpawnThread):
    """A worker node that loads data from file into the shared buffer.

    Calling `request_stop()` signals the node to finish its current iteration
    and then exit cleanly.
    """

    def __init__(
        self,
        loader: Callable[[], cp.ndarray],
        buffer: SPSCBuffer,
        device: str = "cuda:0",
        timeout: float = 0.1,
    ):
        """Initialize a Worker.

        Args:
            loader (Callable[[], cp.ndarray]): A callable that returns a batch of data.
            buffer (SPSCBuffer): The shared buffer to which the worker will write data.
            device (str): The device to which the data should be prefetched
                (e.g., 'cuda:0').
            timeout (float): The maximum time to wait for a the queue to empty.
                After timeout, the worker checks if it should stop.
        """
        abc.ABC.__init__(self)
        _SpawnThread.__init__(self, name="GPU prefetching worker")

        self._stop_event = _SpawnEvent()
        self._buffer = buffer
        self._loader = loader
        self._timeout = timeout
        self._device = device

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")

    def request_stop(self):
        """Signal the node to stop after completing the current iteration."""
        self._stop_event.set()

    def run(self):
        """Run the node process, spinning until stopped."""
        device_id = int(self._device.split(":", 1)[1])
        print(device_id)
        with cp.cuda.Device(device_id):
            # Attach only the producer buffers
            self._buffer.open(mode="producer")

            while not self._stop_event.is_set():
                try:
                    data = cp.asarray(self._loader())
                    self._buffer.put(data, timeout=self._timeout)
                except BufferFullError:
                    print(f"Buffer full in '{self._name}', waiting for space...")
                    pass
                except Exception as e:
                    print(f"Error in '{self._name}' during loop(): {e}")

        # clean up our producer attachments
        self._buffer.cleanup()
        self.close()

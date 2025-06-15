"""Prefetcher for loading data in numpy and to GPU."""
from typing import Callable, Any, Tuple, Union, Optional
import cupy as cp
from .worker import Worker, ctx
from .memory import SPSCBuffer


class NoDataException(Exception):
    """Custom exception for no data in the buffer after timeout."""

    def __init__(self):
        """Initialize the BufferClosedError with a message.

        Args:
            message (str): The error message to be displayed.
        """
        super().__init__("No data available in the buffer after timeout.")


class Prefetcher:
    """Context manager to manage the lifecycle the prefetcher.

    Usage:
        with Prefetcher(loader, capacity, device, nworkers) as prefetcher:
            # Within this block, nworkers will be used to prefetch data
            mydata_on_device = next(prefetcher)
    """

    def __init__(
        self,
        loader: Callable[[], cp.ndarray],
        capacity: int,
        dtype: Any,
        shape: Tuple[int, ...],
        device: str,
        post: Callable[[cp.ndarray], Any] = None,
        timeout: Optional[float] = None,
        nworkers: int = 1,
    ):
        """Initialize the Prefetcher.

        Args:
            loader (Callable[[], cp.ndarray]): A callable that returns a batch of data.
            capacity (int): The number of batches to prefetch.
            dtype (Any): The data type of the items in the buffer.
            shape (Tuple[int, ...]): The shape of the items in the buffer.
            device (str):
                The device to which the data should be prefetched (e.g., 'cuda:0').
            post (Callable[[cp.ndarray], Any], optional):
                A function to apply to the data after it has been prefetched.
                Defaults to identity function (returns the data unchanged).
            timeout (float): The maximum time to wait for the loading operation.
                After timeout,
                - the worker will check if it should stop,
                - if not, it will continue to wait for the queue to empty;
                - the reader will raise a NoDataException.
                Thus, set timeout large enough so that the worker can load the data
                confortably, but not too large to cause delays when things go wrong.
            nworkers (int): The number of worker threads to use for prefetching.
        """
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if not isinstance(device, str):
            raise TypeError(
                "device must be a string representing the device (e.g., 'cuda:0')."
            )
        if not callable(loader):
            raise TypeError("loader must be a callable that returns a batch of data.")
        if post is not None and not callable(post):
            raise TypeError("post must be a callable that processes the data.")

        self._dtype = dtype
        self._shape = shape
        self._capacity = capacity
        self._device = device
        self._post = post if post is not None else lambda x: x
        self._timeout = timeout
        self._loader = loader
        self._nworkers = nworkers

    def __next__(self) -> Union[cp.array, Any]:
        """Get the next batch of data from the prefetcher.

        Returns:
            cp.ndarray: The next batch of data prefetched to the specified device.

        Raises:
            NoDataException: If no data is available in the buffer after the timeout.
            BufferClosedError: If the buffer is closed and no data is available.
        """
        # This method should be implemented in subclasses
        data = self._buffer.get(timeout=self._timeout)
        if data is None:
            raise NoDataException()
        return self._post(data)

    def __enter__(self):
        """Start the worker for prefetching."""
        # Prepare the buffer
        self._buffer = SPSCBuffer(
            capacity=self._capacity,
            dtype=self._dtype,
            shape=self._shape,
            ctx=ctx,
        )
        self._buffer.open("consumer")

        # Prepare the workers
        self._workers = [
            Worker(
                loader=self._loader,
                buffer=self._buffer,
                timeout=self._timeout,
                device=self._device,
            )
            for _ in range(self._nworkers)
        ]

        # Start all workers
        for worker in self._workers:
            worker.start()

        # Return self so that user can call next(prefetcher)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Request all workers to stop and clean up resources.

        Args:
            exc_type: The type of exception raised, if any.
            exc_val: The value of the exception raised, if any.
            exc_tb: The traceback object, if an exception was raised.
        """
        # Ask the worker to stop
        try:
            for worker in self._workers:
                worker.request_stop()
            for worker in self._workers:
                worker.join()
        except Exception:
            pass

        # Clean up resources
        self._buffer.cleanup()

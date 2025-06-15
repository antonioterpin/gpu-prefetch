"""Fixed CupySharray with correct memory size calculation."""
from dataclasses import dataclass
from typing import Tuple
import cupy as cp
from cupy.cuda import MemoryPointer, UnownedMemory


@dataclass
class CupySharray:
    """Serialized representation of a CuPy array."""

    ptr: int
    ipc_handle: bytes
    shape: Tuple[int, ...]
    dtype: cp.dtype
    strides: Tuple[int, ...]
    device: int

    @classmethod
    def from_array(cls, arr_cp: cp.ndarray) -> "CupySharray":
        """Create a CupySharray from a CuPy array.

        Args:
            arr_cp (cp.ndarray): The CuPy array to serialize.

        Returns:
            CupySharray: An instance containing serialized data.
        """
        ptr = int(arr_cp.data.ptr)
        ipc_handle = cp.cuda.runtime.ipcGetMemHandle(ptr)
        return cls(
            ptr=ptr,
            ipc_handle=ipc_handle,
            shape=arr_cp.shape,
            dtype=arr_cp.dtype,
            strides=arr_cp.strides,
            device=arr_cp.device.id,
        )

    def open(self) -> cp.ndarray:
        """Open the shared CuPy array and return the underlying data.

        This method reconstructs a CuPy array from the serialized data,
        including the IPC handle, shape, dtype, and strides.

        Returns:
            cp.ndarray: The reconstructed CuPy ndarray.
        """
        # Switch to the correct device/context
        with cp.cuda.Device(self.device):
            try:
                # Try opening IPC handle (for cross-process)
                ptr = cp.cuda.runtime.ipcOpenMemHandle(
                    self.ipc_handle,
                    cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess,
                )
            except cp.cuda.runtime.CUDARuntimeError:
                # Fallback: same-context, raw pointer
                ptr = self.ptr

            # Use the total number of elements multiplied by itemsize
            total_elements = 1
            for dim in self.shape:
                total_elements *= dim
            size = int(cp.dtype(self.dtype).itemsize * total_elements)

            # Wrap in UnownedMemory + MemoryPointer
            mem = UnownedMemory(ptr, size, owner=None)
            mp = MemoryPointer(mem, 0)

            return cp.ndarray(
                self.shape, dtype=self.dtype, memptr=mp, strides=self.strides
            )

    @classmethod
    def copy_to(cls, a: cp.ndarray, b: cp.ndarray) -> None:
        """Copy a to b (possibly efficiently).

        Args:
            a (cp.ndarray): Source array to copy from.
            b (cp.ndarray): Destination array to copy to.
        """
        b[:] = a

# deflate.py
"""Defines PulsarInflater class: instances copy a numpy array to shared memory,
and (after pickling) will reinflate to a numpy array that refers to the shared
data.
"""

from multiprocessing import shared_memory, resource_tracker
import numpy as np

class memmap(np.ndarray):
    def __del__(self):
        if hasattr(self, "shm"):
            self.shm.close()

        super().__del__()

class PulsarInflater:
    def __init__(self, array):
        self.dtype, self.shape, self.nbytes = array.dtype, array.shape, array.nbytes

        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        self.shmname = shm.name

        # shm.buf[:array.nbytes] = array.view(dtype='uint8').flatten()

        b = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        b[:] = array[:]

        resource_tracker.unregister(shm._name, "shared_memory")

    def inflate(self):
        shm = shared_memory.SharedMemory(self.shmname)

        # ret = np.array(shm.buf[:self.nbytes], copy=False).view(dtype=self.dtype).reshape(self.shape).view(memmap)

        c = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf).view(memmap)
        c.shm = shm

        return c

    def destroy(self):
        shm = shared_memory.SharedMemory(self.shmname)
        shm.unlink()

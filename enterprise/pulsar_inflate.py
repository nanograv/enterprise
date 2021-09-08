# pulsar_inflate.py
"""Defines PulsarInflater class: instances copy a numpy array to shared memory,
and (after pickling) will reinflate to a numpy array that refers to the shared
data.
"""

import numpy as np

try:
    from multiprocessing import shared_memory, resource_tracker
except:
    # shared_memory unavailable in Python < 3.8
    pass


class memmap(np.ndarray):
    def __del__(self):
        if self.base is None and hasattr(self, "shm"):
            self.shm.close()


# lifecycle of shared pulsar arrays:
# - begin life as numpy arrays in Pulsar object
# - upon psr.deflate(), replaced by PulsarInflater objects
#   - these objects save the array metadata, create a SharedMemory buffer, copy the arrays into it
#   - the PulsarInflater objects cannot be used as arrays until re-inflated
# - upon psr.inflate, the PulsarInflater objects are replaced with ndarray views of the SharedMemory buffers
#   - the views are special memmap objects that hold a reference to the SharedMemory, and close it on destruction
# - upon psr.destroy, the SharedMemory objects are unlinked and the arrays become unusable
# - standard usage requires 3+ processes:
#   - a creator, who calls deflate then pickle
#   - one or more users, who unpickle then inflate
#   - a destroyer, who unpickles then destroys


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

        c = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf).view(memmap)
        c.shm = shm

        return c

    def destroy(self):
        shm = shared_memory.SharedMemory(self.shmname)
        shm.unlink()

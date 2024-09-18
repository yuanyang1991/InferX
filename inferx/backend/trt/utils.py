import ctypes
from typing import Optional, Union

import numpy as np
from cuda import cuda, cudart
import tensorrt as trt


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self, size: int, dtype: Optional[np.dtype] = None, name=None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def get_binding_mem(dtype: trt.DataType, tensor_shape) -> HostDeviceMem:
    size = trt.volume(tensor_shape)
    if trt.nptype(dtype):
        dtype = np.dtype(trt.nptype(dtype))
        binding_mem = HostDeviceMem(size, dtype)
    else:
        size = int(size * dtype.itemsize)
        binding_mem = HostDeviceMem(size)
    return binding_mem


def copy_mem_host_device(mem_info: HostDeviceMem, stream):
    cuda_call(cudart.cudaMemcpyAsync(mem_info.device, mem_info.host, mem_info.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))


def copy_mem_device_host(mem_info: HostDeviceMem, stream):
    cuda_call(cudart.cudaMemcpyAsync(mem_info.host, mem_info.device, mem_info.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))

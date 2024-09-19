import numpy as np
import tensorrt as trt
from cuda import cuda, cudart


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


def get_io(engine: trt.ICudaEngine, mode):
    nums = engine.num_io_tensors
    result = []
    for idx in range(nums):
        tensor_name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(tensor_name) == mode:
            result.append(tensor_name)
    return result


def copy_data_to_gpu(array: np.ndarray, stream):
    def is_contiguous(obj: np.ndarray):
        return obj.flags["C_CONTIGUOUS"]

    if not is_contiguous(array):
        array = np.ascontiguousarray(array)
    device_ptr = cuda_call(cudart.cudaMalloc(array.nbytes))
    host_ptr = array.ctypes.data
    cuda_call(cudart.cudaMemcpyAsync(device_ptr, host_ptr, array.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

    return device_ptr


def free_device_memory(device_ptr):
    cuda_call(cudart.cudaFree(device_ptr))


def synchronize_stream(stream):
    cuda_call(cudart.cudaStreamSynchronize(stream))

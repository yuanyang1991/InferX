from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

from .convert.dynamic_axis import DynamicAxisInfo
from .convert.onnx_tensorrt import ONNX2TensorRT
from .utils import cuda_call, get_io, \
    copy_data_to_gpu, free_device_memory, synchronize_stream
from ...base import ABSBackend


class OutputAllocator(trt.IOutputAllocator):

    def __init__(self):
        trt.IOutputAllocator.__init__(self)
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        if tensor_name in self.buffers:
            del self.buffers[tensor_name]
        device_ptr = cuda_call(cudart.cudaMalloc(size))
        self.buffers[tensor_name] = device_ptr
        return int(device_ptr)

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)


class TensorRTBackend(ABSBackend):
    """
    TensorRT推理后端
    """

    def __init__(self, model_path, engine_path, dynamic_axes: list[DynamicAxisInfo] = None):
        self._model_path = model_path
        self._engine_path = engine_path
        if not Path(self._engine_path).exists():
            ONNX2TensorRT(self._model_path, dynamic_axes).convert(engine_path=self._engine_path)

        self._logger = trt.Logger(trt.Logger.INFO)
        self._runtime = trt.Runtime(self._logger)
        with open(self._engine_path, 'rb') as f:
            self._engine: trt.ICudaEngine = self._runtime.deserialize_cuda_engine(f.read())
        self._context: trt.IExecutionContext = self._engine.create_execution_context()

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        stream = cuda_call(cudart.cudaStreamCreate())

        input_buffers = {}
        for tensor_name in get_io(self._engine, trt.TensorIOMode.INPUT):
            array = inputs[tensor_name]
            if self._engine.is_shape_inference_io(tensor_name):
                ptr = array.ctypes.data
            else:
                ptr = copy_data_to_gpu(array, stream)
                if tensor_name in input_buffers:
                    free_device_memory(input_buffers[tensor_name])
                input_buffers[tensor_name] = ptr
            if self._context.get_tensor_address(tensor_name) != ptr:
                self._context.set_tensor_address(tensor_name, ptr)
            shape = array.shape
            if self._context.get_tensor_shape(tensor_name) != shape:
                self._context.set_input_shape(tensor_name, shape)

        output_allocator = OutputAllocator()
        for tensor_name in get_io(self._engine, trt.TensorIOMode.OUTPUT):
            self._context.set_output_allocator(tensor_name, output_allocator)

        # 执行运算
        self._context.execute_async_v3(stream_handle=stream)

        # 从CUDA复制数据到主存
        outputs = {}
        for tensor_name in get_io(self._engine, trt.TensorIOMode.OUTPUT):
            dtype = trt.nptype(self._engine.get_tensor_dtype(tensor_name))
            shape = output_allocator.shapes[tensor_name]
            array: np.ndarray = np.empty(shape, dtype)
            array_ptr = array.ctypes.data
            ptr = output_allocator.buffers[tensor_name]
            cuda_call(cudart.cudaMemcpyAsync(array_ptr, ptr, array.nbytes,
                                             cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
            outputs[tensor_name] = array
        synchronize_stream(stream)
        for _, devPtr in input_buffers.items():
            free_device_memory(devPtr)
        for _, devPtr in output_allocator.buffers.items():
            free_device_memory(devPtr)
        return outputs

    def release(self):
        del self._engine
        del self._context

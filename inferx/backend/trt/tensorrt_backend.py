from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

from .convert.dynamic_axis import DynamicAxisInfo
from .convert.onnx_tensorrt import ONNX2TensorRT
from .utils import HostDeviceMem, get_binding_mem, copy_mem_host_device, copy_mem_device_host, cuda_call
from ...base import ABSBackend


@dataclass
class TensorInfo:
    tensor_name: str
    tensor_shape: tuple
    tensor_dtype: trt.DataType
    memo_info: HostDeviceMem


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

    def _allocate_buffers(self, inputs: dict[str, np.ndarray]):
        # TODO: 直接从numpy复制数据到device;直接从device复制数据到numpy中，避免中间转换
        input_tensors, output_tensors = [], []
        num_tensors = self._engine.num_io_tensors

        for idx in range(num_tensors):
            tensor_name = self._engine.get_tensor_name(idx)
            dtype: trt.DataType = self._engine.get_tensor_dtype(tensor_name)
            if self._engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                tensor_shape = inputs[tensor_name].shape  # 使用输入数据得到shape，避免动态轴造成的不便
                input_tensors.append(
                    TensorInfo(tensor_name=tensor_name,
                               tensor_shape=tensor_shape,
                               tensor_dtype=dtype,
                               memo_info=get_binding_mem(dtype, tensor_shape)))
            else:
                tensor_shape = self._engine.get_tensor_shape(tensor_name)  # 使用输出的shape，注意：此处不支持动态维度的输出
                output_tensors.append(
                    TensorInfo(tensor_name=tensor_name,
                               tensor_shape=tensor_shape,
                               tensor_dtype=dtype,
                               memo_info=get_binding_mem(dtype, tensor_shape)))

        return input_tensors, output_tensors

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        stream = cuda_call(cudart.cudaStreamCreate())
        input_tensors, output_tensors = self._allocate_buffers(inputs)

        # 设置输入地址，复制输入数据
        for input_tensor in input_tensors:
            self._context.set_input_shape(input_tensor.tensor_name, input_tensor.tensor_shape)
            self._context.set_tensor_address(input_tensor.tensor_name, input_tensor.memo_info.device)
            copy_mem_host_device(input_tensor.memo_info, stream)

        # 设置输出地址
        for output_tensor in output_tensors:
            self._context.set_tensor_address(output_tensor.tensor_name, output_tensor.memo_info.device)

        # 执行运算
        self._context.execute_async_v3(stream_handle=stream)

        # 从CUDA复制数据到主存
        [copy_mem_device_host(output_tensor.memo_info, stream) for output_tensor in output_tensors]
        cuda_call(cudart.cudaStreamSynchronize(stream))
        outputs = {
            output_tensor.tensor_name: output_tensor.memo_info.host.reshape(output_tensor.tensor_shape).copy()
            for output_tensor in output_tensors}

        # 释放内存
        for tensors in input_tensors + output_tensors:
            tensors.memo_info.free()

        return outputs

    def release(self):
        del self._engine
        del self._context

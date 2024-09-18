from enum import Enum
from .backend.trt.convert.dynamic_axis import DynamicAxisInfo


class Backend(Enum):
    TensorRT = 1
    Onnx = 2


class Device(Enum):
    CPU = 1
    GPU = 2


class Config:

    def __init__(self, model_path: str,
                 engine_path: str = None,
                 backend: Backend = Backend.TensorRT,
                 device: Device = Device.GPU,
                 dynamic_axes: list[DynamicAxisInfo] = None
                 ):
        """
        初始化框架的相关配置
        :param model_path:  onnx模型地址
        :param engine_path: tensorrt engine地址。如果使用ONNX后端推理，此参数可不传。
        :param device:      当使用ONNX后端时，指定使用CPU或者CUDA推理
        :param backend:      选取使用的推理后端
        :param dynamic_axes: 当选取TensorRT推理后端且需要模型转换时，用于配置模型的动态轴信息
        """

        self._model_path = model_path
        self._engine_path = engine_path
        self._device = device
        self._backend = backend
        self._dynamic_axes = dynamic_axes

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def engine_path(self) -> str:
        return self._engine_path

    @property
    def device(self) -> Device:
        return self._device

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def dynamic_axes(self) -> list[DynamicAxisInfo]:
        return self._dynamic_axes

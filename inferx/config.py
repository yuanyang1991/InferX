from enum import Enum
from typing import List, Optional

from .backend.trt.convert.dynamic_axis import DynamicAxisInfo


class Backend(Enum):
    TensorRT = 1
    Onnx = 2


class Device(Enum):
    CPU = "cpu"
    GPU = "cuda"


class Config:

    def __init__(self, model_path: str,
                 engine_path: Optional[str] = None,
                 backend: Backend = Backend.TensorRT,
                 devices: Optional[List[Device]] = None,
                 dynamic_axes: Optional[List[DynamicAxisInfo]] = None,
                 enable_log: bool = False
                 ):
        """
        初始化框架的相关配置
        :param model_path:  onnx模型地址
        :param engine_path: tensorrt engine地址。如果使用ONNX后端推理，此参数可不传。
        :param devices:      当使用ONNX后端时，指定使用CPU或者CUDA推理
        :param backend:      选取使用的推理后端
        :param dynamic_axes: 当选取TensorRT推理后端且需要模型转换时，用于配置模型的动态轴信息
        """

        self._model_path = model_path
        self._engine_path = engine_path
        self._devices = devices
        self._backend = backend
        self._dynamic_axes = dynamic_axes
        self._enable_log = enable_log

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def engine_path(self) -> Optional[str]:
        return self._engine_path

    @property
    def devices(self) -> Optional[List[Device]]:
        return self._devices

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def dynamic_axes(self) -> Optional[List[DynamicAxisInfo]]:
        return self._dynamic_axes

    @property
    def enable_log(self) -> bool:
        return self._enable_log

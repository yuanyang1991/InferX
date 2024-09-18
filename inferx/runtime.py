import numpy as np

from .backend.trt.tensorrt_backend import TensorRTBackend
from .backend.onx.onnx_backend import ONNXBackend
from .config import *


class Runtime:

    def __init__(self, config: Config):
        self._config: Config = config
        if not self._config.model_path:
            raise RuntimeError("model path cannot empty")
        if config.backend == Backend.TensorRT:
            self._backend = TensorRTBackend(config.model_path, config.engine_path, config.dynamic_axes)
        else:
            self._backend = ONNXBackend()

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._backend.run(inputs)

    def release(self):
        self._backend.release()

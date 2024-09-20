import numpy as np
from .config import *


class Runtime:

    def __init__(self, config: Config):
        self._config: Config = config
        if not self._config.model_path:
            raise RuntimeError("model path cannot empty")
        if config.backend == Backend.TensorRT:
            from .backend.trt.tensorrt_backend import TensorRTBackend
            self._backend = TensorRTBackend(config.model_path, config.engine_path, config.dynamic_axes)
        else:
            from .backend.onx.onnx_backend import ONNXBackend
            self._backend = ONNXBackend(config.devices, config.model_path)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._backend.run(inputs)

    def release(self):
        self._backend.release()

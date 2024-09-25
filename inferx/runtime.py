from typing import Dict

import numpy as np
from .config import *


class Runtime:

    def __init__(self, config: Config):
        self._config: Config = config
        if not self._config.model_path:
            raise RuntimeError("model path cannot empty")
        if config.backend == Backend.TensorRT:
            from .backend.trt.tensorrt_backend import TensorRTBackend
            self._backend = TensorRTBackend(config.model_path, config.engine_path, config.dynamic_axes,
                                            config.enable_log)
        else:
            from .backend.onx.onnx_backend import ONNXBackend
            self._backend = ONNXBackend(config.devices, config.model_path, config.enable_log)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self._backend.run(inputs)

    def release(self):
        self._backend.release()

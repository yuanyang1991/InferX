from typing import List, Dict

import numpy as np
import onnxruntime as onnxrt

from ...base import ABSBackend
from ...config import Device


class ONNXBackend(ABSBackend):

    def __init__(self, devices: List[Device], model_path: str, enable_log: bool = False):
        self._devices = devices
        self._model_path = model_path
        self._enable_log = enable_log
        support_providers = onnxrt.get_available_providers()
        providers = []
        for device in devices:
            if device in support_providers:
                providers.append(device)
            else:
                for ele in support_providers:
                    if device.value.lower() in ele.lower() or ele.lower() in device.value.lower():
                        providers.append(ele)
        if not providers:
            providers.append("CPUExecutionProvider")
        options = onnxrt.SessionOptions()
        if self._enable_log:
            options.log_severity_level = 0
        self._session = onnxrt.InferenceSession(model_path, providers=providers, sess_options=options)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inference_outputs = self._session.run(None, inputs)
        output_nodes = self._session.get_outputs()
        results = {}
        for node, out in zip(output_nodes, inference_outputs):
            results[node.name] = out
        return results

    def release(self):
        pass

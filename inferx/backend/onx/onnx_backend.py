import numpy as np

from ...base import ABSBackend


class ONNXBackend(ABSBackend):

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pass

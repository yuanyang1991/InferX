from abc import ABC, abstractmethod

import numpy as np


class ABSBackend(ABC):

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        执行推理
        :param inputs: key:输入名称 value:输入数据
        :return:       key:输出名称 value:输出数据
        """
        pass

    @abstractmethod
    def release(self):
        pass

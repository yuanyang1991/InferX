import time

import GPUtil
from threading import Thread


def get_memory_used():
    gpus = GPUtil.getGPUs()
    return gpus[0].memoryUsed


class DeviceMemoryStat(Thread):

    def __init__(self):
        super(DeviceMemoryStat, self).__init__()
        self._start_memory_used = 0
        self._max_memory_used = 0
        self.stopped = False

    def begin(self):
        self._start_memory_used = get_memory_used()
        self._max_memory_used = self._start_memory_used
        self.start()

    def run(self):
        while not self.stopped:
            current = get_memory_used()
            if current > self._max_memory_used:
                self._max_memory_used = current
            time.sleep(0.001)

    def end(self):
        self.stopped = True

    def get_max_memory_used(self):
        return self._max_memory_used - self._start_memory_used

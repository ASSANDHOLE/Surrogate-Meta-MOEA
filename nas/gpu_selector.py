from typing import List
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
import pynvml


class GpuSelector:
    def __init__(self, num_gpu: int,
                 gpu_ids: List[int] | None = None,
                 default_ram_per_task: float = 2.0):
        """
        default_ram_per_task: the default ram per task, in GB
        """
        self.num_gpu = num_gpu
        if gpu_ids is None:
            gpu_ids = list(range(num_gpu))
        self.gpu_ids = gpu_ids
        pynvml.nvmlInit()
        self.devs = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids}
        self.tasks = {gpu_id: [] for gpu_id in gpu_ids}
        self.default_ram_per_task = default_ram_per_task

    def _has_enough_ram(self, gpu_id: int, estimate_ram: float) -> bool:
        """
        Check if the gpu has enough ram
        """
        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(self.devs[gpu_id])
        return info.free / 1024 ** 3 > estimate_ram

    def get_gpu(self, estimate_ram: float | None = None, lock: Lock = None) -> int:
        """
        Get the gpu with the least tasks and has enough ram
        """
        with lock:
            if estimate_ram is None:
                estimate_ram = self.default_ram_per_task
            sorted_gpu_ids = sorted(self.gpu_ids, key=lambda gi: len(self.tasks[gi]))
            for gpu_id in sorted_gpu_ids:
                if self._has_enough_ram(gpu_id, estimate_ram):
                    self.tasks[gpu_id].append(estimate_ram)
                    return gpu_id
        raise RuntimeError('No enough ram on any gpu')

    def return_gpu(self, gpu_id: int, estimate_ram: float | None = None, lock: Lock = None):
        """
        Return the gpu
        """
        with lock:
            if estimate_ram is None:
                estimate_ram = self.default_ram_per_task
            try:
                idx = self.tasks[gpu_id].index(estimate_ram)
                self.tasks[gpu_id].pop(idx)
            except (ValueError, IndexError):
                self.tasks[gpu_id].pop()


class GpuManager(BaseManager):
    def __init__(self):
        super().__init__()
        GpuManager.register('GpuSelector', GpuSelector)
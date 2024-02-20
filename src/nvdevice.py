from dataclasses import dataclass
from typing import List
import pynvml

@dataclass
class NvDevice:
    device_id: int = 0
    name: str = ""
    compute: tuple = (0,0)
    mem: int = 0
    @property
    def major_cap(self):
        return self.compute[0]
    @property
    def minor_cap(self):
        return self.compute[1]

def get_devices(devices=None) -> List[NvDevice]:
    pynvml.nvmlInit()
    if devices is None:
        device_count = pynvml.nvmlDeviceGetCount()
        devices = range(device_count)

    ret = {}
    for i in devices:
        if isinstance(i, str):
            i = int(i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_name = pynvml.nvmlDeviceGetName(handle)
        compute = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = mem.total // (1024 ** 3)

        print(f"GPU {i}: {device_name}: {compute}")
        dev = NvDevice(device_id=i, name=device_name, compute=compute, mem=total)
        ret[i] = dev

    pynvml.nvmlShutdown()
    return ret

def has_same_capability(devices: List[NvDevice]) -> bool:
    assert len(devices) > 0
    cap = devices[0].major_cap
    for device in devices[1:]:
        if device.major_cap != cap:
            return False
    return True

def filter_by_major_cap(devices: List[NvDevice], idx=0):
    cap = devices[idx].major_cap
    return list(filter(lambda d: d.major_cap == cap, devices))
import dgtrt
import numpy as np

dgtrt.enable_request_storage()
a = np.arange(0, 10, dtype=np.int32)
idx = dgtrt.add_request_storage(a)
print(idx)
dgtrt.remove_request_storage(idx)

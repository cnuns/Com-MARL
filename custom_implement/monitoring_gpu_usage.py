from pynvml import *
import time
import torch

if torch.cuda.is_available():
    nvmlInit()
    
def monitor_gpu_memory():
    
    while 1:
        time.sleep(0.1)
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)

        #print(f'total    : {(info.total)/(10**9)}')
        #print(f'used     : {(info.used)/(10**9)}')
        print(f'free     : {(info.free)/(1024**3):.2f}')

def get_gpu_memory():
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    return f'{(info.free)/(1024**3):.2f}'

if __name__ == '__main__':
    get_gpu_memory()

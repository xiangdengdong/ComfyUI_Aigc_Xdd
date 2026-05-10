
import sys

def detect_gpus():
    gpus_found = []
    
    # 1. 检测 NVIDIA CUDA GPU (台式机/笔记本独显)
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"CUDA:{i}:{name}:{round(mem, 1)}")
                gpus_found.append(('cuda', i, name))
    except:
        pass
    
    # 2. 检测 AMD ROCm GPU
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            count = torch.hip.device_count()
            for i in range(count):
                name = torch.hip.get_device_name(i)
                props = torch.hip.get_device_properties(i)
                mem = props.total_memory / 1024**3 if hasattr(props, 'total_memory') else 0
                print(f"AMD:{i}:{name}:{round(mem, 1)}")
                gpus_found.append(('amd', i, name))
    except:
        pass
    
    # 3. 检测 Intel XPU (Arc 显卡)
    try:
        import torch
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            count = torch.xpu.device_count()
            for i in range(count):
                name = torch.xpu.get_device_name(i)
                props = torch.xpu.get_device_properties(i)
                mem = props.total_memory / 1024**3 if hasattr(props, 'total_memory') else 0
                print(f"XPU:{i}:{name}:{round(mem, 1)}")
                gpus_found.append(('xpu', i, name))
    except:
        pass
    
    # 4. 检测 Apple MPS (Mac M1/M2/M3)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"MPS:0:Apple Silicon GPU:0")
            gpus_found.append(('mps', 0, 'Apple Silicon'))
    except:
        pass
    
    # 5. 检测 DirectML (Windows 通用，支持 AMD/Intel/NVIDIA)
    try:
        import torch_directml
        count = torch_directml.device_count()
        for i in range(count):
            name = torch_directml.device_name(i)
            print(f"DML:{i}:{name}:0")
            gpus_found.append(('dml', i, name))
    except:
        pass
    
    if not gpus_found:
        print("NOGPU")

detect_gpus()

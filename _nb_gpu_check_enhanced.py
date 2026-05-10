import json
import sys
import signal

# 设置超时
def timeout_handler(signum, frame):
    raise TimeoutError("检测超时")

result = {"cuda_available": False, "cuda_version": None, "devices": [], "error": None, "pytorch_version": None}

try:
    # 设置 10 秒超时
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
    
    import torch
    result["pytorch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    
    if result["cuda_available"]:
        result["cuda_version"] = torch.version.cuda
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                result["devices"].append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            except Exception as dev_err:
                result["devices"].append({
                    "index": i,
                    "error": str(dev_err)
                })
    
    # 取消超时
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)
        
except ImportError as e:
    result["error"] = f"PyTorch 未安装: {str(e)}"
except TimeoutError:
    result["error"] = "CUDA 检测超时（可能是驱动问题）"
except Exception as e:
    result["error"] = f"检测失败: {str(e)}"

print(json.dumps(result, ensure_ascii=False))

import sys
import platform
import os

try:
    import torch
    is_torch = True
except ImportError:
    is_torch = False

# 获取真实的 Windows 版本
def get_windows_version():
    try:
        if sys.platform == 'win32':
            ver = sys.getwindowsversion()
            # Windows 11 的 build 号从 22000 开始
            if ver.major == 10 and ver.build >= 22000:
                return f"Windows 11 (Build {ver.build})"
            elif ver.major == 10:
                return f"Windows 10 (Build {ver.build})"
            elif ver.major == 6 and ver.minor == 3:
                return "Windows 8.1"
            elif ver.major == 6 and ver.minor == 2:
                return "Windows 8"
            elif ver.major == 6 and ver.minor == 1:
                return "Windows 7"
            else:
                return f"Windows {ver.major}.{ver.minor} (Build {ver.build})"
        else:
            return platform.platform()
    except:
        return platform.platform()

print("")
print("=" * 40)
print("        环境与设备检测报告")
print("=" * 40)
print(f"Python 版本 : {sys.version.split()[0]}")
print(f"执行路径    : {sys.executable}")
print(f"操作系统    : {get_windows_version()}")
print("-" * 40)

if is_torch:
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU 型号    : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"显存大小    : {round(vram, 2)} GB")
            if hasattr(torch.version, 'cuda'):
                 print(f"CUDA 版本   : {torch.version.cuda}")
        except Exception as e:
            print(f"GPU 信息获取失败: {e}")
    else:
        print("GPU         : 未检测到或不可用 (CPU mode)")
else:
    print("PyTorch     : 未安装 (无法检测显卡)")

print("=" * 40)
print("✅ 检测完毕")

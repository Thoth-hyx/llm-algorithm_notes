import torch
print(f"当前 PyTorch 使用的 CUDA 版本: {torch.version.cuda}")
print(f"显卡名称: {torch.cuda.get_device_name(0)}")
print(f"显卡计算能力: {torch.cuda.get_device_capability(0)}")
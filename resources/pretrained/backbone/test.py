import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name()}")
device = torch.device("cuda")
print(device)

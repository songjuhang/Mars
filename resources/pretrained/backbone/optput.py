import torch

# 1. 加载模型权重或对象
pre = torch.load("resources/pretrained/backbone/yolov8n.pt", map_location="cpu", weights_only=False)
my_state_dict = torch.load("resources/pretrained/backbone/best_weights.pth", map_location="cpu", weights_only=False)

# 2. 提取 state_dict
pre_state_dict = pre['model'].state_dict()
my_keys = list(my_state_dict.keys())

print("="*50 + "\nPretrained Model Keys Sample (前1000个):")
for key in list(pre_state_dict.keys())[:1000]:
    print(key)

print("="*50 + "\nMy Model Keys Sample (前1000个):")
for key in my_keys[:1000]:
    print(key)

print("="*50)

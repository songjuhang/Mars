
import torch

# 1. 加载模型权重或对象
pre = torch.load("resources/pretrained/backbone/yolov8n.pt", map_location="cpu", weights_only=False)
my_state_dict = torch.load("resources/pretrained/backbone/best_weights.pth", map_location="cpu", weights_only=False)

# 2. 提取 state_dict
pre_state_dict = pre['model'].state_dict()
print("Pretrained keys sample:", list(pre_state_dict.keys())[:10])
print("My model keys sample:", list(my_state_dict.keys())[:10])

# 3. 定义映射规则
key_mapping = {
    'model.0': 'backbone.conv1',
    'model.1': 'backbone.conv2',
    'model.2': 'backbone.c2f2',
    'model.3': 'backbone.conv3',
    'model.4': 'backbone.c2f3',
    'model.5': 'backbone.conv4',
    'model.6': 'backbone.c2f4',
    'model.7': 'backbone.conv5',
    'model.8': 'backbone.c2f5',
    'model.9': 'backbone.sppf',
    'model.12': ''
}

# 4. 执行 key 替换（严格前缀匹配）
converted_state_dict = {}
for k, v in pre_state_dict.items():
    for old_prefix, new_prefix in key_mapping.items():
        if k.startswith(old_prefix + '.'):  # 严格前缀匹配，防止 model.1 匹配到 model.10
            new_key = k.replace(old_prefix, new_prefix, 1)
            converted_state_dict[new_key] = v
            break

# 5. 保存转换后的权重文件
torch.save(converted_state_dict, "resources/pretrained/backbone/backbone_nano.pth")
print("Filtered keys:", list(converted_state_dict.keys()))

# 6. 打印未使用的 key（可选）
skipped_keys = [k for k in pre_state_dict if not any(k.startswith(p + '.') for p in key_mapping)]
print("Skipped keys (not in model.0~9):", skipped_keys[:10])
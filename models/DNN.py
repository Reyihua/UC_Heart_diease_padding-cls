import torch
import torch.nn as nn


class DeepMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(DeepMLPClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 使用示例
input_size = 10  # 输入特征维度
hidden_sizes = [64, 32]  # 每个隐藏层的大小
num_classes = 2  # 分类类别数

# 初始化模型
model = DeepMLPClassifier(input_size, hidden_sizes, num_classes)
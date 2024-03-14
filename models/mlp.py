import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.fc1(x)
        out1 = self.relu1(x1)
        x2 = self.fc2(out1)
        out2 = self.relu2(x2)
        x3 = self.fc3(out2)
        out = self.sigmoid(x3)
        return out



#input_size = 10  # 输入特征维度
#hidden_size = 32  # 隐藏层大小
#num_classes = 2  # 分类类别数





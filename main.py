import torch
import os
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_padding import padding
from models.mlp import MLPClassifier
import torch.nn as nn
import torch.optim as optim

def setup(args):
    print(args.input_size)
    print(args.hidden_size)
    print(args.output_size)
    model = MLPClassifier(args.input_size,  args.hidden_size, args.output_size)

    return model

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 根据索引获取单个样本
        sample = self.data[index]
        # 将样本转换为PyTorch的张量
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        return sample_tensor

def valid(model, dataloader):
    correct = 0
    total = 0

    # 禁用梯度计算
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            y = data[:, -1:].cuda()
            x = data[:, :-1].cuda()
            y[y > 0] = 1  # [0 1 2 3]->[0 1]
            y = torch.squeeze(y, dim=1)


            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)# 获取预测结果中的最大值及其索引
            #y = torch.cat((1 - y, y), dim=1)
            total += y.size(0)  # 统计总样本数
            correct += (predicted == y).sum().item()  # 统计预测正确的样本数


    accuracy = correct / total
    return accuracy

def train(args, model):
    data1 = pd.read_csv(args.dataset_list, sep=',', names=[i for i in range(14)])

    model.cuda()

    data = np.array(data1)
    data = padding(data) #padding
    data1 = CustomDataset(data)


    dataloader = DataLoader(data1, batch_size=args.batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.06, weight_decay=0.05)

    best_accuary = 0
    for epoch in range(args.num_epochs):
      print("epoch:")
      print(epoch)
      model.train()
      for i, data in enumerate(dataloader):
          y = data[:, -1:].cuda()
          x = data[:, :-1].cuda()
          y[y > 0] = 1  # [0 1 2 3]->[0 1]

          output = model(x)
          #max_values, max_indices = torch.max(output, dim=1)
          #y = torch.squeeze(y, dim=1)
          y = torch.cat((1 - y, y), dim=1)# one hot
          loss = criterion(output, y)
          optimizer.zero_grad()  # 梯度清零
          loss.backward()  # 反向传播，计算梯度

        # 更新模型参数
          optimizer.step()
      accuary = valid(model, dataloader)
      if (accuary > best_accuary):
          best_accuary = accuary
      print(best_accuary)








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", required=True, help="Path of the data")
    parser.add_argument("--batch_size", default=16, help="size of data")
    parser.add_argument("--input_size", default=13, help="input_size")
    parser.add_argument("--hidden_size", default=256, help="size of data")
    parser.add_argument("--output_size", default=2, help="output_size")
    parser.add_argument("--num_epochs", default=50, help="num of epochs")

    args = parser.parse_args()
    model = setup(args)
    train(args, model)

if __name__=="__main__":
    main()


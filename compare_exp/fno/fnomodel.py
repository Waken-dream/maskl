import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class Burgers_ON(nn.Module):
    '''
    Masked Training Network: recover origin buregrs equation data.
    '''
    def __init__(self, in_features, length, width):
        super(Burgers_ON, self).__init__()
        self.in_features = in_features
        self.length = length
        self.width = width
        self.conv1 = nn.Conv1d(in_channels=width+1, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(2*self.width)
        self.conv2 = nn.Conv1d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(2*self.width)
        self.conv3 = nn.Conv1d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(2*self.width)
        self.conv4 = nn.Conv1d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=2*width, out_channels=width, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(self.width)
        self.conv6 = nn.Conv1d(in_channels=width, out_channels=self.in_features, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.width, num_layers=2)
        self.gelu = F.gelu
        
        
    def __get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])  # (batchsize, size_x, size_y, 1)
        return gridx.to(device)  # (batchsize, size_x, size_y, 2)
    

    def forward(self, x):
        x, (_, __) = self.lstm(x)
        grid = self.__get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 2, 1)  # convert to (batch_size, 1, resolution)
        x = self.gelu(self.bn1(self.conv1(x)))
        x1 = x   
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.gelu(self.bn3(self.conv3(x)))    # (batch, width, resolution)
        x = self.gelu(self.conv4(x))
        x = x + x1
        x = self.gelu(self.bn5(self.conv5(x)))
        x = self.gelu(self.conv6(x))
        x = x.permute(0, 2, 1)

        return x


class Darcy_ON(nn.Module):
    def __init__(self, in_features, length, width):
        super(Darcy_ON, self).__init__()
        self.in_features = in_features
        self.length = length
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=width+2, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2*self.width)
        self.conv2 = nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2*self.width)
        self.conv3 = nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2*self.width)
        self.conv5 = nn.Conv2d(in_channels=2*width, out_channels=width, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(self.width)
        self.conv6 = nn.Conv2d(in_channels=width, out_channels=self.in_features, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.width, num_layers=2)
        self.gelu = F.gelu
    
    
    def __get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (batchsize, size_x, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)  # (batchsize, size_x, size_y, 2)

    def forward(self, x):
        n_sample = x.shape[0]
        x = x.view(n_sample, -1 ,1)
        x, (_, __) = self.lstm(x)
        x = x.view(n_sample, self.length, self.length, self.width)
        grid = self.__get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(self.bn1(self.conv1(x)))
        x1 = x   
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.gelu(self.conv3(x))    # (batch, width, resolution, resolution)
        x = x + x1
        x = self.gelu(self.bn4(self.conv4(x)))
        x = self.gelu(self.bn5(self.conv5(x)))
        x = self.gelu(self.conv6(x))
        x = x.permute(0, 2, 3, 1)

        return x


class ns_ON(nn.Module):
    def __init__(self, in_features, length, width):
        super(ns_ON, self).__init__()
        self.in_features = in_features
        self.length = length
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=4*self.width+2, out_channels=5*self.width, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(5*self.width)
        self.conv2 = nn.Conv2d(in_channels=5*self.width, out_channels=5*self.width, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(5*self.width)
        self.conv3 = nn.Conv2d(in_channels=5*self.width, out_channels=5*self.width, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=5*self.width, out_channels=3*self.width, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(3*self.width)
        self.conv5 = nn.Conv2d(in_channels=3*self.width, out_channels=2*self.width, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(2*self.width)
        self.conv6 = nn.Conv2d(in_channels=2*self.width, out_channels=self.in_features, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=4*self.width, num_layers=2)
        self.gelu = F.gelu

    def __get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (batchsize, size_x, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)  # (batchsize, size_x, size_y, 2)
    
    def forward(self, x):
        n_sample, t = x.shape[0], x.shape[-1]
        x = x.view(n_sample, -1 ,t)
        x, (_, __) = self.lstm(x)
        x = x.view(n_sample, self.length, self.length, 4*self.width)
        grid = self.__get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(self.bn1(self.conv1(x)))
        x1 = x   
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.gelu(self.conv3(x))    # (batch, width, resolution, resolution)
        x = x + x1
        x = self.gelu(self.bn4(self.conv4(x)))
        x = self.gelu(self.bn5(self.conv5(x)))
        x = self.gelu(self.conv6(x))
        x = x.permute(0, 2, 3, 1)
        return x
        


if __name__ == "__main__":
    rnn = nn.LSTM(1, 20, 2)    # [input_size, hidden_size, num_layers]
    input = torch.randn(5, 1024, 1)   # [batchsize, max_length, embedding_size]
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input)
    print(f"output: {output.shape}, hn: {hn.shape}, cn: {cn.shape}")
    print(hn[:-1,...].shape)

    print("-----------------------------------------------")
    lstm2 = nn.LSTM(input_size=1, hidden_size=20, num_layers=2)
    input = torch.randn(1000, 49, 49, 1)
    input = input.view(1000, -1, 1)
    output, (hn, cn) = lstm2(input)
    print(f"output: {output.shape}, hn: {hn.shape}, cn: {cn.shape}")
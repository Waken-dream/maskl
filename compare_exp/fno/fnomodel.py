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

class Burgers_ON_l(nn.Module):
    '''
    Masked Training Network: recover origin burgers equation data.
    '''
    def __init__(self, in_features, length, width):
        super(Burgers_ON_l, self).__init__()
        self.in_features = in_features
        self.length = length
        self.width = width
        self.conv1 = nn.Conv1d(in_channels=width+1, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(2*self.width)
        self.conv2 = nn.Conv1d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(2*self.width)
        self.conv3 = nn.Conv1d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(2*self.width)
        self.conv4 = nn.Conv1d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=2*width, out_channels=width, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(self.width)
        self.conv6 = nn.Conv1d(in_channels=width, out_channels=self.in_features, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.width, num_layers=2)
        self.gelu = F.gelu
        
    def __get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
    def forward(self, x):
        x, (_, __) = self.lstm(x)
        grid = self.__get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 2, 1)
        x = self.gelu(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        x1 = x
        x = self.gelu(self.ln2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = self.gelu(self.ln3(self.conv3(x).transpose(1, 2)).transpose(1, 2))
        x = self.gelu(self.conv4(x))
        x = x + x1
        x = self.gelu(self.ln5(self.conv5(x).transpose(1, 2)).transpose(1, 2))
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

class Darcy_ON_l(nn.Module):
    def __init__(self, in_features, length, width):
        super(Darcy_ON_l, self).__init__()
        self.in_features = in_features
        self.length = length
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=width+2, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(2*self.width)
        self.conv2 = nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(2*self.width)
        self.conv3 = nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2*width, out_channels=2*width, kernel_size=3, stride=1, padding=1)
        self.ln4 = nn.LayerNorm(2*self.width)
        self.conv5 = nn.Conv2d(in_channels=2*width, out_channels=width, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(self.width)
        self.conv6 = nn.Conv2d(in_channels=width, out_channels=self.in_features, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.width, num_layers=2)
        self.gelu = F.gelu
    
    def __get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def forward(self, x):
        n_sample = x.shape[0]
        x = x.view(n_sample, -1, 1)
        x, (_, __) = self.lstm(x)
        x = x.view(n_sample, self.length, self.length, self.width)
        grid = self.__get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(self.ln1(self.conv1(x).transpose(1, 3)).transpose(1, 3))
        x1 = x
        x = self.gelu(self.ln2(self.conv2(x).transpose(1, 3)).transpose(1, 3))
        x = self.gelu(self.conv3(x))
        x = x + x1
        x = self.gelu(self.ln4(self.conv4(x).transpose(1, 3)).transpose(1, 3))
        x = self.gelu(self.ln5(self.conv5(x).transpose(1, 3)).transpose(1, 3))
        x = self.gelu(self.conv6(x))
        x = x.permute(0, 2, 3, 1)
        return x

class ns_ON_l(nn.Module):
    def __init__(self, in_features, length, width):
        super(ns_ON_l, self).__init__()
        self.in_features = in_features
        self.length = length
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=4*self.width+2, out_channels=5*self.width, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(5*self.width)
        self.conv2 = nn.Conv2d(in_channels=5*self.width, out_channels=5*self.width, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(5*self.width)
        self.conv3 = nn.Conv2d(in_channels=5*self.width, out_channels=5*self.width, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=5*self.width, out_channels=3*self.width, kernel_size=3, stride=1, padding=1)
        self.ln4 = nn.LayerNorm(3*self.width)
        self.conv5 = nn.Conv2d(in_channels=3*self.width, out_channels=2*self.width, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(2*self.width)
        self.conv6 = nn.Conv2d(in_channels=2*self.width, out_channels=self.in_features, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=4*self.width, num_layers=2)
        self.gelu = F.gelu
    
    def __get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def forward(self, x):
        n_sample, t = x.shape[0], x.shape[-1]
        x = x.view(n_sample, -1, t)
        x, (_, __) = self.lstm(x)
        x = x.view(n_sample, self.length, self.length, 4*self.width)
        grid = self.__get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(self.ln1(self.conv1(x).transpose(1, 3)).transpose(1, 3))
        x1 = x
        x = self.gelu(self.ln2(self.conv2(x).transpose(1, 3)).transpose(1, 3))
        x = self.gelu(self.conv3(x))
        x = x + x1
        x = self.gelu(self.ln4(self.conv4(x).transpose(1, 3)).transpose(1, 3))
        x = self.gelu(self.ln5(self.conv5(x).transpose(1, 3)).transpose(1, 3))
        x = self.gelu(self.conv6(x))
        x = x.permute(0, 2, 3, 1)
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP_1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP_1d, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP_1d(self.width, self.width, self.width)
        self.mlp1 = MLP_1d(self.width, self.width, self.width)
        self.mlp2 = MLP_1d(self.width, self.width, self.width)
        self.mlp3 = MLP_1d(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP_1d(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        # width = 20, modes1=12, modes2=12
        # in_channels = out_channels = width =20
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication 定义向量乘的规则，即定义input和weights如何做乘
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # [20,20,12,12] * [20,20,12,12] = [20,20,12,12]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d_time(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_time, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic

        self.p = nn.Linear(12, self.width)  # 将输入的12个channel映射到想要的channel，这里设置为width个channel
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # 4 Fourier Layers:
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (batchsize, size_x, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)  # (batchsize, size_x, size_y, 2)


class MLC_1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC_1d, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels, True)
        self.mlp2 = nn.Linear(mid_channels, out_channels, True)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class ON_1d(nn.Module):
    '''
    Masked Training Network: recover origin data.
    '''
    def __init__(self, in_features, width):
        super(ON_1d, self).__init__()
        self.in_feature = in_features
        self.width = width
        self.p = nn.Linear(self.in_feature + 1, self.width)
        self.q = MLC_1d(in_channels=self.width, out_channels=self.in_feature, mid_channels=self.width * 4)
        self.mlp0 = MLC_1d(self.width, self.width, self.width)
        self.mlp1 = MLC_1d(self.width, self.width, self.width)
        self.mlp2 = MLC_1d(self.width, self.width, self.width)
        self.mlp3 = MLC_1d(self.width, self.width, self.width)
        self.gelu = F.gelu

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])  # (batchsize, size_x, size_y, 1)
        return gridx.to(device)  # (batchsize, size_x, size_y, 2)

    def forward(self, x):
        grid = self.get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)  # Add two dimension of location information (1,256,256,160+2)
        x = self.p(x)  # into latent space
        x1 = x

        # forward propagation
        x = self.gelu(self.mlp0(x))
        x = self.gelu(self.mlp1(x))
        x = self.gelu(self.mlp2(x))
        x = self.gelu(self.mlp3(x))
        x = x + x1

        x = self.q(x)  # back to original space

        return x

class MLC_2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC_2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class ON_2d(nn.Module):
    '''
    Masked Training Network: recover origin data.
    '''
    def __init__(self, in_features, width):
        super(ON_2d, self).__init__()
        self.in_feature = in_features
        self.width = width
        self.p = nn.Linear(self.in_feature+2, self.width)
        self.q = MLC_2d(in_channels=self.width, out_channels=self.in_feature, mid_channels=self.width * 4)
        self.mlp0 = MLC_2d(self.width, self.width, self.width)
        self.mlp1 = MLC_2d(self.width, self.width, self.width)
        self.mlp2 = MLC_2d(self.width, self.width, self.width)
        self.mlp3 = MLC_2d(self.width, self.width, self.width)
        self.gelu = F.gelu

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (batchsize, size_x, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)  # (batchsize, size_x, size_y, 2)

    def forward(self, x):
        grid = self.get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)  # Add two dimension of location information (1,256,256,160+2)
        x = self.p(x)  # into latent space
        x = x.permute(0, 3, 1, 2).contiguous() 
        x1 = x

        # forward propagation
        x = self.gelu(self.mlp0(x))
        x = self.gelu(self.mlp1(x))
        x = self.gelu(self.mlp2(x))
        x = self.gelu(self.mlp3(x))
        x = x + x1

        x = self.q(x)
        x = x.permute(0, 2, 3, 1).contiguous()   # back to original space
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
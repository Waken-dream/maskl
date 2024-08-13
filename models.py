import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_grid(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (batchsize, size_x, size_y, 1)
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)  # (batchsize, size_x, size_y, 2)

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


class MLC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

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

        self.p = nn.Linear(self.modes1+2, self.width)  # 将输入的12个channel映射到想要的channel，这里设置为width个channel
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # 4 Fourier Layers:
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLC(self.width, self.width, self.width)
        self.mlp1 = MLC(self.width, self.width, self.width)
        self.mlp2 = MLC(self.width, self.width, self.width)
        self.mlp3 = MLC(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLC(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

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
        return x  # (batch_size, size_x, size_y, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (batchsize, size_x, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)  # (batchsize, size_x, size_y, 2)


class MTN(nn.Module):
    """
    Masked Training Network: Prediction
    """
    def __init__(self, in_features, width):
        super(MTN, self).__init__()
        self.in_features = in_features
        self.width = width
        self.p = nn.Linear(self.in_features+2, self.width)
        self.q = MLC(in_channels=self.width, out_channels=1, mid_channels=self.width * 4)
        self.mlp0 = MLC(self.width, self.width, self.width)
        self.mlp1 = MLC(self.width, self.width, self.width)
        self.mlp2 = MLC(self.width, self.width, self.width)
        self.mlp3 = MLC(self.width, self.width, self.width)
        self.gelu = F.gelu

    def forward(self, x):
        grid = get_grid(shape=x.shape, device=x.device)
        x = torch.cat((x, grid), dim=-1)  # Add two dimension of location information (1,256,256,160+2)
        x = self.p(x)  # into latent space
        x = x.permute(0, 3, 1, 2).contiguous()
        x1 = x

        # forward propagation
        x = self.gelu(self.mlp0(x))
        x = self.gelu(self.mlp1(x))
        x = x + x1
        x1 = x
        x = self.gelu(self.mlp2(x))
        x = self.gelu(self.mlp3(x))
        x = x + x1

        x = self.q(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # back to original space
        return x


class ON(nn.Module):
    '''
    Masked Training Network: recover origin data.
    '''
    def __init__(self, in_features, width):
        super(ON, self).__init__()
        self.in_feature = in_features
        self.width = width
        self.p = nn.Linear(self.in_feature+2, self.width)
        self.q = MLC(in_channels=self.width, out_channels=self.in_feature, mid_channels=self.width * 4)
        self.mlp0 = MLC(self.width, self.width, self.width)
        self.mlp1 = MLC(self.width, self.width, self.width)
        self.mlp2 = MLC(self.width, self.width, self.width)
        self.mlp3 = MLC(self.width, self.width, self.width)
        self.gelu = F.gelu 

    def forward(self, x):
        grid = get_grid(shape=x.shape, device=x.device)
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


class DON(nn.Module):
    """
    Masked Train Network: down stream works (predict future
    """
    def __init__(self, origin_model, in_features, width):
        super(DON,self).__init__()
        self.origin_model = origin_model
        self.in_features = in_features
        self.width = width
        self.gelu = F.gelu
        self.new_layer = nn.Sequential(
            nn.Linear(self.in_features, 4 * self.in_features),
            nn.GELU(),
            nn.Linear(4 * self.in_features, 4 * self.in_features),
            nn.GELU(),
            nn.Linear(4 * self.in_features, 1)
        )

    def forward(self,x):
        x = self.origin_model(x)
        x = self.new_layer(x)

        return x

class DON_lite(nn.Module):
    """
    Masked Train Network: down stream works (predict future
    """
    def __init__(self, origin_model, in_features, width):
        super(DON_lite,self).__init__()
        self.origin_model = origin_model
        self.in_features = in_features
        self.width = width
        self.gelu = F.gelu
        self.new_layer = nn.Sequential(
            nn.Linear(self.in_features, 4*self.in_features),
            nn.GELU(),
            nn.Linear(4*self.in_features, 1)
        )

    def forward(self,x):
        x = self.origin_model(x)
        x = self.new_layer(x)

        return x

def time_embed(x:torch.Tensor,t:float)->torch.Tensor:
    """
    TIme embedding as Transformer
    x.shape: (batch_size, resolution, time)
    """
    channels, resolution, _, in_features= x.shape
    b = torch.arange(t, t + in_features, device=x.device, dtype=torch.float32)
    b = b.repeat(channels, resolution, resolution, 1)
    c = torch.sin(b/in_features)
    d = torch.cos(b/in_features)
    b[:, :, :, ::2] = c[:, :, :, ::2]
    b[:, :, :, 1::2] = d[:, :, :, 1::2]
    return b


class MLC_2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(mid_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = F.relu(x)
        x = self.norm2(self.conv2(x))
        return x

class TimeDon2d(torch.nn.Module):
    """
    Time DON to predict 2D tensor Type Data
    Input Tensor: num * resolution * time
    """

    def __init__(self, in_features, out_features, width):
        super(TimeDon2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.gelu = torch.nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.p = MLC_2(self.in_features, self.width*2, self.width)
        self.conv1 = MLC_2(self.width*2, self.width*4, self.width*4)
        self.conv2 = MLC_2(self.width*4, self.width*2, self.width*4)
        self.conv3 = MLC_2(self.width*2, self.width*2, self.width*2)
        self.conv4 = MLC_2(self.width*2, self.width*2, self.width*2)
        self.q = MLC_2(self.width*2, self.out_features, self.width*2)


    def forward(self, x, t):
        b = time_embed(x, t)
        x = x + b
        # x = x.permute(0, 2, 1)
        x = torch.permute(x, [0, 3, 1, 2]).contiguous()
        x = self.p(x)
        x1 = x
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        # x = self.maxpool(x)
        x = x + x1
        x1 = x
        x = self.gelu(self.conv3(x))
        x = self.gelu(self.conv4(x))
        x = x + x1
        x = self.gelu(self.q(x))
        # x = self.avgpool(x)
        # x = x.permute(0, 2, 1)
        x = torch.permute(x, [0, 2, 3, 1]).contiguous()
        return x
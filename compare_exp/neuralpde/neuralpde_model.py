import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralPDE1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(NeuralPDE1d, self).__init__()  
        self.conv1 = nn.Conv1d(in_channels=in_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv4 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv5 = nn.Conv1d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv6 = nn.Conv1d(in_channels=out_channel,
                               out_channels=in_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.act = F.gelu

    def forward(self, t, u):
        # logging.info('u:{}'.format(u.is_cuda))
        u1 = u
        u = self.act(self.conv1(u))
        u = self.act(self.conv2(u))
        u = u + u1
        u1 = u
        u = self.act(self.conv3(u))
        u = self.act(self.conv4(u))
        u = self.act(self.conv5(u))
        u = u + u1
        u = self.act(self.conv6(u))
        return u
    

class NeuralPDE2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(NeuralPDE2d, self).__init__()  # (80,80)
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv4 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv5 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv6 = nn.Conv2d(in_channels=out_channel,
                               out_channels=in_channel,
                               stride=stride,
                               kernel_size=kernel_size,
                               padding=padding)
        self.act = nn.SELU(inplace=True)

    def forward(self, t, u):
        # logging.info('u:{}'.format(u.is_cuda))
        u = self.act(self.conv1(u))
        u1 = u
        u = self.act(self.conv2(u))
        u = self.act(self.conv3(u))
        u = u + u1
        u1 = u
        u = self.act(self.conv4(u))
        u = self.act(self.conv5(u))
        u = u + u1
        u = self.act(self.conv6(u))
        return u
    


if __name__ == "__main__":
    from torchdiffeq import odeint_adjoint as odeint
    import torch.distributed as dist

    y0 = torch.randn(100, 80, 80)
    #y0 = torch.randn(10, 1, 1024)
    t = torch.linspace(0,1,10)
    model = NeuralPDE2d(in_channel=100, out_channel=16)
    options = {
        "dtype": torch.float64,
        # "first_step":1.0e-9,
        # "grid_points":t,
    }
    adjoint_options = {
        "norm": "seminorm"
    }

    u_pre = odeint(
        model, y0, t, method='dopri5',
        rtol=0.0, atol=1e-5,
        options=options,
        adjoint_options=adjoint_options
    )
    
    print(f"u_pre: {u_pre.shape}")
    u_pre = torch.squeeze(u_pre)
    print(f"u_pre: {u_pre.shape}")

import random
import torch
import os
import sys
import logging
import numpy as np
import scipy
import operator
import matplotlib.pyplot as plt

sys.path.append('/public/home/mzh/maskl/navier_stokes/')
sys.path.append('/Users/maozihao/Downloads/navier-stocks/')
from models import *
from functools import reduce, partial
from utils import LpLoss, add_noise, mask_data, set_seed


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
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
        self.padding = 8  # pad the domain if input is non-periodic

        self.p = nn.Linear(161, self.width)  # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 2)  # output channel_dim is 1: u1(x)

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


def train(model, epochs: int, out_slices: int, train_loader, test_loader, T=200, num_intervals=5):
    model.train()
    better_loss = 10000000

    for epoch in range(epochs):
        train_l2_step = 0
        train_l2_full = 0

        for a, u in train_loader:
            a = a.to(device)
            u = u.to(device)
            T = u.shape[-1]
            loss = 0

            for t in range(T):
                u1 = u[..., t:t + 1]
                im = model(a)
                loss = loss + loss_fn(im, u1)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                a = torch.cat((a[..., 1:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = loss_fn(pred.reshape(batch_size, -1), u.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            test_l2_step = 0
            test_l2_full = 0
            with torch.no_grad():
                for xx, yy in test_loader:
                    test_loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)

                    for t in range(0, T):
                        y = yy[..., t:t + 1]
                        im = model(xx)
                        test_loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)

                        xx = torch.cat((xx[..., 1:], im), dim=-1)

                    test_l2_step += test_loss.item()
                    test_l2_full += loss_fn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                    if test_l2_full < better_loss:
                        # dist.barrier()
                        better_loss = test_l2_full

                        torch.save(model.state_dict(),
                                       f"./results/checkpoint/fno_burgers_0612.pth")

            if epoch % 10 == 0:
                print(epoch, train_l2_step / 16 / T, train_l2_full / 16, test_l2_step / 4 / T, test_l2_full / 4)
                logging.info(
                    f"epoch: {epoch}, mean_l2_step={train_l2_step / 16 / T}, mean_train_l2_full={train_l2_full / 16}, "
                    f"mean_test_l2_step={test_l2_step / 4 / T}, mean_test_l2_full={test_l2_full / 4}")




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"./results/fno_burgers_0612.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))

    sub = 2 ** 3  #subsampling rate
    h = 2 ** 13 // sub  #total grid size divided by the subsampling rate
    s = h

    batch_size = 8
    learning_rate = 0.001

    epochs = 10000

    modes = 16
    width = 64

    PATH = "./burgers_1d.mat"
    raw_data = scipy.io.loadmat(PATH)
    N = raw_data['data'].shape[0]
    T = raw_data['data'].shape[2]
    train_a = torch.tensor(raw_data['data'][:int(0.8 * N), :, :int(0.8 * T)]).float()
    train_u = torch.tensor(raw_data['data'][:int(0.8 * N), :, int(0.8 * T):]).float()
    test_a = torch.tensor(raw_data['data'][int(0.8 * N):, :, :int(0.8 * T)]).float()
    test_u = torch.tensor(raw_data['data'][int(0.8 * N):, :, int(0.8 * T):]).float()
    iterations = epochs * (raw_data['data'].shape[0] // batch_size)
    del raw_data

    model = FNO1d(modes, width).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    loss_fn = LpLoss(size_average=False)

    train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_a, test_u)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    num_interval = 5
    T = 200
    out_slices = 4
    train(model=model,
          epochs=epochs,
          out_slices=out_slices,
          train_loader=train_loader,
          test_loader=test_loader,
          T=T,
          num_intervals=num_interval)

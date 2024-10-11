import os
import scipy
import argparse
import torch
import torch.nn as nn

from model import *
from utilities3 import *

def args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Recover Net")

    parser.add_argument("--data", type=str, help="PDE data selection")
    parser.add_argument("--epoch", type=int, default=10000)

    args = parser.parse_args()
    return args

def mask_burgers(data: torch.Tensor) -> torch.tensor:
    return

def train(model, args, train_loader, test_loader) -> nn.Module:

    return


if __name__ == "__main__":
    args = args()

    if args.data == "burgers":
        ntrain = 1000
        ntest = 100
        sub = 2 ** 3  #subsampling rate
        h = 2 ** 13 // sub  #total grid size divided by the subsampling rate
        s = h
        batch_size = 20
        learning_rate = 0.001
        epochs = 500
        iterations = epochs * (ntrain // batch_size)
        modes = 16
        width = 64

        dataloader = MatReader('data/burgers_data_R10.mat')
        x_data = dataloader.read_field('a')[:, ::sub]  # (2048,1024) Tensor
        y_data = dataloader.read_field('u')[:, ::sub]  # (2048,1024) Tensor

        x_train = x_data[:ntrain, :]  # (1000,1024)
        y_train = y_data[:ntrain, :]  # (1000,1024)
        x_test = x_data[-ntest:, :]  # (100,1024)
        y_test = y_data[-ntest:, :]  # (100,1024)

        x_train = x_train.reshape(ntrain, s, 1) # torch.Size([1000, 1024, 1])
        x_test = x_test.reshape(ntest, s, 1)    # torch.Size([100, 1024, 1])

        mask_x_train = mask_burgers(x_train)
        mask_x_test = mask_burgers(x_test)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                                shuffle=False)
        
        model = burgers_ON()

        train(model, args, train_loader=train_loader, test_loader=test_loader)
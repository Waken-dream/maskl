"""
Train Recovery Network for burgers, darcy flow and navier stocks equcations
Returns:
    _type_: _description_
"""
import os
import time
import logging
from copy import deepcopy
import torch
import torch.nn as nn
from torch import optim

from fnomodel import *
from utilities3 import *

import sys
sys.path.append('/home/maozihao/maskl')
from utils import LpLoss, set_seed

def args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Recover Net")

    parser.add_argument("--data", type=str, required=True, help="PDE data selection")
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sim", type=float, default=0.1, help="Similarity between origin and recovery.")
    parser.add_argument("--mask_rate", default=0.3, type=float)
    parser.add_argument("--width", type=int, default=8)

    args = parser.parse_args()
    return args


def train(model, args, train_loader, test_loader) -> nn.Module:
    better_loss = 10000000

    for epoch in range(args.epoch):
        train_l2_step = 0
        test_l2_step = 0
        model.train()

        for i, (mask_a, a, u) in enumerate(train_loader):
            mask_a = mask_a.float().to(args.device)
            a = a.float().to(args.device)
            im = model(mask_a)
            # print(f"mask_a: {mask_a.shape}, a: {a.shape}, im: {im.shape}")
            train_loss = loss_fn(im, a) + args.sim * loss_fn(model(a), a)
            train_l2_step += train_loss.item()
                
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            for i, (mask_a, a, u) in enumerate(test_loader):
                mask_a = mask_a.float().to(args.device)
                a = a.float().to(args.device)

                im = model(mask_a)
                test_loss = loss_fn(im, a) + args.sim * loss_fn(model(a), a)
                test_l2_step += test_loss.item()

                if test_loss.item() < better_loss:
                    better_loss = test_loss.item()
                    torch.save(model.state_dict(), f"results/recover_{args.data}.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 1000, test_l2_step / 100)
            logging.info(
                f"epoch: {epoch}, train_l2_step: {train_l2_step / 1000}, test_l2_step: {test_l2_step/100}")

    return model


if __name__ == "__main__":
    args = args()
    set_seed(42)
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"log/mean_re_{args.data}_{args.mask_rate}_{time.strftime('%m%d', time.localtime())}.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    if args.data == "burgers":
        ntrain = 1000
        ntest = 100
        sub = 2 ** 3  #subsampling rate
        h = 2 ** 13 // sub  #total grid size divided by the subsampling rate
        s = h
        batch_size = args.batch_size
        learning_rate = 0.001
        epochs = 500
        iterations = epochs * (ntrain // batch_size)
        modes = 16
        width = 64

        dataloader = MatReader('/home/maozihao/neuraloperator/data/burgers_data_R10.mat', to_torch=True)
        x_data = dataloader.read_field('a')[:, ::sub]  # (2048,1024) Tensor
        y_data = dataloader.read_field('u')[:, ::sub]  # (2048,1024) Tensor

        x_train = x_data[:ntrain, :]  # (1000,1024)
        y_train = y_data[:ntrain, :]  # (1000,1024)
        x_test = x_data[-ntest:, :]  # (100,1024)
        y_test = y_data[-ntest:, :]  # (100,1024)
        resolution = x_train.shape[-1]

        x_train = x_train.reshape(ntrain, s, 1) # torch.Size([1000, 1024, 1])
        x_test = x_test.reshape(ntest, s, 1)    # torch.Size([100, 1024, 1])

        mask_x_train = mask_burgers(x_train, mask_rate=args.mask_rate)
        mask_x_test = mask_burgers(x_test, mask_rate=args.mask_rate)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            mask_x_train, x_train, y_train), batch_size=batch_size,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            mask_x_test, x_test, y_test), batch_size=batch_size,
                                                shuffle=False)
        
        model = Burgers_ON(in_features=1, length=resolution, width=args.width).to(args.device)
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model, args, train_loader=train_loader, test_loader=test_loader)


    elif args.data == "darcy":

        TRAIN_PATH = '/home/maozihao/neuraloperator/data/piececonst_r241_N1024_smooth1.mat'
        TEST_PATH = '/home/maozihao/neuraloperator/data/piececonst_r241_N1024_smooth2.mat'

        ntrain = 1000
        ntest = 100
        batch_size = args.batch_size
        learning_rate = 0.001
        epochs = 500
        iterations = epochs*(ntrain//batch_size)
        modes = 12
        width = 32
        r = 5
        h = int(((241 - 1)/r) + 1)
        s = h

        reader = MatReader(TRAIN_PATH)
        x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
        y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
        resolution = x_train.shape[-1]

        reader.load_file(TEST_PATH)
        x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
        y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        x_train = x_train.reshape(ntrain,s,s,1) # torch.Size([1000, 49, 49, 1])
        x_test = x_test.reshape(ntest,s,s,1)    # torch.Size([100, 49, 49, 1])

        mask_x_train = mask_darcy(x_train, mask_rate=args.mask_rate)
        mask_x_test = mask_darcy(x_test, mask_rate=args.mask_rate)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_x_train, x_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_x_test, x_test, y_test), batch_size=batch_size, shuffle=False)

        model = Darcy_ON(in_features=1, length=resolution, width=args.width).to(args.device)
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model, args, train_loader=train_loader, test_loader=test_loader)

    elif args.data == "ns":

        TRAIN_PATH = '/home/maozihao/neuraloperator/data/NavierStokes_V1e-5_N1200_T20.mat'
        TEST_PATH = '/home/maozihao/neuraloperator/data/NavierStokes_V1e-5_N1200_T20.mat'

        ntrain = 1000
        ntest = 200
        modes = 12
        width = 20
        batch_size = args.batch_size
        learning_rate = 0.001
        epochs = 500
        iterations = epochs * (ntrain // batch_size)
        sub = 1
        S = 64
        T_in = 10
        T = 10  # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
        step = 1

        reader = MatReader(TRAIN_PATH)
        train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :T_in]
        train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, T_in:T + T_in]

        reader = MatReader(TEST_PATH)
        test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :T_in]
        test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, T_in:T + T_in]

        print(train_u.shape)
        print(test_u.shape)
        assert (S == train_u.shape[-2])
        assert (T == train_u.shape[-1])

        train_a = train_a.reshape(ntrain, S, S, T_in)  # torch.Size([1000, 64, 64, 10])
        test_a = test_a.reshape(ntest, S, S, T_in)  # torch.Size([200, 64, 64, 10])

        mask_x_train = mask_ns(train_a, mask_rate=args.mask_rate)
        mask_x_test = mask_ns(test_a, mask_rate=args.mask_rate)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_x_train, train_a, train_u), batch_size=batch_size,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_x_test, test_a, test_u), batch_size=batch_size,
                                                shuffle=False)
        
        model = ns_ON(in_features=T, length=S, width=args.width).to(args.device)
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model, args, train_loader=train_loader, test_loader=test_loader)

    else:
        raise NotImplementedError
    
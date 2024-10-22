"""
Train Recovery Network for burgers, darcy flow and navier stocks equcations
Returns:
    _type_: _description_
"""
import os
import csv
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
    parser.add_argument("-a", "--action",choices=['train', 'recover', 'eval'], type=str, default="recover", help="Select mode: recover, train, eval")
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sim", type=float, default=0.1, help="Similarity between origin and recovery.")
    parser.add_argument("--mask_rate", default=0.3, type=float)
    parser.add_argument("--width", type=int, default=8)

    args = parser.parse_args()
    return args


def train_ON(model, args, train_loader, test_loader) -> nn.Module:
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
                    torch.save(model.state_dict(), f"results/recover_{args.data}_{time.strftime('%m%d', time.localtime())}.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 1000, test_l2_step / 100)
            logging.info(
                f"epoch: {epoch}, train_l2_step: {train_l2_step / 1000}, test_l2_step: {test_l2_step/100}")

    return model


def train(model, args, train_loader, test_loader) -> nn.Module:
    better_loss = 10000000

    for epoch in range(args.epoch):
        train_l2_step = 0
        test_l2_step = 0
        model.train()

        for i, (mask_a, a, u) in enumerate(train_loader):
            a = a.float().to(args.device)
            u = u.float().to(args.device)
            #print(a.shape)
            im = model(a)
            print(f"mask_a: {mask_a.shape}, a: {a.shape}, im: {im.shape}")
            train_loss = loss_fn(im, u)
            train_l2_step += train_loss.item()
                
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            for i, (mask_a, a, u) in enumerate(test_loader):
                a = a.float().to(args.device)
                u = u.float().to(args.device)

                im = model(a)
                test_loss = loss_fn(im, u)
                test_l2_step += test_loss.item()

                if test_loss.item() < better_loss:
                    better_loss = test_loss.item()
                    torch.save(model.state_dict(), f"results/train_{args.data}_{time.strftime('%m%d', time.localtime())}.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 1000, test_l2_step / 100)
            logging.info(
                f"epoch: {epoch}, train_l2_step: {train_l2_step / 1000}, test_l2_step: {test_l2_step/100}")

    return model

def train_ns(model, args, train_loader, test_loader) -> nn.Module:
    better_loss = 100000
    for ep in range(args.epoch):
        model.train()
        train_l2_step = 0
        train_l2_full = 0
        for _, xx, yy in train_loader:
            loss = 0
            xx = xx.to(args.device)  # torch.Size([20, 64, 64, 10])
            yy = yy.to(args.device)  # torch.Size([20, 64, 64, 10])

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)  # torch.Size([20, 64, 64, 1])
                loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = loss_fn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for _, xx, yy in test_loader:
                loss = 0
                xx = xx.to(args.device)
                yy = yy.to(args.device)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += loss_fn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        
        if test_l2_full < better_loss:
                    better_loss = test_l2_full
                    torch.save(model.state_dict(), f"results/train_{args.data}_{time.strftime('%m%d', time.localtime())}.pth")

        logging.info(f"{ep}, {train_l2_step / ntrain / (T / step)}, {train_l2_full / ntrain}, "
             f"{test_l2_step / ntest / (T / step)}")
        

def eval_model(model, recover_model, args, train_loader, test_loader):
    loss1 = 0
    loss2 = 0
    loss3 = 0
    model.eval()
    with torch.no_grad():
        if args.data != "ns":
            for i, (mask_a, a, u) in enumerate(test_loader):
                mask_a = mask_a.to(device)
                a = a.to(device)
                u = u.to(device)

                rec = recover_model(mask_a)
                out1 = model(mask_a)
                out2 = model(rec)
                out3 = model(a)

                loss1 += loss_fn(out1, u).item()
                loss2 += loss_fn(out2, u).item()
                loss3 += loss_fn(out3, u).item()
        else:
            for ma, xx, yy in test_loader:
                loss_1 = 0
                loss_2 = 0
                loss_3 = 0
                ma = ma.to(args.device)
                xx = xx.to(args.device)
                yy = yy.to(args.device)
                rec = recover_model(ma)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    out1 = model(ma)
                    out2 = model(rec)
                    im = model(xx)
                    loss_1 += loss_fn(out1.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    loss_2 += loss_fn(out2.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    loss_3 += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                loss1 += loss_1.item()
                loss2 += loss_2.item()
                loss3 += loss_3.item()
        
        if not os.path.exists("/home/maozihao/maskl/compare_exp/fno/compare.csv"):
            title_info = [
                ["data", "mask_rate", "corrupt_data_loss", "recovered_data_loss", "origin_data_loss", "rec_path", "model_path"],
                [args.data, args.mask_rate, loss1/ntest, loss2/ntest, loss3/ntest, rec_path, model_path]
                          ]
            with open('compare.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(title_info)
        else:
            eval_info = [
                [args.data, args.mask_rate, loss1/ntest, loss2/ntest, loss3/ntest, rec_path, model_path]
            ]
            with open('compare.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(eval_info)

    return 0


if __name__ == "__main__":
    args = args()
    set_seed(42)
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"log/{args.action}_{args.data}_{args.mask_rate}_{time.strftime('%m%d', time.localtime())}.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)
    device = args.device

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

        if args.action == "recover":
            model = Burgers_ON(in_features=1, length=resolution, width=args.width).to(args.device)
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
            train_ON(model, args, train_loader=train_loader, test_loader=test_loader)
        elif args.action == "train":
            model = FNO1d(modes, width).cuda()
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
            train(model, args, train_loader=train_loader, test_loader=test_loader)
        elif args.action == "eval":
            rec_model = Burgers_ON(in_features=1, length=resolution, width=args.width).to(device)
            rec_path = "/home/maozihao/maskl/compare_exp/fno/results/recover_burgers.pth"
            rec_state_dict = torch.load(rec_path, weights_only=True)
            rec_model.load_state_dict(rec_state_dict)
            loss_fn = LpLoss(size_average=False)
            model = FNO1d(modes, width).to(device)
            model_path = "/home/maozihao/maskl/compare_exp/fno/results/train_burgers_1021.pth"
            model_state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(model_state_dict)
            eval_model(model=model, args=args, recover_model=rec_model, train_loader=train_loader, test_loader=test_loader)


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

        if args.action == "recover":
            model = Darcy_ON(in_features=1, length=resolution, width=args.width).to(args.device)
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
            train_ON(model, args, train_loader=train_loader, test_loader=test_loader)
        elif args.action == "train":
            model = FNO2d(modes, modes, width).cuda()
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
            train(model, args, train_loader=train_loader, test_loader=test_loader)
        elif args.action == "eval":
            rec_model = Darcy_ON(in_features=1, length=resolution, width=args.width).to(args.device)
            rec_path = "/home/maozihao/maskl/compare_exp/fno/results/recover_darcy.pth"
            rec_state_dict = torch.load(rec_path, weights_only=True)
            rec_model.load_state_dict(rec_state_dict)
            loss_fn = LpLoss(size_average=False)
            model = FNO2d(modes, modes, width).cuda()
            model_path = '/home/maozihao/maskl/compare_exp/fno/results/train_darcy_1021.pth'
            model_state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(model_state_dict)
            eval_model(model=model, args=args, recover_model=rec_model, train_loader=train_loader, test_loader=test_loader)

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

        #print(train_u.shape)
        #print(test_u.shape)
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
        

        if args.action == "recover":
            model = ns_ON(in_features=T, length=S, width=args.width).to(args.device)
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
            train_ON(model, args, train_loader=train_loader, test_loader=test_loader)
        elif args.action == "train":
            model = FNO2d_time(modes, modes, width).cuda()
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
            train_ns(model, args, train_loader=train_loader, test_loader=test_loader)
        elif args.action == "eval":
            rec_model = ns_ON(in_features=T, length=S, width=args.width).to(args.device)
            rec_path = "/home/maozihao/maskl/compare_exp/fno/results/recover_ns.pth"
            rec_state_dict = torch.load(rec_path, weights_only=True)
            rec_model.load_state_dict(rec_state_dict)
            loss_fn = LpLoss(size_average=False)
            model = FNO2d_time(modes, modes, width).cuda()
            model_path = '/home/maozihao/maskl/compare_exp/fno/results/train_ns_1021.pth'
            model_state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(model_state_dict)
            eval_model(model=model, args=args, recover_model=rec_model, train_loader=train_loader, test_loader=test_loader)

    else:
        raise NotImplementedError
    
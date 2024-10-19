"""
Use Neural PDE to predict burgers, darcy flow and navier stocks equcations
"""
import os
import time
import logging
import numpy as np
import scipy
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import sys

sys.path.append('/home/maozihao/maskl')
from torchdiffeq import odeint_adjoint as odeint
from neuralpde_model import *
from compare_exp.fno.utilities3 import *

from utils import LpLoss, set_seed

os.environ["CUDA_DEVICES_MAX_CONNECTIONS"]='1'
os.environ["OMP_NUM_THREADS"] = "1"

def args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Recover Net")

    parser.add_argument("--data", type=str, required=True, help="PDE data selection")
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--sim", type=float, default=0.1, help="Similarity between origin and recovery.")
    parser.add_argument("--mask_rate", default=0.3, type=float)
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--method", default="adaptive_heun", help="method in odeint")
    parser.add_argument("--rtol", default=0.0, type=float, help="relative tolerance")
    parser.add_argument("--atol", default=1e-5, type=float, help="absolute tolerance")
    parser.add_argument("--master_port", default=20501, type=int)

    args = parser.parse_args()
    return args


def train(model, t, args, train_loader, test_loader) -> nn.Module:
    better_loss = 10000000

    for epoch in range(args.epoch):
        train_l2_step = 0
        test_l2_step = 0
        model.train()

        for i, (mask_a, a, u) in enumerate(train_loader):
            # mask_a = mask_a.float().to(args.device)
            a = a.float().to(device)
            u = u.float().to(device)
            options = {
                "dtype": torch.float64,
                # "first_step":1.0e-9,
                # "grid_points":t,
            }
            adjoint_options = {
                "norm": "seminorm"
            }

            u_pre = odeint(
                model, a, t, method=args.method,
                rtol=args.rtol, atol=args.atol,
                options=options,
                adjoint_options=adjoint_options
            )

            if args.data == 'ns':
                u_pre = torch.squeeze(u_pre).permute(1, 0, 2, 3)
                train_loss = loss_fn(u_pre.float(), u.float())
            elif args.data == "burgers":
                train_loss = loss_fn(u_pre[-1, ...].float(), u.float())
            elif args.data == "darcy":
                u_pre = torch.squeeze(u_pre).permute(1, 0, 2, 3)
                train_loss = loss_fn(u_pre[:, -1, : ,:].float(), u.float())
            else:
                raise NotImplementedError

            train_l2_step += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()


        with torch.no_grad():
            model.eval()
            for i, (mask_a, a, u) in enumerate(test_loader):
                # mask_a = mask_a.float().to(args.device)
                a = a.float().to(device)
                u = u.float().to(device)

                options = {
                "dtype": torch.float64,
                # "first_step":1.0e-9,
                # "grid_points":t,
            }
            adjoint_options = {
                "norm": "seminorm"
            }

            u_pre = odeint(
                model, a, t, method=args.method,
                rtol=args.rtol, atol=args.atol,
                options=options,
                adjoint_options=adjoint_options
            )

            if args.data == 'ns':
                u_pre = torch.squeeze(u_pre).permute(1, 0, 2, 3)
                test_loss = loss_fn(u_pre.float(), u.float())
            elif args.data == "burgers":
                test_loss = loss_fn(u_pre[-1, ...].float(), u.float())
            elif args.data == "darcy":
                u_pre = torch.squeeze(u_pre).permute(1, 0, 2, 3)
                test_loss = loss_fn(u_pre[:, -1, : ,:].float(), u.float())
            else:
                raise NotImplementedError
        
            test_l2_step += test_loss.item()

            if test_loss.item() < better_loss:
                    better_loss = test_loss.item()
                    torch.save(model.module.state_dict(), f"results/train_{args.data}_{time.strftime('%m%d', time.localtime())}.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 1000, test_l2_step / 100)
            logging.info(
                f"epoch: {epoch}, train_l2_step: {train_l2_step / 1000}, test_l2_step: {test_l2_step/100}")
            
    return model



if __name__ == "__main__":
    args = args()
    set_seed(13)
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"log/neural_{args.data}_{args.mask_rate}_{args.method}_{time.strftime('%m%d', time.localtime())}.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    os.environ["CUDA_DEVICES_MAX_CONNECTIONS"] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MASTER_PORT"] = str(args.master_port)

    print(f"Support distributed environment: {dist.is_available()}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    if args.data == "burgers":

        PATH = '/home/maozihao/neuraloperator/data/burgers_data_R10.mat'

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

        dataloader = MatReader(PATH, to_torch=True)
        x_data = dataloader.read_field('a')[:, ::sub]  # (2048,1024) Tensor
        y_data = dataloader.read_field('u')[:, ::sub]  # (2048,1024) Tensor

        x_train = x_data[:ntrain, :]  # (1000,1024)
        y_train = y_data[:ntrain, :]  # (1000,1024)
        x_test = x_data[-ntest:, :]  # (100,1024)
        y_test = y_data[-ntest:, :]  # (100,1024)
        resolution = x_train.shape[-1]

        x_train = x_train.reshape(ntrain, s, 1).permute(0,2,1) # torch.Size([1000, 1, 1024])
        x_test = x_test.reshape(ntest, s, 1).permute(0,2,1)    # torch.Size([100, 1, 1024])
        time_t = torch.linspace(0, 1, 10).to(device)

        mask_x_train = mask_burgers(x_train, mask_rate=args.mask_rate)
        mask_x_test = mask_burgers(x_test, mask_rate=args.mask_rate)

        train_dataset = torch.utils.data.TensorDataset(mask_x_train, x_train, y_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=train_sampler)
        
        test_dataset = torch.utils.data.TensorDataset(mask_x_test, x_test, y_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=test_sampler)
        
        model = NeuralPDE1d(in_channel=1, out_channel=16).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)  # multiply nodes
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model=model, 
              t=time_t, 
              args=args, 
              train_loader=train_loader, 
              test_loader=test_loader)

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
        h = int(((241 - 1)/r) + 1)  # 49
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
        y_train = y_normalizer.encode(y_train)  # torch.Size([1000, 49, 49])

        x_train = x_train.reshape(ntrain,s,s,1).permute(0, 3, 1, 2) # torch.Size([1000, 1, 49, 49])
        x_test = x_test.reshape(ntest,s,s,1).permute(0, 3, 1, 2)    # torch.Size([100, 1, 49, 49]
        time_t = torch.linspace(0, 1, 10).to(device)

        mask_x_train = mask_darcy(x_train, mask_rate=args.mask_rate)
        mask_x_test = mask_darcy(x_test, mask_rate=args.mask_rate)

        train_dataset = torch.utils.data.TensorDataset(mask_x_train, x_train, y_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=train_sampler)
        
        test_dataset = torch.utils.data.TensorDataset(mask_x_test, x_test, y_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=test_sampler)
        
        model = NeuralPDE2d(in_channel=1, out_channel=16).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)  # multiply nodes
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model=model, 
              t=time_t, 
              args=args, 
              train_loader=train_loader, 
              test_loader=test_loader)

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
        train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :1]
        train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, :T + T_in].permute(0, 3, 1, 2)
        reader = MatReader(TEST_PATH)
        test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :1]
        test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, :T + T_in].permute(0, 3, 1, 2)

        #print(train_u.shape)    # torch.Size([1000, 20, 64, 64])
        #print(test_u.shape)     # torch.Size([200, 20, 64, 64])

        train_a = train_a.reshape(ntrain, S, S, 1).permute(0, 3, 1, 2)  # torch.Size([1000, 1, 64, 64])
        test_a = test_a.reshape(ntest, S, S, 1).permute(0, 3, 1, 2)  # torch.Size([200, 1, 64, 64])
        time_t = torch.linspace(0, 20, 20).to(device)

        mask_x_train = mask_ns(train_u, mask_rate=args.mask_rate)
        mask_x_test = mask_ns(test_u, mask_rate=args.mask_rate)
        
        train_dataset = torch.utils.data.TensorDataset(mask_x_train, train_a, train_u)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=train_sampler)
        
        test_dataset = torch.utils.data.TensorDataset(mask_x_test, test_a, test_u)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=test_sampler)
        
        model = NeuralPDE2d(in_channel=1, out_channel=16).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)  # multiply nodes
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model=model, 
              t=time_t, 
              args=args, 
              train_loader=train_loader, 
              test_loader=test_loader)

    else:
        raise NotImplementedError
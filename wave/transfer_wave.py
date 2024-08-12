"""
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 transfer_wave.py --epoch 60000 --master_port 20502
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 transfer_wave.py --spectral --epoch 60000 --master_port 20503
"""

import logging
import os
import time
import sys
import scipy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from models import ON, DON
from utils import LpLoss, mask_data, mean_mask_data, set_seed

os.environ["CUDA_DEVICES_MAX_CONNECTIONS"]='1'
os.environ["OMP_NUM_THREADS"] = "1"


def args():
    import argparse
    parser = argparse.ArgumentParser(description="Mask Learning")

    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default="../data/wave_2d.mat", type=str)
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--mask_times", default=4, type=int)
    parser.add_argument("--mask_rate", default=0.4, type=float)
    parser.add_argument("--model", default='DON', type=str)
    parser.add_argument("--model_path",default="./checkpoint/re_mean_0.4mask0802.pth", type=str)
    parser.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=2, help="world size")
    parser.add_argument('--init_method', default='tcp://10.10.10.22:29500',help="init-method")
    parser.add_argument('--rank', default=0, help='rank of current process')
    parser.add_argument("--spectral", action="store_true", default=False, help="Use new train fuction")
    parser.add_argument("--master_port", default=20502, type=int)

    args = parser.parse_args()
    return args


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


class MLC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC, self).__init__()
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
        self.p = MLC(self.in_features, self.width*2, self.width)
        self.conv1 = MLC(self.width*2, self.width*4, self.width*4)
        self.conv2 = MLC(self.width*4, self.width*2, self.width*4)
        self.conv3 = MLC(self.width*2, self.width*2, self.width*2)
        self.conv4 = MLC(self.width*2, self.width*2, self.width*2)
        self.q = MLC(self.width*2, self.out_features, self.width*2)


    def forward(self, x, t):
        """
        b = torch.arange(t, t + self.in_features, device=x.device, dtype=torch.float32)
        features = x.shape[0]
        b = b.repeat(features, 1).unsqueeze(1)
        # b = b.to(device=device, dtype=torch.float32)
        x = torch.concat([b, x], dim=1)
        x1 = x
        """
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


def transfer_learning(tmodel, epochs, train_loader, test_loader):
    better_loss = 10000000
    date_time = time.strftime('%m%d', time.localtime())
    for epoch in range(epochs):
        tmodel.train()
        train_l2_step = 0
        train_l2_full = 0

        for a, u in train_loader:
            loss = 0
            a = a.to(device)
            u = u.to(device)
            T = u.shape[-1]

            for t in range(T):
                u1 = u[..., t:t + 1]
                im = tmodel(a)
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
                    im = tmodel(xx)
                    test_loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., 1:], im), dim=-1)

                test_l2_step += test_loss.item()
                test_l2_full += loss_fn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                if test_l2_full < better_loss:
                    #dist.barrier()
                    better_loss = test_l2_full
                    if dist.get_rank() == 0:
                        torch.save(tmodel.module.state_dict(),
                                   f"./results/mean_trans_wave{args.mask_rate}_{date_time}.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 16 / T, train_l2_full / 16, test_l2_step / 4 / T, test_l2_full / 4)
            logging.info(
                f"epoch: {epoch}, mean_l2_step={train_l2_step / 16 / T}, mean_train_l2_full={train_l2_full / 16}, "
                f"mean_test_l2_step={test_l2_step / 4 / T}, mean_test_l2_full={test_l2_full / 4}")


def new_transfer_train(recover_model, tmodel, epochs: int, out_slices: int, train_loader, test_loader, T=200,
                       num_intervals=5):
    better_loss = 10000000
    date_time = time.strftime('%m%d', time.localtime())
    interval = T // num_intervals  # 16
    assert interval // out_slices == interval / out_slices, "Out slices must divide Time interval !"
    for num in range(num_intervals-1):  # Divide timeline into intervals
        each_epoch = epochs // num_intervals
        rel_time = torch.arange(num * interval, num * interval + interval)  # Relative time
        # autograd.set_detect_anomaly(True)
        for epoch in range(each_epoch):
            tmodel.train()
            train_l2_step = 0
            train_l2_full = 0

            for masked_a, u in train_loader:
                start_time = num * interval
                loss = 0
                masked_a = masked_a.to(device)
                u = u.to(device)
                a = recover_model(masked_a)
                for p in range(15):
                    a = recover_model(a)
                a0 = torch.concat([a, u], dim=-1)  
                del a, u

                a = a0[..., num * interval: (num + 1) * interval].to(device)  # interval = model.in_features
                u = a0[..., (num + 1) * interval: (num + 2) * interval].to(device)
                step = interval // out_slices  # step=4, out_slices=4
                # print(f"a: {a.shape}, u: {u.shape}, a0: {a0.shape}")

                for i in range(step):
                    y = u[..., i * out_slices: (i + 1) * out_slices]
                    im = tmodel(a, start_time + i * step)
                    start_time += out_slices
                    loss = loss + loss_fn(im, y)
                    if i == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    a = torch.cat((a[..., out_slices:], im), dim=-1)

                train_l2_step += loss.item()
                train_l2_full = loss_fn(u, pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            del loss

            # Validation
            test_l2_loss = 0
            equal_test_l2_loss = 0
            for masked_a, u in test_loader:
                loss = 0
                start_time = num * interval
                masked_a = masked_a.to(device)
                u = u.to(device)
                # masked_test_a: torch.Size([1, 128, 128, 128]), u: torch.Size([1, 128, 128, 40])
                a = recover_model(masked_a)
                for p in range(15):
                    a = recover_model(a)
                a0 = torch.concat([a, u], dim=-1)


                del a, u
                a = a0[..., num * interval: num * interval + interval].to(device)
                u = a0[..., (num + 1) * interval: (num + 2) * interval].to(device)

                for i in range(step):
                    y = u[..., i * out_slices: (i + 1) * out_slices]
                    im = tmodel(a, start_time + i * step)
                    # print(f"step {i}, start_time: {start_time}, im.shape: {im.shape}, y.shape: {y.shape}")
                    start_time += out_slices
                    loss = loss + loss_fn(im, y)
                    if i == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    a = torch.cat((a[..., out_slices:], im), dim=-1)
                    if i == step - 1:
                        equal_test_l2_loss += loss.item()

                test_l2_loss += loss.item()
                if equal_test_l2_loss < better_loss:
                    better_loss = equal_test_l2_loss
                    if dist.get_rank() == 0:
                        torch.save(tmodel.module.state_dict(), 
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./results/spectral_trans_wave{args.mask_rate}_{date_time}.pth"))

            if epoch % 10 == 0:
                logging.info(
                    f"Num {num}, Epoch {epoch}, train_l2_step: {train_l2_step}, train_l2_full: {train_l2_full}\n"
                    f"test_l2_loss: {test_l2_loss}, equal_test_l2_loss: {equal_test_l2_loss}")
                print(f"Num {num}, Epoch {epoch}, train_l2_step: {train_l2_step}, train_l2_full: {train_l2_full}\n"
                      f"test_l2_loss: {test_l2_loss}, equal_test_l2_loss: {equal_test_l2_loss}")


if __name__ == '__main__':
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./runlog/mean_trans_{time.strftime('%m%d', time.localtime())}.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    print(f"Support distributed environment: {dist.is_available()}")
    set_seed(42)
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["CUDA_DEVICES_MAX_CONNECTIONS"] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    DATA_PATH = args.data_path
    raw_data = scipy.io.loadmat(DATA_PATH)
    batch_size = args.batch_size
    sub = 1
    S = 64
    learning_rate = args.lr
    epochs = args.epochs
    iterations = epochs * (raw_data['data'].shape[0] // batch_size)
    T = raw_data['data'].shape[-1]  # int: 128
    N = raw_data['data'].shape[0]

    train_a = raw_data['data'][:int(0.8 * N), :, :, :int(0.8 * T)]   # (30, 128, 128, 128)
    train_u = raw_data['data'][:int(0.8 * N), :, :, int(0.8 * T):]
    test_a = raw_data['data'][int(0.8 * N):, :, :, :int(0.8 * T)]
    test_u = raw_data['data'][int(0.8 * N):, :, :, int(0.8 * T):]

    print(f"train_u.shape = {train_u.shape}")
    print(f"test_u.shape = {test_u.shape}")

    train_a = mean_mask_data(train_a, mask_rate=args.mask_rate)
    train_a = torch.tensor(train_a).float()
    train_u = torch.tensor(train_u).float()
    train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler)

    test_a = mean_mask_data(test_a, mask_rate=args.mask_rate)
    test_a = torch.tensor(test_a).float()
    test_u = torch.tensor(test_u).float()
    test_dataset = torch.utils.data.TensorDataset(test_a, test_u)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        # shuffle=False,
        sampler=test_sampler)
    del raw_data

    model = ON(in_features=train_a.shape[-1], width=20).cuda()
    if device == torch.device('cpu'):
        model_state_dict = torch.load(args.model_path,
                                      map_location=torch.device('cpu'))
    else:
        model_state_dict = torch.load(args.model_path)
    model.load_state_dict(model_state_dict, False)

    loss_fn = LpLoss(size_average=False)
    model.eval()
   

    if not args.spectral:
        tmodel = DON(origin_model=model, in_features=model.in_feature, width=model.width).cuda()
        tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[local_rank],
                                                       output_device=local_rank)
        
        optimizer = torch.optim.Adam(tmodel.module.new_layer.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        transfer_learning(tmodel=tmodel, 
                          epochs=epochs,
                          train_loader=train_loader, 
                          test_loader=test_loader)
        
    else:
        num_interval = 8
        out_slices = 4  # tmodel output slices
        tmodel = TimeDon2d(in_features=T // num_interval, out_features=out_slices, width=20).cuda()
        tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[local_rank],
                                                           output_device=local_rank)  # multiply nodes
        optimizer = torch.optim.Adam(tmodel.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        new_transfer_train(recover_model=model,
                           tmodel=tmodel,
                           epochs=epochs,
                           out_slices=out_slices,
                           train_loader=train_loader,
                           test_loader=test_loader,
                           T=T,
                           num_intervals=num_interval)



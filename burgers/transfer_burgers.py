"""
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 ./data_generation/burgers/transfer_burgers.py
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 ./data_generation/burgers/transfer_burgers.py --mask_rate 0.3 --master_port 20508 --origin_model re_burgers_0.3mask_noise_False0409

$ torchrun --standalone --nnodes 1 --nproc_per_node 2 ./data_generation/burgers/transfer_burgers.py --spectral
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 ./data_generation/burgers/transfer_burgers.py --mask_rate 0.3 --master_port 20507 --origin_model mean_re_burgers_0.3_0409 --spectral

$ torchrun --standalone --nnodes 1 --nproc_per_node 2 transfer_burgers.py --spectral --data_path './data_generation/burgers/burgers_1d_1000.mat' --epochs 60000
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 ./transfer_burgers.py --spectral --data_path './burgers_1d_1000a.mat' --epochs 60000 --origin_model mean_re_burgers_0.4_0514
$ torchrun --standalone --nnodes 1 --nproc_per_node 2 transfer_burgers.py --spectral --data_path './burgers_1d_1000a.mat' --epochs 60000 --origin_model mean_re_burgers_0.3_0524 --master_port 25009 --mask_rate 0.3
"""
import logging
import os
import sys
import random
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import autograd

from torch.utils.data.distributed import DistributedSampler

from gan_recover import GanRecover
from recover_burgers import ON, _mask_data, _set_seed, mean_mask_data

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import LpLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def args():
    import argparse
    parser = argparse.ArgumentParser(description="Mask Learning")

    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--data_path", default="..data/burgers_1d.mat", type=str)
    parser.add_argument("--origin_model", default="mean_re_burgers_0.4_0409", type=str)
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--mask_times", default=4, type=int)
    parser.add_argument("--mask_rate", default=0.4, type=float)
    parser.add_argument("--model", default='DON', type=str)
    parser.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parser.add_argument("--spectral", action="store_true", default=False, help="Use new train fuction")
    parser.add_argument("--master_port", default=20501, type=int)

    args = parser.parse_args()
    return args

def time_embed(x:torch.Tensor,t:float)->torch.Tensor:
    """
    TIme embedding as Transformer
    x.shape: (batch_size, resolution, time)
    """
    channels, resolution, in_features= x.shape
    b = torch.arange(t, t + in_features, device=x.device, dtype=torch.float32)
    b = b.repeat(channels, resolution, 1)
    c = torch.sin(b/1000)
    d = torch.cos(b/1000)
    b[:, :, ::2] = c[:, :, ::2]
    b[:, :, 1::2] = d[:, :, 1::2]
    return b

class MLC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm1d(mid_channels)
        self.norm2 = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = F.relu(x)
        x = self.norm2(self.conv2(x))
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


class TimeDon1d(torch.nn.Module):
    """
    Time DON to predict 2D tensor Type Data
    Input Tensor: num * resolution * time
    """

    def __init__(self, in_features, out_features, width):
        super(TimeDon1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.gelu = torch.nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
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
        x = torch.permute(x, [0, 2, 1])
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
        x = torch.permute(x, [0, 2, 1])
        return x

def transfer_learning(tmodel, epochs, train_loader, test_loader):
    better_loss = 10000000

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
                                   f"./results/checkpoint/mean_trans_burgers{args.mask_rate}_0613.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 16 / T, train_l2_full / 16, test_l2_step / 4 / T, test_l2_full / 4)
            logging.info(
                f"epoch: {epoch}, mean_l2_step={train_l2_step / 16 / T}, mean_train_l2_full={train_l2_full / 16}, "
                f"mean_test_l2_step={test_l2_step / 4 / T}, mean_test_l2_full={test_l2_full / 4}")

    #torch.save(tmodel.state_dict(), "./data_generation/burgers/results/mean_trans_burgers_10000" + ".pth")


def new_transfer_train(recover_model, tmodel, epochs: int, out_slices: int, train_loader, test_loader, T=200,
                       num_intervals=5):
    recover_model.eval()
    better_loss = 10000000
    interval = T // num_intervals  # 40
    assert interval // out_slices == interval / out_slices, "Out slices must divide Time interval !"

    for num in range(num_intervals-1):  # Divide timeline into intervals
        each_epoch = epochs // num_intervals
        time = torch.arange(num * interval, num * interval + interval)  # Relative time
        #autograd.set_detect_anomaly(True)
        for epoch in range(each_epoch):
            tmodel.train()
            train_l2_step = 0
            train_l2_full = 0

            for masked_a, u in train_loader:
                start_time = num * interval
                loss = 0
                masked_a = masked_a.to(device)
                u = u.to(device)
                # masked_train_a: torch.Size([1, 256, 160]), u: torch.Size([1, 256, 40])
                a = recover_model(masked_a)
                for p in range(15):
                    a = recover_model(a)
                a0 = torch.concat([a, u], dim=-1)  # a0.shape[-1]=200
                del a, u

                a = a0[..., num * interval: (num + 1) * interval].to(device)  # interval = model.in_features
                u = a0[..., (num + 1) * interval: (num + 2) * interval].to(device)
                step = interval // out_slices  # step=10, out_slices=4
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
                # masked_test_a: torch.Size([1, 256, 160]), u: torch.Size([1, 256, 40])
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
                        torch.save(tmodel.module.state_dict(), f"./results/checkpoint/spectral_trans_burgers{args.mask_rate}_0613.pth")

            if epoch % 10 == 0:
                logging.info(
                    f"Num {num}, Epoch {epoch}, train_l2_step: {train_l2_step}, train_l2_full: {train_l2_full}\n"
                    f"test_l2_loss: {test_l2_loss}, equal_test_l2_loss: {equal_test_l2_loss}")
                print(f"Num {num}, Epoch {epoch}, train_l2_step: {train_l2_step}, train_l2_full: {train_l2_full}\n"
                      f"test_l2_loss: {test_l2_loss}, equal_test_l2_loss: {equal_test_l2_loss}")


if __name__ == "__main__":
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"./results/burgers_trans{args.mask_rate}_0613.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    print(f"Support distributed environment: {dist.is_available()}")
    _set_seed(42)

    DATA_PATH = args.data_path
    raw_data = scipy.io.loadmat(DATA_PATH)  # dict u:(20,256,256,200) a:(20,256,256) t:(1,200)
    batch_size = args.batch_size
    sub = 1
    S = 64
    learning_rate = args.lr
    epochs = args.epochs
    iterations = epochs * (raw_data['data'].shape[0] // batch_size)

    N = raw_data['data'].shape[0]
    T = raw_data['data'].shape[2]
    train_a = torch.tensor(raw_data['data'][:int(0.8 * N), :, :int(0.8 * T)]).float()
    train_u = torch.tensor(raw_data['data'][:int(0.8 * N), :, int(0.8 * T):]).float()
    test_a = torch.tensor(raw_data['data'][int(0.8 * N):, :, :int(0.8 * T)]).float()
    test_u = torch.tensor(raw_data['data'][int(0.8 * N):, :, int(0.8 * T):]).float()
    # print(f"train_a: {train_a.shape}, train_u: {train_u.shape}, test_a: {test_a.shape}, test_u: {test_u.shape}")
    # train_a: torch.Size([16, 256, 160]), train_u: torch.Size([16, 256, 40])
    # test_a: torch.Size([4, 256, 160]), test_u: torch.Size([4, 256, 40])
    del raw_data

    os.environ["CUDA_DEVICES_MAX_CONNECTIONS"] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    #model = ON(in_features=train_a.shape[-1], width=20).cuda()
    #model_state_dict = torch.load('./results/' + args.origin_model + '.pth')
    model = GanRecover(in_features=train_a.shape[-1], width=80).to(device)
    model_state_dict = torch.load("./results/checkpoint/generator_0.4_0525a.pth", map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict, False)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = LpLoss(size_average=False)

    masked_train_a = mean_mask_data(train_a, mask_rate=args.mask_rate)
    train_dataset = torch.utils.data.TensorDataset(masked_train_a, train_u)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)

    masked_test_a = mean_mask_data(test_a, mask_rate=args.mask_rate)
    test_dataset = torch.utils.data.TensorDataset(masked_test_a, test_u)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              sampler=test_sampler)

    if not args.spectral:
        tmodel = DON(origin_model=model, in_features=model.in_feature, width=model.width, out=1).cuda()
        tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[local_rank],
                                                           output_device=local_rank)  # multiply nodes
        optimizer = torch.optim.Adam(tmodel.module.new_layer.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        transfer_learning(tmodel=tmodel, epochs=epochs,
                          train_loader=train_loader, test_loader=test_loader)
    else:
        model.eval()
        num_interval = 5 if args.data_path == "..data/burgers_1d.mat" else 25
        T = 200 if args.data_path == "..data/burgers_1d.mat" else 1000 # num of time slices
        out_slices = 4  # tmodel output slices
        tmodel = TimeDon1d(in_features=T // num_interval, out_features=out_slices, width=20).cuda()
        #tmodel_state_dict = torch.load("./results/checkpoint/spectral_comp_burgers0.3_0531.pth")
        #tmodel.load_state_dict(tmodel_state_dict)
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

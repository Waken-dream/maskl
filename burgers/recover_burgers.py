"""
$ torchrun --standalone --nnodes 1 --nproc_per_node 2  ./data_generation/burgers/recover_burgers.py --mask_rate 0.3
$ torchrun --standalone --nnodes 1 --nproc_per_node 2  ./data_generation/burgers/recover_burgers.py --master_port 20502 --mask_rate 0.4

$ torchrun --standalone --nnodes 1 --nproc_per_node 2  recover_burgers.py --mask_rate 0.4 --data_path './burgers_1d_1000.mat' --mask_time 16 --epochs 100000
"""

import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math
import scipy
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import LpLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["OMP_NUM_THREADS"] = "1"


def args():
    import argparse
    parse = argparse.ArgumentParser(description="Mask Learning")

    parse.add_argument("--scheme", default="mask", type=str)
    parse.add_argument("--epochs", default=10000, type=int)
    parse.add_argument("--batch_size", default=4, type=int)
    parse.add_argument("--data_path", default="..data/burgers_1d.mat", type=str)
    parse.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parse.add_argument("--mask_times", default=4, type=int)
    parse.add_argument("--mask_rate", default=0.3, type=float)
    parse.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parse.add_argument("--transfer", action="store_true", default=False, help="Transfer Learning")
    parse.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parse.add_argument("--master_port", default=20501, type=int)

    args = parse.parse_args()
    return args


def get_grid(shape, device):
    batchsize, size_x = shape[0], shape[1]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x,)
    gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])  # (batchsize, size_x, size_y, 1)
    return gridx.to(device)  # (batchsize, size_x, size_y, 2)


def _mask_data(data: torch.Tensor, mask_rate: float) -> torch.Tensor:
    masked_data = data.to('cpu').numpy().copy()
    data_T = masked_data.shape[-1]
    mask_length = int(data_T * mask_rate)
    random_start = torch.randint(low=0, high=data_T - mask_length, size=(1,)).item()
    mask_token = np.argmax(np.abs(masked_data)) * 10
    masked_data[:, :, random_start:random_start + mask_length].fill(
        mask_token)  # use a big number to mask original data
    return torch.tensor(masked_data)


def mean_mask_data(data: torch.Tensor, mask_rate: float) -> torch.Tensor:
    masked_data = data.to('cpu').numpy().copy()
    data_T = masked_data.shape[-1]
    mask_length = int(data_T * mask_rate)
    random_start = torch.randint(low=0, high=data_T - mask_length, size=(1,)).item()
    mask_token = np.mean(masked_data)
    masked_data[:, :, random_start:random_start + mask_length].fill(mask_token)
    return torch.tensor(masked_data)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLC, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels, True)
        self.mlp2 = nn.Linear(mid_channels, out_channels, True)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class ON(nn.Module):
    '''
    Masked Training Network: recover origin data.
    '''

    def __init__(self, in_features, width):
        super(ON, self).__init__()
        self.in_feature = in_features
        self.width = width
        self.p = nn.Linear(self.in_feature + 1, self.width)
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
        x1 = x

        # forward propagation
        x = self.gelu(self.mlp0(x))
        x = self.gelu(self.mlp1(x))
        x = self.gelu(self.mlp2(x))
        x = self.gelu(self.mlp3(x))
        x = x + x1

        x = self.q(x)  # back to original space

        return x


def mask_train(epochs: int, model: nn.Module, train_loader):
    better_loss = 10000000
    for epoch in range(epochs):
        model.train()
        train_l2_step = 0
        train_l2_full = 0

        for masked_a, a, u in train_loader:
            masked_a = masked_a.to(device)
            a = a.to(device)
            # u = u.to(device)
            # T = u.shape[-1]

            im = model(masked_a)
            loss = loss_fn(im, a) + 0.2 * loss_fn(model(a), a)
            train_l2_full += loss.item()

            if train_l2_full < better_loss:
                better_loss = loss.item()
                torch.save(model.module.state_dict(),
                               f"./results/mean_re_burgers_{args.mask_rate}" + f"_0802a" + ".pth")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 160)
            logging.info(
                f"epoch: {epoch}, mean_l2_step={train_l2_step / 160}")


if __name__ == "__main__":
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"./results/mean_re_burgers_{args.mask_rate}" + f"_0802a.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)
    _set_seed(42)
    os.environ["MASTER_PORT"] = str(args.master_port)

    scheme = args.scheme
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
    train_a = torch.tensor(raw_data['data'][:, :, :int(0.8 * T)]).float()
    train_u = torch.tensor(raw_data['data'][:, :, int(0.8 * T):]).float()
    test_a = torch.tensor(raw_data['data'][int(0.8 * N):, :, :int(0.8 * T)]).float()
    test_u = torch.tensor(raw_data['data'][int(0.8 * N):, :, int(0.8 * T):]).float()
    del raw_data

    print(f"train_u.shape = {train_u.shape}")
    print(f"test_u.shape = {test_u.shape}")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    model = ON(in_features=train_a.shape[-1], width=20).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    mask_times = args.mask_times
    print(torch.distributed.is_initialized())

    for time in range(mask_times):  # 4
        masked_train_a = mean_mask_data(train_a, mask_rate=args.mask_rate)
        train_dataset = torch.utils.data.TensorDataset(masked_train_a, train_a, train_u)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=train_sampler)

        mask_train(epochs=epochs // mask_times,
                   model=model,
                   train_loader=train_loader)
    """
    import subprocess
    command = "torchrun --standalone --nnodes 1 --nproc_per_node 2 ./transfer_burgers.py --spectral --data_path './burgers_1d_1000.mat' --epochs 60000"
    subprocess.run(command, shell=True)
    """
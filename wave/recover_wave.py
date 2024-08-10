"""
torchrun --standalone --nnodes 1 --nproc_per_node 2  recover_wave.py --mask_rate 0.4 --data_path 'wave_2d.mat' --mask_time 16 --epochs 100000
"""

import math
import scipy
import numpy as np
import sys
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

sys.path.append('../')
from utils import LpLoss, mean_mask_data
from models import ON

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["OMP_NUM_THREADS"] = "1"


def args():
    import argparse
    parse = argparse.ArgumentParser(description="Mask Learning")

    parse.add_argument("--scheme", default="mask", type=str)
    parse.add_argument("--epochs", default=10000, type=int)
    parse.add_argument("--batch_size", default=4, type=int)
    parse.add_argument("--data_path", default="../data/wave_2d.mat", type=str)
    parse.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parse.add_argument("--mask_times", default=4, type=int)
    parse.add_argument("--mask_rate", default=0.3, type=float)
    parse.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parse.add_argument("--transfer", action="store_true", default=False, help="Transfer Learning")
    parse.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parse.add_argument("--master_port", default=20501, type=int)

    args = parse.parse_args()
    return args


def mask_train(epochs: int, model: nn.Module, train_loader, test_loader):
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
            loss = loss_fn(im, a)
            train_l2_step += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            test_l2_step = 0
            for masked_a, a, u in test_loader:
                masked_a = masked_a.to(device)
                a = a.to(device)
                im = model(masked_a)
                loss = 0.8 * loss_fn(im, a) + 0.2 * loss_fn(model(a), a)
                test_l2_step += loss.item()

            if test_l2_step < better_loss:
                better_loss = loss.item()
                torch.save(model.module.state_dict(), f"./checkpoint/re_mean_{args.mask_rate}" + args.scheme + f"_noise:{args.noise}0802" + ".pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 160)
            logging.info(
                f"epoch: {epoch}, mean_l2_step={train_l2_step / 128}")


if __name__ == '__main__':
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(), f"./runlog/re_mean_{args.mask_rate}" + str(args.scheme)[
                                                                                                  1:] + f"_noise:{args.noise}0802" + ".log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)
    os.environ["MASTER_PORT"] = str(args.master_port)

    scheme = args.scheme
    DATA_PATH = args.data_path
    raw_data = scipy.io.loadmat(DATA_PATH)  # dict u:(30,128,128,128) a:(30,128,128) t:(1,128)
    batch_size = args.batch_size
    sub = 1
    S = 64
    learning_rate = args.lr
    epochs = args.epochs
    local_rank = int(os.environ["LOCAL_RANK"])
    iterations = epochs * (raw_data['data'].shape[0] // batch_size)
    T = raw_data['data'].shape[-1]  # int: 128
    N = raw_data['data'].shape[0]

    train_a = raw_data['data'][:int(0.8 * N), :, :, :int(0.8 * T)]  # (16,256,256,160)
    train_u = raw_data['data'][:int(0.8 * N), :, :, :int(0.8 * T)]
    test_a = raw_data['data'][int(0.8 * N):, :, :, :int(0.8 * T)]
    test_u = raw_data['data'][int(0.8 * N):, :, :, :int(0.8 * T)]

    print(f"train_u.shape = {train_u.shape}")
    print(f"test_u.shape = {test_u.shape}")
    train_u = torch.tensor(train_u).float()

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

    masked_test_a = mean_mask_data(test_a, mask_rate=0.5)
    masked_test_a = torch.tensor(masked_test_a)
    test_a = torch.tensor(test_a).float()
    test_u = torch.tensor(test_u).float()
    test_dataset = torch.utils.data.TensorDataset(masked_test_a, test_a, test_u)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=True)

    for time in range(mask_times):
        masked_train_a = mean_mask_data(train_a, mask_rate=args.mask_rate)
        masked_train_a = torch.tensor(masked_train_a).float()
        train_a = torch.tensor(train_a).float()
        train_dataset = torch.utils.data.TensorDataset(masked_train_a, train_a, train_u)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=train_sampler)

        mask_train(epochs=epochs // mask_times,
                   model=model,
                   train_loader=train_loader,
                   test_loader=test_loader)
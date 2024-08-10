import logging
import os
import random
import matplotlib
import numpy as np
import scipy
import re
import torch
import matplotlib.pyplot as plt

from models import *
from utils import LpLoss, add_noise, set_seed, mask_data


def args():
    import argparse
    parser = argparse.ArgumentParser(description="Test corrupt Limit")

    parser.add_argument('--model_path', type=str, default="re_mask_noise_False1.pth")
    parser.add_argument("--begin_mask_rate", type=float, default=0.05, help="Begin mask rate")
    parser.add_argument("--end_mask_rate", type=float,default=0.95, help="End mask rate")
    parser.add_argument("--data_path", default="./ns_data.mat", type=str)
    parser.add_argument("--batch_size", default=3, type=int)

    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = args()
    set_seed(42)

    DATA_PATH = args.data_path
    batch_size = args.batch_size
    raw_data = scipy.io.loadmat(DATA_PATH)  # dict u:(20,256,256,200) a:(20,256,256) t:(1,200)
    T = raw_data['t'].shape[-1]  # int: 200
    N = raw_data['u'].shape[0]

    train_a = raw_data['u'][:int(0.8 * N), :, :, :int(0.8 * T)]  # (16,256,256,160)
    train_u = raw_data['u'][:int(0.8 * N), :, :, int(0.8 * T):]
    train_t = raw_data['t'][:, :int(0.8 * T)]  # (1,160)

    test_a = raw_data['u'][int(0.8 * N):, :, :, :int(0.8 * T)]
    test_u = raw_data['u'][int(0.8 * N):, :, :, int(0.8 * T):]
    test_t = raw_data['t'][:, int(0.8 * T):]

    train_a = torch.tensor(train_a)
    train_u = torch.tensor(train_u)

    test_a = torch.tensor(test_a)
    test_u = torch.tensor(test_u)

    model = ON(in_features=train_a.shape[-1], width=20)
    model_path = './checkpoint/' + args.model_path
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = LpLoss(size_average=False)
    begin_mask_rate = args.begin_mask_rate
    end_mask_rate = args.end_mask_rate
    step = (end_mask_rate-begin_mask_rate)/20

    model.to(device)
    model.eval()
    mask_rate = [begin_mask_rate + step * i for i in range(20)]
    mask_loss = []
    for i in range(20):
        masked_test_a = mask_data(test_a.to('cpu').numpy(), mask_rate[i])
        masked_test_a = torch.tensor(masked_test_a)
        test_dataset = torch.utils.data.TensorDataset(masked_test_a, test_a, test_u)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_l2_step = 0
            for masked_a, a, u in test_loader:
                masked_a = masked_a.to(device)
                a = a.to(device)
                im = model(masked_a)
                loss = loss_fn(im, a)
                test_l2_step += loss.item()
            mask_loss.append(float(test_l2_step/160))

    pattern = r"/([^/]+)\.pth$"
    match = re.search(pattern, model_path)
    save_name = match.group(1)

    plt.plot(mask_rate, mask_loss, '-ok')
    plt.xlabel("Mask rate")
    plt.ylabel("L2 Loss")
    plt.title("L2 Loss Over Different Mask Rates")
    plt.savefig(f"./results/{save_name}.png", dpi=200)
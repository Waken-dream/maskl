"""
$ python test.py --model DON --model_path ./checkpoint/re_trans1218.pth
$ python test.py --model DON --model_path ./checkpoint/nomask_DON.pth
$ python test.py --model DON --model_path ./checkpoint/re_t-noise1220.pth
"""
import random
import torch
import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from models import *
from utils import LpLoss, add_noise


def args():
    import argparse
    parse = argparse.ArgumentParser(description="Mask Learning")

    parse.add_argument("--batch_size", default=1, type=int)
    parse.add_argument("--data_path", default="./ns_data.mat", type=str)
    parse.add_argument("--model_path", default="./checkpoint/re_mask.pth", type=str)
    parse.add_argument("--model", default="DON", type=str)
    parse.add_argument("--noise", action="store_true", default=False, help="Add Random Noise")

    args = parse.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_model(model, test_loader):
    model.eval()
    test_l2_step = 0
    test_l2_full = 0
    residual = torch.randn(1, 256, 256, 40).to(device)
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(T):
                y = yy[..., t:t + 1]
                im = model(xx)
                loss += loss_fn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., 1:], im), dim=-1)

            rr = pred - yy
            residual = torch.cat((residual, rr), dim=0)
            # print(f"residual.shape = {residual.shape}")

            test_l2_step += loss.item()
            test_l2_full += loss_fn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        print(f"mean_test_l2_step={test_l2_step / 4 / T}, mean_test_l2_full={test_l2_full / 4}")
        print(f"pred.shape = {pred.shape} \nyy.shape = {yy.shape}")

        residual = torch.abs(residual[1:, :, :, :])
        print(f"residual.shape = {residual.shape}")
        max_value = []
        for i in range(residual.shape[0]):
            maximum = torch.max(residual[i, :, :, :])
            max_value.append(maximum)

        print(f"Maximum residual: {max_value}")
        print(f"Maximum Value: {max(max_value)}")


if __name__ == "__main__":
    args = args()
    print(f"Test {args.model} from {args.model_path}.")
    set_seed(42)
    device = torch.device('cuda')

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
    train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size
        # shuffle=True
    )

    test_a = torch.tensor(test_a)
    test_u = torch.tensor(test_u)
    test_dataset = torch.utils.data.TensorDataset(test_a, test_u)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size
        # shuffle=False,
    )
    T = test_u.shape[-1]
    print(f"T = {T}")
    del raw_data, test_a, test_u

    model_name = args.model
    loss_fn = LpLoss(size_average=True)

    if model_name == "DON":
        model = ON(in_features=train_a.shape[-1], width=20).cuda()
        model_state_dict = torch.load('./checkpoint/re_mask.pth')
        model.load_state_dict(model_state_dict, False)
        tmodel = DON(origin_model=model, in_features=model.in_feature, width=model.width).cuda()
        tmodel_state_dict = torch.load(args.model_path)
        tmodel.load_state_dict(tmodel_state_dict, False)
    else:
        raise NotImplemented

    test_model(model=tmodel,
               test_loader=test_loader
               )

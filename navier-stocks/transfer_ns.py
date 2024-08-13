"""
单机多卡：
$ torchrun --nproc_per_node 2 --use_env transfer.py
共享文件系统：
$ python transfer.py --init_method file://public/home/mzh/maskl/navier_stokes --world_size 3 --rank 0

$ python -m torch.distributed.launch --master_port 29500 --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr=10.10.10.22 transfer_ns.py --world_size 4 --local_rank 0
$ python transfer_ns.py --local_rank 1 --world_size 1

$ torchrun --standalone --nnodes 1 --nproc_per_node 8 transfer_ns.py --epoch 60000 --master_port 20510
$ torchrun --standalone --nnodes 1 --nproc_per_node 8 transfer_ns.py --spectral --epoch 60000 --master_port 20511
"""
import logging
import os
import time
import random
import sys
import numpy as np
import scipy
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from models import ON, DON, DON_lite, TimeDon2d
from utils import LpLoss, add_noise, mask_data, mean_mask_data

os.environ["CUDA_DEVICES_MAX_CONNECTIONS"]='1'
os.environ["OMP_NUM_THREADS"] = "1"


def args():
    import argparse
    parser = argparse.ArgumentParser(description="Mask Learning")

    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default="../data/ns_data.mat", type=str)
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--mask_times", default=4, type=int)
    parser.add_argument("--mask_rate", default=0.4, type=float)
    parser.add_argument("--model_name", default='DON', type=str)
    parser.add_argument("--origin_model", default="re_mean_0.4mask_0812", type=str)
    parser.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parser.add_argument("--spectral", action="store_true", default=False, help="Use new train fuction")
    parser.add_argument("--master_port", default=20501, type=int)

    args = parser.parse_args()
    return args


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
                        torch.save(tmodel.module.state_dict(), f"./checkpoint/tran_mean_{args.mask_rate}_{date_time}" + ".pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 16 / T, train_l2_full / 16, test_l2_step / 4 / T, test_l2_full / 4)
            logging.info(
                f"epoch: {epoch}, mean_l2_step={train_l2_step / 16 / T}, mean_train_l2_full={train_l2_full / 16}, "
                f"mean_test_l2_step={test_l2_step / 4 / T}, mean_test_l2_full={test_l2_full / 4}")

    torch.save(tmodel.state_dict(), "./checkpoint/re_trans_10000" + ".pth")


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
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./checkpoint/spectral_trans_ns{args.mask_rate}_{date_time}.pth"))

            if epoch % 10 == 0:
                logging.info(
                    f"Num {num}, Epoch {epoch}, train_l2_step: {train_l2_step}, train_l2_full: {train_l2_full}\n"
                    f"test_l2_loss: {test_l2_loss}, equal_test_l2_loss: {equal_test_l2_loss}")
                print(f"Num {num}, Epoch {epoch}, train_l2_step: {train_l2_step}, train_l2_full: {train_l2_full}\n"
                      f"test_l2_loss: {test_l2_loss}, equal_test_l2_loss: {equal_test_l2_loss}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(), f"./runlog/mean_trans_{args.mask_rate}_{time.strftime('%m%d', time.localtime())}" + ".log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    print(f"Support distributed environment: {dist.is_available()}")
    set_seed(42)

    os.environ["CUDA_DEVICES_MAX_CONNECTIONS"] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    DATA_PATH = args.data_path
    raw_data = scipy.io.loadmat(DATA_PATH)  # dict u:(20,256,256,200) a:(20,256,256) t:(1,200)
    batch_size = args.batch_size
    sub = 1
    S = 64
    learning_rate = args.lr
    epochs = args.epochs
    iterations = epochs * (raw_data['u'].shape[0] // batch_size)
    T = raw_data['t'].shape[-1]  # int: 200
    N = raw_data['u'].shape[0]

    train_a = raw_data['u'][:int(0.8 * N), :, :, :int(0.8 * T)]  # (16,256,256,160)
    train_u = raw_data['u'][:int(0.8 * N), :, :, int(0.8 * T):]
    train_t = raw_data['t'][:, :int(0.8 * T)]  # (1,160)

    test_a = raw_data['u'][int(0.8 * N):, :, :, :int(0.8 * T)]
    test_u = raw_data['u'][int(0.8 * N):, :, :, int(0.8 * T):]
    test_t = raw_data['t'][:, int(0.8 * T):]

    if args.noise:
        train_a = add_noise(train_a, 1.0)


    print(f"train_u.shape = {train_u.shape}")
    print(f"test_u.shape = {test_u.shape}")

    #train_a = mask_data(train_a, mask_rate=0.3)
    train_a = mean_mask_data(train_a, mask_rate=0.3)
    train_a = torch.tensor(train_a)
    train_u = torch.tensor(train_u)
    train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler)

    masked_test_a = mask_data(test_a, mask_rate=0.3)
    test_a = torch.tensor(test_a)
    test_u = torch.tensor(test_u)
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
        model_state_dict = torch.load(f'./checkpoint/{args.origin_model}.pth', map_location=torch.device('cpu'))
    else:
        model_state_dict = torch.load(f'./checkpoint/{args.origin_model}.pth')
    model.load_state_dict(model_state_dict, False)
    model.eval()

    if not args.spectral:
        if args.model == 'DON':
            tmodel = DON(origin_model=model, in_features=model.in_feature, width=model.width).cuda()
        elif args.model == "DON_lite":
            tmodel = DON_lite(origin_model=model, in_features=model.in_feature, width=model.width).cuda()
        else:
            raise NotImplemented

        tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[args.local_rank], output_device=args.local_rank)  # multiply nodes
        loss_fn = LpLoss(size_average=False)
        optimizer = torch.optim.Adam(tmodel.module.new_layer.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        # mask_times = args.mask_times

        transfer_learning(tmodel=tmodel, epochs=epochs,
                        train_loader=train_loader, test_loader=test_loader)
    else:
        num_interval = 5
        out_slices = 4  # tmodel output slices
        tmodel = TimeDon2d(in_features=T // num_interval, out_features=out_slices, width=20).cuda()
        tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[local_rank],
                                                           output_device=local_rank)  # multiply nodes
        loss_fn = LpLoss(size_average=False)
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

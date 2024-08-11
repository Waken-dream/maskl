"""
单机多卡：
$ python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 transfer_ns.py
新版本运行：
$ torchrun --nproc_per_node 2 --use_env transfer.py
共享文件系统：
$ python transfer.py --init_method file://public/home/mzh/maskl/navier_stokes --world_size 3 --rank 0

$ python -m torch.distributed.launch --master_port 29500 --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr=10.10.10.22 transfer_ns.py --world_size 4 --local_rank 0
$ python transfer_ns.py --local_rank 1 --world_size 1
"""
import logging
import os
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
from models import ON, DON, DON_lite
from utils import LpLoss, add_noise, mask_data, mean_mask_data

os.environ["CUDA_DEVICES_MAX_CONNECTIONS"]='1'
os.environ["OMP_NUM_THREADS"] = "1"

'''
os.environ["MASTER_PORT"] = "29501"
os.environ["MASTER_ADDR"] = "10.10.10.22"
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
'''


def args():
    import argparse
    parser = argparse.ArgumentParser(description="Mask Learning")

    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default="..data/ns_data.mat", type=str)
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--mask_times", default=4, type=int)
    parser.add_argument("--mask_rate", default=0.15, type=float)
    parser.add_argument("--model", default='DON', type=str)
    parser.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=2, help="world size")
    parser.add_argument('--init_method', default='tcp://10.10.10.22:29500',help="init-method")
    parser.add_argument('--rank', default=0, help='rank of current process')

    args = parser.parse_args()
    return args


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
                        torch.save(tmodel.module.state_dict(), "./checkpoint/re_mean0.4_t_0424" + ".pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 16 / T, train_l2_full / 16, test_l2_step / 4 / T, test_l2_full / 4)
            logging.info(
                f"epoch: {epoch}, mean_l2_step={train_l2_step / 16 / T}, mean_train_l2_full={train_l2_full / 16}, "
                f"mean_test_l2_step={test_l2_step / 4 / T}, mean_test_l2_full={test_l2_full / 4}")

    torch.save(tmodel.state_dict(), "./checkpoint/re_trans_10000" + ".pth")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(), "./runlog/mean_trans_0424" + ".log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    print(f"Support distributed environment: {dist.is_available()}")
    set_seed(42)

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    #device = torch.device('cuda')
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
    #dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=3, rank=args.rank)

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

    #test_a = mask_data(test_a, mask_rate=0.3)
    test_a = mean_mask_data(test_a, mask_rate=0.3)
    test_a = torch.tensor(test_a)
    test_u = torch.tensor(test_u)
    test_dataset = torch.utils.data.TensorDataset(test_a, test_u)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        # shuffle=False,
        sampler=test_sampler)
    del raw_data, test_a, test_u

    model = ON(in_features=train_a.shape[-1], width=20)
    if device == torch.device('cpu'):
        model_state_dict = torch.load('./checkpoint/re_mean_0.4mask_noise_False0423.pth', map_location=torch.device('cpu'))
    else:
        model_state_dict = torch.load('./checkpoint/re_mean_0.4mask_noise_False0423.pth')
    model.load_state_dict(model_state_dict, False)

    if args.model == 'DON':
        tmodel = DON(origin_model=model, in_features=model.in_feature, width=model.width).cuda()
    elif args.model == "DON_lite":
        tmodel = DON_lite(origin_model=model, in_features=model.in_feature, width=model.width).cuda()
    else:
        raise NotImplemented

    #for param in tmodel.origin_model.parameters():
        #param.requires_grad = False
    #tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[args.local_rank])  # single node
    tmodel = torch.nn.parallel.DistributedDataParallel(tmodel, device_ids=[args.local_rank], output_device=args.local_rank)  # multiply nodes
    loss_fn = LpLoss(size_average=False)

    optimizer = torch.optim.Adam(tmodel.module.new_layer.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    # mask_times = args.mask_times

    transfer_learning(tmodel=tmodel, epochs=epochs,
                      train_loader=train_loader, test_loader=test_loader)

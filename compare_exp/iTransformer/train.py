import os
import sys
import time
import logging
import numpy as np
import scipy
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from jinja2 import optimizer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

sys.path.append('/home/maozihao/maskl')
from iTransformer import iTransformer
from utils import set_seed, LpLoss
from compare_exp.fno.utilities3 import MatReader, UnitGaussianNormalizer, mask_burgers, mask_darcy, mask_ns


def args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Recover Net")

    parser.add_argument("--data", type=str, required=True, help="PDE data selection")
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sim", type=float, default=0.1, help="Similarity between origin and recovery.")
    parser.add_argument("--mask_rate", default=0.3, type=float)
    parser.add_argument("--master_port", default=20501, type=int)
    parser.add_argument("--resolution", type=int, help="Input resolution")

    parser.add_argument('--seq_len', type=int, default=40, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=20,
                        help='start token length')  # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=40, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=128, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=128, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=128,
                        help='output size')  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    args = parser.parse_args()
    return args


def train(model, t, args, train_loader, test_loader):
    better_loss = 10000000

    for epoch in range(args.epoch):
        train_l2_step = 0
        test_l2_step = 0
        model.train()

        for i, (mask_a, a, u) in enumerate(train_loader):
            u = u.float().to(args.device)
            a = a.float().to(args.device)
            batch_num = a.shape[0]
            t_in = t.repeat(batch_num, 1).unsqueeze(-1)
            #print(f"u: {u.shape}, a: {a.shape}, t: {t.shape}")
            im = model(a, t_in, t, t)
            # print(f"mask_a: {mask_a.shape}, a: {a.shape}, im: {im.shape}")
            train_loss = loss_fn(im, u)
            train_l2_step += train_loss.item()
                
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
        
        with torch.no_grad():
            model.eval()
            for i, (mask_a, a, u) in enumerate(test_loader):
                u = u.float().to(args.device)
                a = a.float().to(args.device)
                batch_num = a.shape[0]
                t_in = t.repeat(batch_num, 1).unsqueeze(-1)
                im = model(a, t_in, t, t)
                test_loss = loss_fn(im, u)
                test_l2_step += test_loss.item()

                if test_loss.item() < better_loss:
                    better_loss = test_loss.item()
                    torch.save(model.state_dict(), f"./results/iTransformer_{args.data}_{time.strftime('%m%d', time.localtime())}.pth")

        if epoch % 10 == 0:
            print(epoch, train_l2_step / 1000, test_l2_step / 100)
            logging.info(
                f"epoch: {epoch}, train_l2_step: {train_l2_step / 1000}, test_l2_step: {test_l2_step/100}")

    return model

    


if __name__ == "__main__":
    args = args()
    set_seed(13)
    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
        args.data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil)

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)
    
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                            f"log/iTransformer_{args.data}_{args.mask_rate}_{time.strftime('%m%d', time.localtime())}.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)

    if args.use_multi_gpu:
        os.environ["CUDA_DEVICES_MAX_CONNECTIONS"] = '1'
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MASTER_PORT"] = str(args.master_port)

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())
        print("Torch Distributed Data Parallel initialised successfully")
    else:
        device =  torch.device('cuda')

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
        time_t = torch.linspace(0, 1, 1).to(device)
        #time_t = time_t.repeat(args.batch_size, 1).unsqueeze(-1)

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
        
        model = iTransformer(args).float().cuda()
        if args.use_multi_gpu:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        else:
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        print("Start training")
        train(model, time_t, args, train_loader=train_loader, test_loader=test_loader)

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
        time_t = torch.linspace(0, 1, 1).to(device)
        #time_t = time_t.repeat(args.batch_size, 1).unsqueeze(-1)

        mask_x_train = mask_darcy(x_train, mask_rate=args.mask_rate).reshape(ntrain, 1, -1)
        mask_x_test = mask_darcy(x_test, mask_rate=args.mask_rate).reshape(ntest, 1, -1)

        x_train = x_train.reshape(ntrain, 1, -1)
        y_train = y_train.unsqueeze(1).reshape(ntrain, 1, -1)
        x_test = x_test.reshape(ntest, 1, -1)
        y_test = y_test.unsqueeze(1).reshape(ntest, 1, -1)

        train_dataset = torch.utils.data.TensorDataset(mask_x_train, x_train, y_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=train_sampler)
        
        test_dataset = torch.utils.data.TensorDataset(mask_x_test, x_test, y_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=test_sampler)
        
        model = iTransformer(args).float().cuda()
        if args.use_multi_gpu:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        else:
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        
        train(model, time_t, args, train_loader=train_loader, test_loader=test_loader)

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
        train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :T]
        train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, T:T + T_in].permute(0, 3, 1, 2)
        reader = MatReader(TEST_PATH)
        test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :T]
        test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, T:T + T_in].permute(0, 3, 1, 2)

        #print(train_u.shape)    # torch.Size([1000, 10, 64, 64])
        #print(test_u.shape)     # torch.Size([200, 10, 64, 64])

        train_a = train_a.reshape(ntrain, S, S, T_in).permute(0, 3, 1, 2)  # torch.Size([1000, 10, 64, 64])
        test_a = test_a.reshape(ntest, S, S, T_in).permute(0, 3, 1, 2)  # torch.Size([200, 10, 64, 64])
        time_t = torch.linspace(0, T, T)/T
        #time_t = time_t.repeat(args.batch_size, 1).unsqueeze(-1).to(device)
        time_t = time_t.to(device)

        mask_x_train = mask_ns(train_u, mask_rate=args.mask_rate).reshape(ntrain, T, -1)
        mask_x_test = mask_ns(test_u, mask_rate=args.mask_rate).reshape(ntest, T, -1)

        train_a = train_a.reshape(ntrain, T, -1)
        train_u = train_u.reshape(ntrain, T, -1)
        test_a = test_a.reshape(ntest, T_in, -1)
        test_u = test_u.reshape(ntest, T_in, -1)
        
        train_dataset = torch.utils.data.TensorDataset(mask_x_train, train_a, train_u)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=train_sampler)
        
        test_dataset = torch.utils.data.TensorDataset(mask_x_test, test_a, test_u)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=test_sampler)
        
        model = iTransformer(args).float().cuda()
        if args.use_multi_gpu:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        else:
            loss_fn = LpLoss(size_average=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        train(model=model, 
              t=time_t, 
              args=args, 
              train_loader=train_loader, 
              test_loader=test_loader)


    elif args.data == 'test':
        """
        python compare_exp.py --data test --seq_len 1 --pred_len 1
        """
        model = iTransformer.Model(args).float().to(device)
        x_in = torch.randn(20,1,1024).to(device)    # (batch_size, seq_len, enc_in)
        T = 1
        df_stamp = torch.arange(T) / T
        df_stamp = df_stamp.repeat(20, 1).unsqueeze(-1).to(device)
        x_mark = df_stamp
        output = model(x_in, x_mark, df_stamp, df_stamp)
        print(f"output: {output.shape}")    # (batch_size, pred_len, enc_in)    torch.Size([20, 1, 1024])

    else:
        raise NotImplementedError


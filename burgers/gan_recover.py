"""
Use GAN to recover data from corrupted data in Burgers Equation

$ torchrun --standalone --nnodes 1 --nproc_per_node 2 gan_recover.py
"""
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torch.autograd as autograd
import scipy
import numpy as np
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import LpLoss
from recover_burgers import get_grid, _mask_data, mean_mask_data, _set_seed, MLC, ON

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["OMP_NUM_THREADS"] = "1"
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def args():
    import argparse
    parse = argparse.ArgumentParser(description="Mask Learning")

    parse.add_argument("--epochs", default=100000, type=int)
    parse.add_argument("--batch_size", default=4, type=int)
    parse.add_argument("--data_path", default="./burgers_1d_1000a.mat", type=str)
    parse.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parse.add_argument("--mask_times", default=4, type=int)
    parse.add_argument("--width", default=80, type=int)
    parse.add_argument("--mask_rate", default=0.3, type=float)
    parse.add_argument("--noise", action="store_true", default=False, help="Add Random Gaussian Noise")
    parse.add_argument("--ON", action="store_true", default=False, help="Use ON model as generator")
    parse.add_argument("--GP", action="store_true", default=False, help="Use Gradient Penalty")
    parse.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parse.add_argument("--master_port", default=20501, type=int)

    args = parse.parse_args()
    return args


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0],real_samples.shape[1], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class GanRecover(nn.Module):
    def __init__(self, in_features, width):
        super(GanRecover, self).__init__()
        self.in_feature = in_features
        self.width = width
        self.p = nn.Linear(self.in_feature + 1, self.width)
        self.q = MLC(in_channels=self.width, out_channels=self.in_feature, mid_channels=self.width * 4)
        self.mlp0 = MLC(self.width, self.width, self.width)
        self.mlp1 = MLC(self.width, self.width, self.width)
        self.mlp2 = MLC(self.width, self.width, self.width)
        self.mlp3 = MLC(self.width, self.width, self.width)
        self.mlp4 = MLC(self.width, self.width, self.width)
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
        x = x + x1
        x1 = x
        x = self.gelu(self.mlp3(x))
        x = self.gelu(self.mlp4(x))
        x = x + x1
        x = self.q(x)  # back to original space

        return x


class GanDiscriminator(nn.Module):
    def __init__(self, in_features):
        super(GanDiscriminator, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leakyReLU(self.fc1(x))
        x = self.leakyReLU(self.fc2(x))
        x = self.leakyReLU(self.fc3(x))
        x = self.fc4(x)
        return x


def gan_mask_train(epochs, generator, discriminator, train_loader):
    better_loss = 100000
    for epoch in range(epochs):
        generator.train()

        for i, (masked_a, a, u) in enumerate(train_loader):
            masked_a = masked_a.to(device)
            a = a.to(device)
            u = u.to(device)
            im = generator(masked_a).detach()
            gradient_penalty = compute_gradient_penalty(discriminator, a, im)
            if not args.GP:
                loss_D = -torch.mean(discriminator(a)) + torch.mean(discriminator(im)) + loss_fn(a,im)
            else:
                loss_D = -torch.mean(discriminator(a)) + torch.mean(discriminator(im)) + 5 * gradient_penalty
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            scheduler_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if i % 3 == 0:
                optimizer_G.zero_grad()
                gen_im = generator(masked_a)
                loss_G = -torch.mean(discriminator(gen_im)) + loss_fn(a,gen_im)
                loss_G.backward()
                optimizer_G.step()
                scheduler_G.step()

                if loss_G < better_loss:
                    better_loss = loss_G
                    torch.save(generator.module.state_dict(),f"./results/checkpoint/generator_{args.mask_rate}_0525b.pth")


                logging.info(f"Epoch: {epoch}--{i}, loss_D: {loss_D:.4f}, loss_G: {loss_G:.4f}")
                print(f"Epoch: {epoch}--{i}, loss_D: {loss_D:.4f}, loss_G: {loss_G:.4f}; {-torch.mean(discriminator(gen_im))}; {loss_fn(a,gen_im)}")



if __name__ == "__main__":
    args = args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(os.getcwd(),
                                              f"./results/gan_recover_{args.mask_rate}" + f"_0525b.log"),
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('------------------------------------------------------------------------------------')
    logging.info('File path: {}'.format(os.path.abspath(__file__)))
    logging.info(args)
    _set_seed(42)
    os.environ["MASTER_PORT"] = str(args.master_port)

    DATA_PATH = args.data_path
    raw_data = scipy.io.loadmat(DATA_PATH)  # dict u:(20,256,256,200) a:(20,256,256) t:(1,200)
    batch_size = args.batch_size
    epochs = args.epochs
    iterations = epochs * (raw_data['data'].shape[0] // batch_size)

    N = raw_data['data'].shape[0]
    T = raw_data['data'].shape[2]
    train_a = torch.tensor(raw_data['data'][:, :, :int(0.8 * T)]).float()
    train_u = torch.tensor(raw_data['data'][:, :, int(0.8 * T):]).float()
    test_a = torch.tensor(raw_data['data'][int(0.8 * N):, :, :int(0.8 * T)]).float()  # torch.Size([50, 256, 800])
    test_u = torch.tensor(raw_data['data'][int(0.8 * N):, :, int(0.8 * T):]).float()
    del raw_data

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', world_size=torch.cuda.device_count())

    if not args.ON:
        generator = GanRecover(in_features=train_a.shape[-1], width=args.width).to(device)
    else:
        generator = ON(in_features=train_a.shape[-1], width=args.width).to(device)
    generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[local_rank], output_device=local_rank)
    discriminator = GanDiscriminator(in_features=train_a.shape[-1]).to(device)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank)

    loss_fn = LpLoss(size_average=False)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=iterations)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=iterations)
    mask_times = args.mask_times
    print(torch.distributed.is_initialized())

    for time in range(mask_times):
        masked_train_a = mean_mask_data(train_a, mask_rate=args.mask_rate)
        #print(f"masked_train_a: {masked_train_a.shape}, train_a: {train_a.shape}")
        train_dataset = torch.utils.data.TensorDataset(masked_train_a, train_a, train_u)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=train_sampler)

        gan_mask_train(epochs=epochs // mask_times,
                       generator=generator,
                       discriminator=discriminator,
                       train_loader=train_loader)

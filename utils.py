import random

import numpy as np
import torch
from typing import List

def mask_data(data: np.ndarray, mask_rate: float) -> np.ndarray:
    """
    Creat masked data, use 'mask_rate' to control proportion.
    Input should be a 4D Tensor: [sample_times, length, width, time]
    """
    masked_data = data.copy()
    data_T = data.shape[-1]
    mask_length = int(data_T * mask_rate)
    random_start = torch.randint(low=0, high=data_T - mask_length, size=(1,)).item()
    mask_token = np.argmax(np.abs(masked_data)) * 100
    # mas = torch.ones_like(data.shape[1],data.shape[2]) * 1024
    masked_data[:, :, :, random_start:random_start + mask_length].fill(mask_token)  # use a big number to mask original data

    return masked_data


def mean_mask_data(data: np.ndarray, mask_rate: float) -> np.ndarray:
    # masked_data = data.to('cpu').numpy().copy()
    masked_data = data.copy()
    data_T = masked_data.shape[-1]
    mask_length = int(data_T * mask_rate)
    random_start = torch.randint(low=0, high=data_T - mask_length, size=(1,)).item()
    mask_token = np.mean(masked_data)
    masked_data[:, :, :, random_start:random_start + mask_length].fill(mask_token)
    return masked_data


def add_noise(data: np.ndarray, variance: float) -> np.ndarray:
    """
    Add random gaussian noise into data
    """
    shape = data.shape
    noise = np.random.normal(loc=0, scale=np.sqrt(variance), size=(shape[0], shape[1], shape[2]))
    data[:, :, :, :shape[3]] += noise[:, :, :, np.newaxis]
    return data

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def releative_loss(a:torch.Tensor, b:torch.Tensor) -> float:
    """
    Compute the relative error
    a: true value [feature, width, height, time]
    b: predicted value
    """
    assert a.shape == b.shape
    n = a.shape[0]

    for i in range(n):
        res = torch.abs(a-b)
        rel = res / torch.abs(a)
    torch.mean()
    
def try_all_gpus():
    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]
    return devices

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLM:
    def __init__(self, *,padding_token: int, mask_token: int, no_mask_tokens: List[int], n_tokens: int,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1,
                 no_change_prob: float = 0.1,
                 ):
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        for t in self.no_mask_tokens:
            full_mask &= x != t
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)
        mask = full_mask & ~random_token_mask & ~unchanged
        y = x.clone()
        x.masked_fill_(mask, self.mask_token)
        x[random_token_idx] = random_tokens
        y.masked_fill_(~full_mask, self.padding_token)
        return x, y



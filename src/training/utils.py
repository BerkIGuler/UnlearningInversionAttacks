import torch
import random
import numpy as np

def set_random_seed(seed):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    random.seed(seed + 6)
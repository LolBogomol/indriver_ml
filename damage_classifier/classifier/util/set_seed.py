from damage_classifier.config import SEED
import numpy as np
import random
import torch


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

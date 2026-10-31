"""Cố định seed cho random / numpy / torch. (TV2)"""
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

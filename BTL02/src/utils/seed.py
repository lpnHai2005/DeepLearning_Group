"""Đặt seed cho mọi nguồn ngẫu nhiên. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T07."""
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # shortcut: chỉ bật cudnn.deterministic, không dùng use_deterministic_algorithms (một số toán tử CUDA không hỗ trợ);
    # chạy lại cùng seed trên GPU vẫn có thể lệch ở chữ số thập phân cuối, nâng cấp nếu T kiểm tái lập thấy lệch lớn.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

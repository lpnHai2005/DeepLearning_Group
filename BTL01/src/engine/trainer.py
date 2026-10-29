"""Vòng huấn luyện. (TV2 – T2.2)

fit() KHÔNG nhận dữ liệu test. Chọn checkpoint và dừng sớm chỉ theo val loss.
"""


def train_one_epoch(model, loader, opt, crit):
    """zero_grad -> forward -> loss -> backward -> step; trả loss trung bình theo trọng số batch."""
    raise NotImplementedError("TV2 - T2.2")


def validate(model, loader, crit):
    """model.eval() + torch.no_grad(); trả loss trung bình theo trọng số batch."""
    raise NotImplementedError("TV2 - T2.2")


def fit(model, train_loader, val_loader, cfg, run_dir):
    """-> dict(best_epoch, epochs_run, train_time_s). Ghi history.csv và best.pt vào run_dir."""
    raise NotImplementedError("TV2 - T2.2")

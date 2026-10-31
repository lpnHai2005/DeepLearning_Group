"""Vòng huấn luyện.
fit() KHÔNG nhận dữ liệu test. Chọn checkpoint và dừng sớm chỉ theo val loss.
"""
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.io import write_csv
from src.utils.timer import Timer

OPTS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def train_one_epoch(model, loader, opt, crit):
    """zero_grad -> forward -> loss -> backward -> step; trả loss trung bình theo trọng số batch."""
    model.train()
    tot = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        assert pred.shape == yb.shape, f"{pred.shape} != {yb.shape}"   # (N,1) vs (N,) sẽ broadcast sai
        loss = crit(pred, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
    return tot / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, crit):
    """model.eval() + torch.no_grad(); trả loss trung bình theo trọng số batch."""
    model.eval()
    tot = 0.0
    for xb, yb in loader:
        pred = model(xb)
        assert pred.shape == yb.shape, f"{pred.shape} != {yb.shape}"
        tot += crit(pred, yb).item() * xb.size(0)
    return tot / len(loader.dataset)


def fit(model, train_loader, val_loader, cfg, run_dir):
    """-> dict(best_epoch, epochs_run, train_time_s, best_val_loss). Ghi history.csv và best.pt vào run_dir.

    cfg cần: optimizer, lr, max_epochs, patience; tùy chọn min_epochs (mặc định 0).
    Khi trả về, model đã được nạp trọng số tốt nhất theo val. Loss ghi trên thang đã scale.
    """
    run_dir = Path(run_dir)
    opt = OPTS[cfg["optimizer"]](model.parameters(), lr=cfg["lr"])
    crit = nn.MSELoss()
    weights = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]
    best, best_epoch, best_state, wait, history = math.inf, 0, None, 0, []

    with Timer() as t:
        for epoch in range(1, cfg["max_epochs"] + 1):
            tr = train_one_epoch(model, train_loader, opt, crit)
            va = validate(model, val_loader, crit)
            row = {"epoch": epoch, "train_loss": tr, "val_loss": va}
            # chuẩn gradient từng lớp ở batch cuối của epoch: bằng chứng để nói (hoặc không) về vanishing gradient
            row.update({f"grad_W{i + 1}": w.grad.norm().item() for i, w in enumerate(weights)})
            history.append(row)
            if va < best:
                best, best_epoch, wait = va, epoch, 0
                best_state = copy.deepcopy(model.state_dict())   # deepcopy: state_dict là tham chiếu
            else:
                wait += 1
                if wait >= cfg["patience"] and epoch >= cfg.get("min_epochs", 0):
                    break

    model.load_state_dict(best_state)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "best.pt")
    write_csv(run_dir / "history.csv", history)
    return {"best_epoch": best_epoch, "epochs_run": epoch, "train_time_s": t.elapsed, "best_val_loss": best}

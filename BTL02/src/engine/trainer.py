"""Vòng huấn luyện CNN cho FER2013: fit(), luật giữ/xóa weights, quét run bị kill.

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T07.

Hợp đồng (mục 4 PHAN_CONG_NHIEM_VU.md): fit(model, train_loader, val_loader, cfg, run_dir) -> dict.
Mỗi loader trả từng batch (x float (B,C,H,W), y int64 (B,)). fit không nhận PrivateTest.
Mỗi run để lại trong run_dir: run.log, history.csv, metrics.json và best.pt (chỉ khi run hợp lệ, mục 9.3).
fit KHÔNG ném ngoại lệ do huấn luyện: lỗi được ghi (Failed/Aborted) vào metrics.json và run.log rồi trả về, để
vòng lặp nhiều run chạy tiếp. Ctrl+C cũng được ghi (Aborted) rồi trả về; người gọi tự quyết dừng hay tiếp.

Khóa cfg (mặc định trong ngoặc): epochs (60), optimizer ("sgd"|"adam"|"adamw"|"rmsprop", "sgd"), lr (0.01),
momentum (0.9), nesterov (True), weight_decay (5e-4), scheduler ("plateau"|"constant"|"step"|"cosine"|"onecycle",
"plateau"), factor (0.5) và lr_patience (5) cho plateau, step_size (20) và gamma (0.5) cho step,
early_stopping_patience (15; 0 là tắt), label_smoothing (0.0), class_weights (None hoặc list 7 số), amp (True,
chỉ có tác dụng trên CUDA), device (None: tự chọn), exp_id (mặc định tên run_dir).
Loss train có label_smoothing/class_weights; loss val luôn là CE thường để so sánh được giữa các run.
"""
import copy
import csv
import logging
import math
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from src.models.cnn import NUM_CLASSES
from src.utils.io import atomic_torch_save, sha256_file, write_json

HISTORY_COLS = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1", "lr", "epoch_time_s"]
# shortcut: dung sai 1e-4 (nhỏ hơn 1 mẫu của PublicTest = 2.8e-4, tức val_acc tính lại phải khớp từng mẫu); nhóm chưa
# chốt dung sai, nới ra nếu AMP fp16 làm lật 1 mẫu giữa hai lần chạy.
RELOAD_TOL = 1e-4


class _NanLoss(Exception):
    def __init__(self, epoch):
        super().__init__(f"loss không hữu hạn ở epoch {epoch}")
        self.epoch = epoch


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _open_log(run_dir, exp_id):
    if hasattr(sys.stdout, "reconfigure"):  # console Windows mặc định cp1252, thông báo lỗi tiếng Việt sẽ làm hỏng log
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logger = logging.getLogger(f"fit.{Path(run_dir).resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in (logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    return logger


def _log(logger, exp_id, *fields):
    logger.info(" | ".join([_now(), exp_id, *map(str, fields)]))


def _close_log(logger):
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def macro_f1(conf):
    """Macro-F1 từ ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán). Lớp không có mẫu và không được đoán: F1 = 0."""
    conf = conf.double()
    return (2 * conf.diag() / (conf.sum(0) + conf.sum(1)).clamp(min=1)).mean().item()


def _run_epoch(model, loader, device, criterion, amp, optimizer=None, scaler=None, batch_sched=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if batch_sched is not None:
                    batch_sched.step()
            pred = out.argmax(1)
            loss_sum += loss.detach().float() * len(y)
            correct += (pred == y).sum()
            conf += torch.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            n += len(y)
    if n == 0:
        raise ValueError("loader không có mẫu nào")
    return loss_sum.item() / n, correct.item() / n, conf


def _make_optimizer(params, cfg):
    name, lr, wd = cfg.get("optimizer", "sgd"), cfg.get("lr", 0.01), cfg.get("weight_decay", 5e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg.get("momentum", 0.9), nesterov=cfg.get("nesterov", True),
                               weight_decay=wd)
    makers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop}
    if name not in makers:
        raise ValueError(f"optimizer '{name}' không hợp lệ, chọn một trong {['sgd', *makers]}")
    return makers[name](params, lr=lr, weight_decay=wd)


def _make_scheduler(opt, cfg, epochs, steps_per_epoch):
    """Trả (scheduler, theo_batch). plateau theo dõi val_acc (mode max); onecycle bước theo batch."""
    name = cfg.get("scheduler", "plateau")
    if name == "constant":
        return None, False
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg.get("factor", 0.5),
                                                          patience=cfg.get("lr_patience", 5)), False
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("step_size", 20), gamma=cfg.get("gamma", 0.5)), False
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)), False
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 0.01),
                                                   total_steps=max(epochs * steps_per_epoch, 1)), True
    raise ValueError(f"scheduler '{name}' không hợp lệ")


def _read_history(run_dir):
    p = Path(run_dir) / "history.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _reload_val_acc(model, path, val_loader, device, amp):
    m = copy.deepcopy(model)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.to(device)
    return _run_epoch(m, val_loader, device, nn.CrossEntropyLoss(), amp)[1]


def finalize_weights(run_dir, fail_reason, recompute_val_acc):
    """Luật giữ/xóa weights (mục 9.3), kiểm theo thứ tự; luật không xét metric.

    fail_reason: None hoặc mã lý do do fit đặt (EXCEPTION:<lỗi> | NAN_LOSS@epochN | INTERRUPTED@epochN).
    recompute_val_acc: hàm không tham số, nạp best.pt và trả val_acc tính lại trên tập val.
    Chỉ đụng thư mục của chính run này. Trả dict các cột weights_*."""
    run_dir = Path(run_dir)
    best = run_dir / "best.pt"
    reason = fail_reason
    if reason is None and not best.exists():
        reason = "NO_CHECKPOINT"
    if reason is None:
        hist = _read_history(run_dir)
        top = max(range(len(hist)), key=lambda i: (hist[i]["val_acc"], -i))  # epoch val_acc cao nhất, sớm nhất
        try:
            if abs(recompute_val_acc() - hist[top]["val_acc"]) > RELOAD_TOL:
                reason = "RELOAD_MISMATCH"
        except Exception as e:  # best.pt hỏng hoặc không nạp được
            reason = f"EXCEPTION:{type(e).__name__}"
    if reason is not None:
        for f in run_dir.glob("*.pt*"):  # *.pt và *.pt.tmp dở; giữ config.json, run.log, history.csv
            f.unlink()
        return dict(weights_quyet_dinh="DISCARD", weights_ma_ly_do=reason, weights_sha256="", weights_size_mb="")
    return dict(weights_quyet_dinh="KEEP", weights_ma_ly_do="OK", weights_sha256=sha256_file(best),
                weights_size_mb=round(best.stat().st_size / 2 ** 20, 3))


def _finish(run_dir, exp_id, logger, status, fail_reason, error_tb, stopped_by, recompute, t0):
    run_dir = Path(run_dir)
    hist = _read_history(run_dir)
    w = finalize_weights(run_dir, fail_reason, recompute)
    top = max(range(len(hist)), key=lambda i: (hist[i]["val_acc"], -i)) if hist else None
    last = hist[-1] if hist else {}
    m = dict(
        exp_id=exp_id, status=status, stopped_by=stopped_by, epochs_run=len(hist),
        best_epoch=int(hist[top]["epoch"]) if hist else 0,
        best_val_acc=hist[top]["val_acc"] if hist else None,
        best_val_macro_f1=hist[top]["val_macro_f1"] if hist else None,
        best_val_loss=min(r["val_loss"] for r in hist) if hist else None,
        train_loss_cuoi=last.get("train_loss"), train_acc_cuoi=last.get("train_acc"), val_loss_cuoi=last.get("val_loss"),
        epoch_time_s_tb=sum(r["epoch_time_s"] for r in hist) / len(hist) if hist else None,
        train_time_s=round(time.perf_counter() - t0, 3), **w,
    )
    write_json(run_dir / "metrics.json", m)
    if status == "Failed":
        _log(logger, exp_id, "ERROR", f"status={status}", f"reason={w['weights_ma_ly_do']}")
        if error_tb:
            logger.info(error_tb.rstrip())
    else:
        _log(logger, exp_id, "END", f"status={status}", f"reason={w['weights_ma_ly_do']}", f"stopped_by={stopped_by}",
             f"best_epoch={m['best_epoch']}", f"train_time_s={m['train_time_s']}")
    _close_log(logger)
    return m


def fit(model, train_loader, val_loader, cfg, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        raise FileExistsError(f"{hist_path} đã có: dùng exp_id/run_dir mới, không ghi đè run cũ")
    exp_id = cfg.get("exp_id", run_dir.name)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    epochs = int(cfg.get("epochs", 60))
    patience = int(cfg.get("early_stopping_patience", 15))
    best_path = run_dir / "best.pt"
    logger = _open_log(run_dir, exp_id)
    hf = open(hist_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(hf)
    writer.writerow(HISTORY_COLS)
    hf.flush()
    t0 = time.perf_counter()
    status, fail_reason, error_tb, stopped_by = "Aborted", "INTERRUPTED", None, "interrupted"
    n_rows, best_acc, stale = 0, -1.0, 0
    try:
        try:
            model.to(device)
            optimizer = _make_optimizer([p for p in model.parameters() if p.requires_grad], cfg)
            scheduler, per_batch = _make_scheduler(optimizer, cfg, epochs, len(train_loader))
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            weight = cfg.get("class_weights")
            criterion = nn.CrossEntropyLoss(weight=None if weight is None else torch.tensor(weight, dtype=torch.float32, device=device),
                                            label_smoothing=cfg.get("label_smoothing", 0.0))
            val_criterion = nn.CrossEntropyLoss()
            _log(logger, exp_id, "START", f"device={device}", f"epochs={epochs}", f"amp={amp}",
                 f"optimizer={cfg.get('optimizer', 'sgd')}", f"lr={cfg.get('lr', 0.01)}", f"seed={cfg.get('seed')}")
            stopped_by = "max_epochs"
            for epoch in range(1, epochs + 1):
                t = time.perf_counter()
                tl, ta, _ = _run_epoch(model, train_loader, device, criterion, amp, optimizer, scaler,
                                       scheduler if per_batch else None)
                vl, va, conf = _run_epoch(model, val_loader, device, val_criterion, amp)
                if not (math.isfinite(tl) and math.isfinite(vl)):
                    raise _NanLoss(epoch)
                f1, lr, dt = macro_f1(conf), optimizer.param_groups[0]["lr"], time.perf_counter() - t
                writer.writerow([epoch, tl, vl, ta, va, f1, lr, round(dt, 3)])
                hf.flush()
                n_rows += 1
                if va > best_acc:
                    best_acc, stale = va, 0
                    atomic_torch_save(model.state_dict(), best_path)
                else:
                    stale += 1
                _log(logger, exp_id, f"{epoch}/{epochs}", f"train_loss={tl:.4f}", f"val_loss={vl:.4f}", f"val_acc={va:.4f}",
                     f"val_macro_f1={f1:.4f}", f"lr={lr:.3g}", f"{dt:.1f}s", "ok")
                if scheduler is not None and not per_batch:
                    scheduler.step(va) if cfg.get("scheduler", "plateau") == "plateau" else scheduler.step()
                if patience and stale >= patience:
                    stopped_by = "early_stopping"
                    break
            status, fail_reason = "Completed", None
        except KeyboardInterrupt:
            pass  # giữ mặc định Aborted/INTERRUPTED
        except _NanLoss as e:
            status, fail_reason, stopped_by = "Failed", f"NAN_LOSS@epoch{e.epoch}", "nan_loss"
        except Exception as e:
            status, fail_reason, stopped_by = "Failed", f"EXCEPTION:{type(e).__name__}", "exception"
            error_tb = traceback.format_exc()
    finally:
        hf.close()
        if fail_reason == "INTERRUPTED":
            fail_reason = f"INTERRUPTED@epoch{n_rows}"
        result = _finish(run_dir, exp_id, logger, status, fail_reason, error_tb, stopped_by,
                         lambda: _reload_val_acc(model, best_path, val_loader, device, amp), t0)
    return result


def record_failure(run_dir, exp_id, exc):
    """Ghi một run lỗi xảy ra TRƯỚC khi vào fit (config sai, loader lỗi): run.log, metrics.json, không để lại .pt."""
    run_dir = Path(run_dir)
    logger = _open_log(run_dir, exp_id)
    _log(logger, exp_id, "START", "giai đoạn=chuẩn bị (trước fit)")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _finish(run_dir, exp_id, logger, "Failed", f"EXCEPTION:{type(exc).__name__}", tb, "exception", lambda: 0.0, time.perf_counter())


def scan_orphans(results_dir, min_idle_s=1800):
    """Run bị kill cứng (có dòng START, không có END/ERROR): đánh INTERRUPTED@epoch<N cuối>, xóa *.pt dở.
    Trả danh sách run_dir vừa xử lý.

    shortcut: run.log phải im lặng ít nhất min_idle_s (30 phút) mới coi là bị kill, để không phá run đang chạy ở
    cửa sổ khác trên cùng máy; nếu một epoch lâu hơn 30 phút thì tăng giá trị này."""
    done = []
    for log in sorted(Path(results_dir).glob("*/*/run.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        if " | START | " not in text or " | END | " in text or " | ERROR | " in text:
            continue
        if time.time() - log.stat().st_mtime < min_idle_s:
            continue
        run_dir, exp_id = log.parent, log.parent.name
        hist = _read_history(run_dir)
        logger = _open_log(run_dir, exp_id)
        _finish(run_dir, exp_id, logger, "Aborted", f"INTERRUPTED@epoch{len(hist)}", None, "killed", lambda: 0.0,
                time.perf_counter() - sum(r["epoch_time_s"] for r in hist))
        done.append(run_dir)
    return done

# Phụ trách: TV3 – Võ Hoàng Bảo
"""Đánh giá mô hình FER2013: dự đoán (hỗ trợ TTA), chỉ số accuracy/macro-F1/per-class và ma trận nhầm lẫn.

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T08.

Hợp đồng (mục 4 PHAN_CONG_NHIEM_VU.md):
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predict(model, loader, tta=False) -> (y_true, y_prob)
    metrics(y_true, y_pred) -> dict(acc, macro_f1, per_class, cm, cm_norm, labels)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Thứ tự nhãn chuẩn Kaggle (0..6); dùng thống nhất trên toàn bộ repo để tránh lệch tên lớp.
from src.data.fer2013 import EMOTIONS


def _ten_crops(x: torch.Tensor) -> list[torch.Tensor]:
    """Tạo 10 crop (4 góc + tâm, bản gốc & lật ngang) rồi phóng về kích thước (H, W) ban đầu."""
    _, _, h, w = x.shape
    dh = max(2, h // 12)
    dw = max(2, w // 12)
    ch, cw = max(1, h - dh), max(1, w - dw)
    cy, cx = (h - ch) // 2, (w - cw) // 2
    boxes = [
        (0, 0, ch, cw),
        (0, w - cw, ch, w),
        (h - ch, 0, h, cw),
        (h - ch, w - cw, h, w),
        (cy, cx, cy + ch, cx + cw),
    ]
    crops = []
    for y0, x0, y1, x1 in boxes:
        patch = F.interpolate(x[:, :, y0:y1, x0:x1], size=(h, w), mode="bilinear", align_corners=False)
        crops.append(patch)
        crops.append(torch.flip(patch, dims=[-1]))
    return crops


def _forward_probs(model: torch.Tensor, x: torch.Tensor, tta_mode: str, amp: bool, device_type: str) -> torch.Tensor:
    with torch.autocast(device_type, enabled=amp):
        if tta_mode == "none":
            return torch.softmax(model(x), dim=1)
        if tta_mode == "flip":
            p1 = torch.softmax(model(x), dim=1)
            p2 = torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)
            return 0.5 * (p1 + p2)
        if tta_mode == "ten_crop":
            crops = _ten_crops(x)
            prob_sum = None
            for crop in crops:
                p = torch.softmax(model(crop), dim=1)
                prob_sum = p if prob_sum is None else (prob_sum + p)
            return prob_sum / len(crops)
    raise ValueError(f"Chế độ TTA '{tta_mode}' không hợp lệ; chọn False/True/'flip'/'ten_crop'")


def _normalize_tta(tta) -> str:
    if tta is False or tta is None or tta == "none":
        return "none"
    if tta is True or tta == "flip":
        return "flip"
    if tta in ("ten_crop", "10crop", "ten-crop"):
        return "ten_crop"
    raise ValueError(f"Chế độ TTA '{tta}' không hợp lệ; chọn False/True/'flip'/'ten_crop'")


def predict(model, loader, tta=False, device=None, amp=True):
    """Chạy suy luận trên loader, trả (y_true int64 (N,), y_prob float32 (N, C))."""
    tta_mode = _normalize_tta(tta)
    dev = torch.device(device) if device is not None else next(model.parameters(), torch.empty(0)).device
    if dev.type == "meta":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp) and dev.type == "cuda"

    model.eval()
    model.to(dev)
    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            p = _forward_probs(model, x, tta_mode, use_amp, dev.type)
            ys.append(torch.as_tensor(y, dtype=torch.int64).cpu().numpy())
            probs.append(p.float().cpu().numpy())
    if not ys:
        raise ValueError("loader không có mẫu nào để đánh giá")
    return np.concatenate(ys, axis=0).astype(np.int64), np.concatenate(probs, axis=0).astype(np.float32)


def metrics(y_true, y_pred, labels=EMOTIONS):
    """Tính accuracy, macro-F1, chỉ số từng lớp và ma trận nhầm lẫn (thô + chuẩn hóa theo hàng).

    y_pred có thể là nhãn 1D (N,) hoặc ma trận xác suất/logits 2D (N, C).
    """
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp_arr = np.asarray(y_pred)
    if yp_arr.ndim == 2:
        yp = yp_arr.argmax(axis=1).astype(np.int64)
    else:
        yp = yp_arr.astype(np.int64).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"Số mẫu không khớp: y_true={yt.shape[0]} vs y_pred={yp.shape[0]}")
    if yt.shape[0] == 0:
        raise ValueError("Không có mẫu nào để tính chỉ số")

    n_classes = len(labels)
    if (yt < 0).any() or (yt >= n_classes).any() or (yp < 0).any() or (yp >= n_classes).any():
        raise ValueError(f"Nhãn phải nằm trong [0, {n_classes - 1}]")

    cm = np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, pred_count, out=np.zeros(n_classes, dtype=np.float64), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros(n_classes, dtype=np.float64), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros(n_classes, dtype=np.float64), where=denom > 0)

    per_class = {}
    for i, name in enumerate(labels):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "acc": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "cm": cm,
        "cm_norm": cm_norm,
        "labels": list(labels),
    }


def save_predictions(out_path, y_true, y_prob):
    """Lưu dự đoán ra .npz để kiểm thử độc lập (T) hoặc phân tích lỗi/Grad-CAM (T20/T23)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_prob=np.asarray(y_prob, dtype=np.float32),
        y_pred=np.asarray(y_prob).argmax(axis=1).astype(np.int64),
    )
    return out_path

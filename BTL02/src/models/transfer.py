"""ResNet18 / VGG16 của torchvision cho FER2013 (7 lớp), dùng cho thí nghiệm E6 (transfer learning).

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T06 (E6 ở T17).

Đầu vào là ảnh 3 kênh (loader phải phóng ảnh và nhân 3 kênh, khác với cnn.py nhận 1x48x48). Mô hình trả về
*logits* (N, 7). Lớp cuối được thay bằng Linear(..., 7); đó là "head" mới, mọi lớp còn lại là "backbone".
  pretrained=True       nạp trọng số ImageNet (cần mạng lần đầu, torchvision lưu vào ~/.cache/torch). Khi đó dữ liệu phải
                        chuẩn hóa theo mean/std của ImageNet nếu muốn đúng với cách mô hình được huấn luyện.
  freeze_backbone=True  chỉ huấn luyện head; backbone requires_grad=False; BatchNorm của backbone được chuyển sang eval
                        mỗi lần forward (không cập nhật thống kê chạy), kể cả khi gọi model.train(). Dropout của
                        classifier VGG vẫn hoạt động khi train.

"Fail case" VGG16 ở 48 px: 5 lần MaxPool đưa ảnh 48 -> 24 -> 12 -> 6 -> 3 -> 1, torchvision phải phóng bản đồ 1x1 lên 7x7
(AdaptiveAvgPool) trước khi vào FC. Xem bằng: python -m src.models.transfer --size 48

In số tham số và kích thước bản đồ đặc trưng: python -m src.models.transfer [--size 96] [--out results/transfer_summary.csv]
"""
import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .cnn import NUM_CLASSES, n_params

# tên -> (hàm tạo, lớp Weights ImageNet)
ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "vgg16_bn": (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
}


def _frozen_bn_eval(model, _inputs):
    """Hook module-level (không dùng closure) nên deepcopy và pickle vẫn đúng: BatchNorm bị đóng băng luôn dùng thống kê
    đã học và không cập nhật running stats, kể cả khi model.train()."""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


def build_transfer(name="resnet18", pretrained=False, freeze_backbone=False, num_classes=NUM_CLASSES):
    if name not in ARCHS:
        raise ValueError(f"không có kiến trúc '{name}', chọn một trong {sorted(ARCHS)}")
    make, weights = ARCHS[name]
    m = make(weights=weights if pretrained else None)
    if name == "resnet18":
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    else:
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        head = m.classifier[-1]
    if freeze_backbone:
        head_ids = {id(p) for p in head.parameters()}
        for p in m.parameters():
            p.requires_grad = id(p) in head_ids
        m.register_forward_pre_hook(_frozen_bn_eval)
    return m


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model, input_shape=(1, 3, 96, 96)):
    """(tên, kích thước đầu ra, số tham số) của từng giai đoạn: ResNet theo các con trực tiếp (conv1, layer1..4, fc...),
    VGG theo từng MaxPool + avgpool + classifier. Kích thước lấy bằng hook khi chạy thật."""
    if hasattr(model, "fc"):
        stages = list(model.named_children())
    else:
        stages = [(f"features.{i}_pool", l) for i, l in enumerate(model.features) if isinstance(l, nn.MaxPool2d)]
        stages += [("avgpool", model.avgpool), ("classifier", model.classifier)]
    rows, hooks = [], []
    for name, m in stages:
        hooks.append(m.register_forward_hook(lambda mod, inp, out, name=name: rows.append(
            (name, tuple(out.shape[1:]), sum(p.numel() for p in mod.parameters())))))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return rows


def _write_summary(path, cases):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "input_size", "layer", "output_shape", "params_layer", "params_total"])
        for name, size in cases:
            m = build_transfer(name)
            for layer, shape, p in describe(m, (1, 3, size, size)):
                w.writerow([name, size, layer, "x".join(map(str, shape)), p, n_params(m)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In số tham số và kích thước bản đồ đặc trưng của ResNet18/VGG16")
    ap.add_argument("--size", type=int, default=96, help="cạnh ảnh đầu vào (mặc định 96)")
    ap.add_argument("--out", help="ghi chi tiết ra CSV cho các cặp (kiến trúc, kích thước) dùng trong E6")
    a = ap.parse_args()
    for n in ARCHS:
        m = build_transfer(n)
        sizes = [(l, s) for l, s, _ in describe(m, (1, 3, a.size, a.size)) if s != (NUM_CLASSES,)]
        print(f"{n:9s} {a.size}px tham số={n_params(m):>12,d}  " + " -> ".join(f"{l}:{'x'.join(map(str, s))}" for l, s in sizes[:8]))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _write_summary(a.out, [("resnet18", 96), ("vgg16", 48), ("vgg16", 96), ("vgg16_bn", 96)])
        print("đã ghi", a.out)

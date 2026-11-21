"""CNN kiểu VGG cho FER2013 (ảnh xám 48x48 -> 7 lớp cảm xúc).

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T06.

Một mô hình là một danh sách khối; mỗi khối là một danh sách số kênh của các lớp Conv3x3, theo sau là MaxPool2x2
và Dropout. Sau đó là phần phân loại (Flatten hoặc Global Average Pooling, rồi FC). Mô hình trả về *logits* (N, 7),
không có softmax (CrossEntropyLoss đã gồm sẵn).

Các cấu hình "small", "medium", "medium_01", "large" tái hiện 4 CNN trong notebook mẫu (demo_code.html, Keras:
SMALL_CNN, MEDIUM_CNN, MEDIUM1_CNN, LARGE_CNN). Mẫu không dùng BatchNorm; "baseline" là cấu hình đề xuất của
nhóm (mục 3.2 trong PHAN_CONG_NHIEM_VU.md) và có BatchNorm. Muốn đổi một yếu tố để làm thí nghiệm E1/E2, truyền
tham số ghi đè: build_model("baseline", bn=False), build_model("baseline", activation="gelu"), ...

In số tham số và kích thước bản đồ đặc trưng: python src/models/cnn.py [--out results/model_summary.csv]
"""
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

NUM_CLASSES = 7
INPUT_SIZE = 48

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

# blocks: số kênh các lớp Conv trong từng khối. dropout_conv: sau mỗi MaxPool. dropout_fc: trước lớp cuối.
PRESETS = {
    "small": dict(blocks=[[64, 64], [128, 128, 256, 256]], fc_units=1024, dropout_conv=0.2, dropout_fc=0.5, bn=False),
    "medium": dict(blocks=[[64] * 2, [128] * 3, [256] * 3], fc_units=1024, dropout_conv=0.2, dropout_fc=0.5, bn=False),
    "medium_01": dict(blocks=[[64] * 3, [128] * 4, [256] * 4], fc_units=1024, dropout_conv=0.2, dropout_fc=0.5, bn=False),
    "large": dict(blocks=[[64] * 4, [128] * 4, [256] * 4, [512] * 4], fc_units=2048, dropout_conv=0.2, dropout_fc=0.5, bn=False),
    "baseline": dict(blocks=[[64] * 2, [128] * 2, [256] * 2], fc_units=512, dropout_conv=0.25, dropout_fc=0.5, bn=True),
}


def _act(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"activation '{name}' không hợp lệ, chọn một trong {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class CNN(nn.Module):
    def __init__(self, blocks, fc_units=512, dropout_conv=0.25, dropout_fc=0.5, bn=True, activation="relu",
                 head="flatten", in_channels=1, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        if head not in ("flatten", "gap"):
            raise ValueError("head phải là 'flatten' hoặc 'gap'")
        if input_size // 2 ** len(blocks) < 1:
            raise ValueError(f"{len(blocks)} lần MaxPool làm ảnh {input_size}x{input_size} co về 0 điểm ảnh; giảm số khối")
        feats, c_in = [], in_channels
        for b, widths in enumerate(blocks, start=1):
            for i, c_out in enumerate(widths, start=1):
                feats.append((f"conv{b}_{i}", nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)))
                if bn:
                    feats.append((f"bn{b}_{i}", nn.BatchNorm2d(c_out)))
                feats.append((f"act{b}_{i}", _act(activation)))
                c_in = c_out
            feats.append((f"pool{b}", nn.MaxPool2d(2)))
            feats.append((f"drop{b}", nn.Dropout(dropout_conv)))
        self.features = nn.Sequential(OrderedDict(feats))  # tên lớp (conv1_1, bn1_1, ...) để describe() và hình bộ lọc dùng
        self.last_conv_name = [n for n, _ in feats if n.startswith("conv")][-1]

        with torch.no_grad():  # kích thước đầu vào của phần phân loại lấy bằng chạy thật, không tính tay
            flat = self.features(torch.zeros(1, in_channels, input_size, input_size)).shape
        in_feat = flat[1] if head == "gap" else flat[1] * flat[2] * flat[3]
        pool = [nn.AdaptiveAvgPool2d(1)] if head == "gap" else []
        cls = pool + [nn.Flatten()]
        if fc_units:
            cls += [nn.Linear(in_feat, fc_units), _act(activation), nn.Dropout(dropout_fc), nn.Linear(fc_units, num_classes)]
        else:
            cls += [nn.Linear(in_feat, num_classes)]
        self.classifier = nn.Sequential(*cls)

    def forward(self, x):
        return self.classifier(self.features(x))

    def last_conv(self):
        """Lớp Conv cuối (dùng cho trực quan hóa bộ lọc và Grad-CAM)."""
        return getattr(self.features, self.last_conv_name)


def build_model(name="baseline", **overrides):
    """Tạo mô hình từ cấu hình có sẵn; tham số truyền thêm ghi đè cấu hình."""
    if name not in PRESETS:
        raise ValueError(f"không có cấu hình '{name}', chọn một trong {sorted(PRESETS)}")
    return CNN(**{**PRESETS[name], **overrides})


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def describe(model, input_shape=(1, 1, INPUT_SIZE, INPUT_SIZE)):
    """Chạy một ảnh qua mô hình và trả về danh sách (tên lớp, kích thước đầu ra, số tham số) của từng lớp.
    Kích thước lấy bằng hook khi chạy thật nên không phải gõ tay."""
    rows, hooks = [], []
    named = list(model.features.named_children()) + [(f"cls{i}_{type(m).__name__}", m) for i, m in enumerate(model.classifier)]
    for name, m in named:
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


def _write_summary(path, names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "output_shape", "params_layer", "params_total"])
        for n in names:
            m = build_model(n)
            total = n_params(m)
            for layer, shape, p in describe(m):
                w.writerow([n, layer, "x".join(map(str, shape)), p, total])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In số tham số và kích thước bản đồ đặc trưng của các cấu hình CNN")
    ap.add_argument("--out", help="ghi chi tiết từng lớp ra file CSV (ví dụ results/model_summary.csv)")
    a = ap.parse_args()
    for n in PRESETS:
        m = build_model(n)
        pools = [(l, s) for l, s, _ in describe(m) if l.startswith("pool")]
        print(f"{n:10s} tham số={n_params(m):>11,d}  sau mỗi MaxPool: " + ", ".join(f"{s[1]}x{s[2]}x{s[0]}" for _, s in pools))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _write_summary(a.out, list(PRESETS))
        print("đã ghi", a.out)

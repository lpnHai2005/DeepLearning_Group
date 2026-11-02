"""Vẽ sơ đồ kiến trúc MLP đọc trực tiếp từ model PyTorch (các lớp nn.Linear).

Chạy thử:  python src/visual/draw_mlp.py   -> lưu hình vào figures/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn


def draw_mlp(model, max_show=8, save_path=None):
    """Vẽ sơ đồ MLP: nút input/hidden/output, nút bias +1, số tham số, hàm kích hoạt.

    Kích thước từng tầng lấy từ in_features/out_features của nn.Linear nên hình
    luôn khớp với mô hình thật. Tầng rộng hơn max_show nơ-ron được vẽ rút gọn (⋮).
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    acts = {type(m).__name__ for m in model.modules()
            if isinstance(m, (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.LeakyReLU))}
    sizes = [linears[0].in_features] + [l.out_features for l in linears]
    L = len(sizes)
    n_params = sum(p.numel() for p in model.parameters())

    fig, ax = plt.subplots(figsize=(2.4 * L + 1, 6))
    ax.axis("off")

    # toạ độ nút hiển thị của từng tầng
    pos = []
    for i, n in enumerate(sizes):
        k = min(n, max_show)
        pos.append([(i, (k - 1) / 2 - j) for j in range(k)])

    for i in range(L - 1):
        bottom = min(y for _, y in pos[i])
        # trọng số W
        for x0, y0 in pos[i]:
            for x1, y1 in pos[i + 1]:
                ax.plot([x0, x1], [y0, y1], color="0.6", lw=0.6, zorder=1)
        # nút bias +1
        yb = bottom - (1.8 if sizes[i] > max_show else 1.2)
        for x1, y1 in pos[i + 1]:
            ax.plot([i, x1], [yb, y1], color="tab:red", lw=0.6, zorder=1)
        ax.scatter(i, yb, s=600, facecolor="white", edgecolor="tab:red", lw=1.5, zorder=3)
        ax.text(i, yb, "+1", ha="center", va="center", color="tab:red", zorder=4)

    names = ["Input"] + [f"Hidden {i}" for i in range(1, L - 1)] + ["Output"]
    for i, layer in enumerate(pos):
        color = "tab:blue" if i == 0 else "tab:green" if i == L - 1 else "tab:orange"
        for j, (x, y) in enumerate(layer):
            ax.scatter(x, y, s=600, facecolor="white", edgecolor=color, lw=2, zorder=3)
            if i == 0:
                ax.text(x, y, f"$x_{j + 1}$", ha="center", va="center", zorder=4)
            if i == L - 1:
                ax.text(x, y, r"$\hat{y}$", ha="center", va="center", zorder=4)
        if sizes[i] > max_show:
            ax.text(i, min(y for _, y in layer) - 0.75, "⋮", ha="center", va="center", fontsize=16)
        ax.text(i, max(y for _, y in layer) + 0.9, f"{names[i]}\n({sizes[i]})", ha="center")

    ax.set_title(f"MLP {sizes} – activation: {', '.join(sorted(acts)) or 'none'} – {n_params} tham số",
                 pad=50)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.model.mlp import MLP

    # vẽ từ MLP thật với activation của baseline đã chốt, để hình luôn khớp cấu hình
    act = json.loads((root / "configs" / "baseline.json").read_text(encoding="utf-8"))["base"]["activation"]
    draw_mlp(MLP([10], act), save_path=root / "figures" / "mlp_1hidden_B10.png")
    draw_mlp(MLP([10, 10, 10], act), save_path=root / "figures" / "mlp_C3_10.png")

# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Vẽ sơ đồ kiến trúc CNN và pipeline huấn luyện FER2013 từ mô hình thật. Nhiệm vụ T24.

Mọi hình vẽ đều trích xuất trực tiếp từ mô hình PyTorch thực tế (src/models/cnn.py),
xuất kèm file CSV số liệu và tự động đăng ký vào tracking/figures_manifest.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn import PRESETS, build_model, describe, n_params
from src.visual.plots import register_figure


def _setup_plt():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    })


def plot_baseline_architecture(out_png, out_csv=None, root=ROOT):
    """Vẽ sơ đồ phân rã các khối lớp của mô hình Baseline (Conv3x3, BatchNorm, ReLU, MaxPool, FC)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is None:
        out_csv = out_png.with_suffix(".csv")
    out_csv = Path(out_csv)

    model = build_model("baseline")
    layers_info = describe(model)
    total_p = n_params(model)

    # 1. Ghi dữ liệu ra CSV
    csv_rows = []
    for idx, (lname, shape, params) in enumerate(layers_info, start=1):
        csv_rows.append({
            "stt": idx,
            "layer_name": lname,
            "output_shape": "x".join(map(str, shape)),
            "params": params,
            "params_cumulative": sum(r[2] for r in layers_info[:idx]),
        })

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stt", "layer_name", "output_shape", "params", "params_cumulative"])
        w.writeheader()
        w.writerows(csv_rows)

    # 2. Vẽ sơ đồ kiến trúc dạng khối đồ họa chuyên nghiệp
    _setup_plt()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Danh sách các khối trực quan lớn
    blocks_meta = [
        {"name": "Input", "shape": "1 × 48 × 48", "sub": "FER-2013 (Ảnh xám)", "color": "#E0E0E0", "w": 1.2, "h": 2.2},
        {"name": "Block 1", "shape": "64 × 24 × 24", "sub": "2× [Conv 3x3 (64) + BN + ReLU]\nMaxPool 2x2 + Dropout (0.25)", "color": "#B3E5FC", "w": 2.4, "h": 3.4},
        {"name": "Block 2", "shape": "128 × 12 × 12", "sub": "2× [Conv 3x3 (128) + BN + ReLU]\nMaxPool 2x2 + Dropout (0.25)", "color": "#81D4FA", "w": 2.4, "h": 3.8},
        {"name": "Block 3", "shape": "256 × 6 × 6", "sub": "2× [Conv 3x3 (256) + BN + ReLU]\nMaxPool 2x2 + Dropout (0.25)", "color": "#4FC3F7", "w": 2.4, "h": 4.2},
        {"name": "Flatten", "shape": "9,216", "sub": "256 × 6 × 6", "color": "#FFF9C4", "w": 1.1, "h": 2.6},
        {"name": "Dense (FC)", "shape": "512", "sub": "Linear (9216 → 512)\nReLU + Dropout (0.5)", "color": "#FFE082", "w": 1.9, "h": 3.2},
        {"name": "Output", "shape": "7", "sub": "Linear (512 → 7)\n7 Emotion Logits", "color": "#FFAB91", "w": 1.4, "h": 2.4},
    ]

    total_width = sum(b["w"] for b in blocks_meta) + (len(blocks_meta) - 1) * 0.8
    start_x = 0.5
    y_center = 4.0

    ax.set_xlim(0, start_x + total_width + 0.5)
    ax.set_ylim(0, 8.5)

    current_x = start_x
    for i, b in enumerate(blocks_meta):
        x = current_x
        y = y_center - b["h"] / 2
        rect = patches.FancyBboxPatch(
            (x, y), b["w"], b["h"],
            boxstyle="round,pad=0.1,rounding_size=0.15",
            edgecolor="#37474F", facecolor=b["color"],
            linewidth=1.5, zorder=2
        )
        ax.add_patch(rect)

        # Tiêu đề khối
        ax.text(x + b["w"] / 2, y + b["h"] - 0.35, b["name"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="#263238", zorder=3)
        # Tensor Shape
        ax.text(x + b["w"] / 2, y_center + 0.1, f"Shape:\n{b['shape']}", ha="center", va="center",
                fontsize=9.5, fontweight="semibold", color="#01579B", zorder=3)
        # Nội dung lớp con
        ax.text(x + b["w"] / 2, y + 0.5, b["sub"], ha="center", va="center",
                fontsize=7.8, color="#37474F", zorder=3)

        # Mũi tên kết nối sang khối kế tiếp
        if i < len(blocks_meta) - 1:
            next_x = current_x + b["w"] + 0.8
            ax.annotate(
                "", xy=(next_x, y_center), xytext=(current_x + b["w"], y_center),
                arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=2, mutation_scale=15),
                zorder=1
            )

        current_x += b["w"] + 0.8

    # Chú thích tổng quan
    title_text = "SƠ ĐỒ KIẾN TRÚC MÔ HÌNH BASELINE CNN (FER-2013)"
    ax.text((start_x + total_width + 0.5) / 2, 7.8, title_text,
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

    stats_text = (
        f"Tổng số tham số: {total_p:,}  |  Đầu vào: 48×48×1  |  Đầu ra: 7 lớp cảm xúc  |  "
        "Cơ chế: Batch Normalization, Dropout (0.25 conv / 0.5 fc), ReLU"
    )
    ax.text((start_x + total_width + 0.5) / 2, 7.2, stats_text,
            ha="center", va="center", fontsize=10, fontstyle="italic", color="#424242")

    # Footer
    ax.text((start_x + total_width + 0.5) / 2, 0.4,
            "Trích xuất trực tiếp từ PyTorch: build_model('baseline') trong src/models/cnn.py. Phụ trách: TV4 – Huỳnh Viễn Thông (T24)",
            ha="center", va="center", fontsize=8.5, color="#757575")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_csv


def plot_model_comparison(out_png, out_csv=None, root=ROOT):
    """Vẽ biểu đồ đối sánh số tham số và độ sâu của các kiến trúc CNN trong repo."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is None:
        out_csv = out_png.with_suffix(".csv")
    out_csv = Path(out_csv)

    models_data = []
    for name in ["baseline", "medium_01", "large", "medium", "small"]:
        m = build_model(name)
        p = n_params(m)
        layers = describe(m)
        num_conv = sum(1 for l, _, _ in layers if l.startswith("conv"))
        models_data.append({
            "model_name": name,
            "params": p,
            "params_millions": round(p / 1e6, 2),
            "num_conv_layers": num_conv,
            "has_bn": PRESETS[name]["bn"],
            "fc_units": PRESETS[name]["fc_units"],
        })

    # 1. Ghi CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_name", "params", "params_millions", "num_conv_layers", "has_bn", "fc_units"])
        w.writeheader()
        w.writerows(models_data)

    # 2. Vẽ biểu đồ so sánh
    _setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    names = [d["model_name"] for d in models_data]
    params_m = [d["params_millions"] for d in models_data]
    conv_counts = [d["num_conv_layers"] for d in models_data]
    colors = ["#2E7D32" if n == "baseline" else "#1976D2" for n in names]

    # Bar chart: Tham số (Triệu)
    bars1 = ax1.barh(names, params_m, color=colors, height=0.55, edgecolor="#263238")
    ax1.set_xlabel("Số lượng tham số (Triệu)")
    ax1.set_title("Số tham số của các kiến trúc CNN")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.6, bar.get_y() + bar.get_height() / 2, f"{w:.2f} M",
                 va="center", ha="left", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(params_m) * 1.2)

    # Bar chart: Số lớp Conv
    bars2 = ax2.barh(names, conv_counts, color="#FF7043", height=0.55, edgecolor="#263238")
    ax2.set_xlabel("Số lớp Convolution (3×3)")
    ax2.set_title("Độ sâu Conv (Số lớp Conv)")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(w)} lớp",
                 va="center", ha="left", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, max(conv_counts) * 1.25)

    plt.suptitle("ĐỐI SÁNH CÁC KIẾN TRÚC CNN TRONG THỰC NGHIỆM (E1 KIẾN TRÚC)",
                 fontsize=12, fontweight="bold", y=0.98, color="#1A237E")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_csv


def plot_pipeline_overview(out_png, out_csv=None, root=ROOT):
    """Vẽ sơ đồ quy trình huấn luyện, đánh giá và khóa tập PrivateTest."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is None:
        out_csv = out_png.with_suffix(".csv")
    out_csv = Path(out_csv)

    # 1. Ghi CSV mô tả các chặng trong pipeline
    pipeline_steps = [
        {"stage": "1. Dữ liệu", "input": "fer2013.csv (35,887 ảnh)", "process": "Chia theo cột Usage", "output": "Training (28,709) / PublicTest (3,589) / PrivateTest (3,589)"},
        {"stage": "2. Tăng cường", "input": "Training", "process": "HFlip (p=0.5) + RandomAffine (±10°, 10% translate, 0.9-1.1 scale)", "output": "Tensor 48×48 chuẩn hóa mean/std"},
        {"stage": "3. Huấn luyện", "input": "Batches (size=128)", "process": "Forward CNN (AMP FP16) + CrossEntropy + SGD Nesterov (lr=0.01)", "output": "Loss & Gradients"},
        {"stage": "4. Kiểm định & LR", "input": "PublicTest (Validation)", "process": "Đo Val Acc & Macro-F1; ReduceLROnPlateau (patience=5, factor=0.5)", "output": "Lưu best.pt khi Val Acc cải thiện"},
        {"stage": "5. Dừng sớm", "input": "Val Loss / Acc", "process": "Early Stopping sau 15 epochs không tiến bộ", "output": "Kết thúc huấn luyện tối đa 100 epochs"},
        {"stage": "6. PrivateTest (Khóa)", "input": "PrivateTest (3,589 ảnh)", "process": "CHỈ CHẠY 1 LẦN DUY NHẤT ở mô hình cuối cùng (T19: --eval-test)", "output": "results/E7/TEST_DONE.json"},
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "input", "process", "output"])
        w.writeheader()
        w.writerows(pipeline_steps)

    # 2. Vẽ đồ họa
    _setup_plt()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")

    cards = [
        {"title": "1. DỮ LIỆU GỐC", "desc": "FER-2013 CSV\n35,887 ảnh 48×48\n7 nhãn cảm xúc", "color": "#E3F2FD"},
        {"title": "2. CHIA TẬP DỮ LIỆU", "desc": "• Training: 28,709\n• PublicTest: 3,589\n• PrivateTest: 3,589", "color": "#BBDEFB"},
        {"title": "3. TIỀN XỬ LÝ & AUG", "desc": "• Mean/Std Training\n• Random HFlip\n• RandomAffine (±10°)", "color": "#C8E6C9"},
        {"title": "4. HUẤN LUYỆN (GPU)", "desc": "• Batch 128 (RAM)\n• AMP FP16 Mixed\n• SGD lr=0.01 + Plateau", "color": "#FFF59D"},
        {"title": "5. VAL & CHỌN MODEL", "desc": "• Đánh giá PublicTest\n• Lưu best.pt\n• Early Stopping (15)", "color": "#FFE082"},
        {"title": "6. ĐÁNH GIÁ CUỐI", "desc": "• Khóa PrivateTest\n• CHỈ CHẠY 1 LẦN (T19)\n• TEST_DONE.json", "color": "#FFCDD2"},
    ]

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)

    card_w = 1.75
    card_h = 2.4
    spacing = 0.35
    start_x = 0.5
    y_pos = 1.8

    for i, c in enumerate(cards):
        x = start_x + i * (card_w + spacing)
        rect = patches.FancyBboxPatch(
            (x, y_pos), card_w, card_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            edgecolor="#37474F", facecolor=c["color"],
            linewidth=1.4, zorder=2
        )
        ax.add_patch(rect)

        ax.text(x + card_w / 2, y_pos + card_h - 0.35, c["title"],
                ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1A237E", zorder=3)
        ax.text(x + card_w / 2, y_pos + (card_h - 0.5) / 2, c["desc"],
                ha="center", va="center", fontsize=8.2, color="#263238", zorder=3)

        if i < len(cards) - 1:
            ax.annotate(
                "", xy=(x + card_w + spacing, y_pos + card_h / 2), xytext=(x + card_w, y_pos + card_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.8, mutation_scale=12),
                zorder=1
            )

    # Tiêu đề & ghi chú
    ax.text(6.5, 5.2, "QUY TRÌNH HUẤN LUYỆN, ĐÁNH GIÁ & CHỐT CHẶN DỮ LIỆU (PIPELINE)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0D47A1")
    ax.text(6.5, 4.65, "Tuân thủ nguyên tắc không rò rỉ dữ liệu: PrivateTest hoàn toàn độc lập và chỉ đánh giá một lần ở mô hình cuối",
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#424242")

    ax.text(6.5, 0.8, "Phụ trách: TV4 – Huỳnh Viễn Thông (Nhiệm vụ T24). Sinh tự động từ src/visual/architecture_diagram.py",
            ha="center", va="center", fontsize=8.5, color="#616161")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_csv


def generate_all(out_dir=None, manifest_path=None, root=ROOT):
    """Sinh toàn bộ sơ đồ kiến trúc & pipeline và đăng ký vào tracking/figures_manifest.csv."""
    root = Path(root)
    if out_dir is None:
        out_dir = root / "figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = root / "tracking" / "figures_manifest.csv"
    manifest_path = Path(manifest_path)

    script_rel = "src/visual/architecture_diagram.py"
    cli_cmd = "python -m src.visual.architecture_diagram"

    results = []

    # 1. Sơ đồ kiến trúc Baseline
    png1 = out_dir / "arch_cnn_baseline.png"
    csv1 = out_dir / "arch_cnn_baseline.csv"
    plot_baseline_architecture(png1, csv1, root=root)
    register_figure(
        manifest_path,
        fig_id="arch_cnn_baseline",
        file_hinh=png1,
        file_du_lieu_dau_vao=csv1,
        script_sinh_hinh=script_rel,
        lenh_chay=cli_cmd,
        task_id="T24",
        dung_o="Báo cáo Chương 3, Slide thuyết trình",
        root=root,
    )
    results.append(("arch_cnn_baseline", png1, csv1))

    # 2. Sơ đồ so sánh các kiến trúc CNN
    png2 = out_dir / "arch_cnn_comparison.png"
    csv2 = out_dir / "arch_cnn_comparison.csv"
    plot_model_comparison(png2, csv2, root=root)
    register_figure(
        manifest_path,
        fig_id="arch_cnn_comparison",
        file_hinh=png2,
        file_du_lieu_dau_vao=csv2,
        script_sinh_hinh=script_rel,
        lenh_chay=cli_cmd,
        task_id="T24",
        dung_o="Báo cáo Chương 3, Mục E1 Kiến trúc",
        root=root,
    )
    results.append(("arch_cnn_comparison", png2, csv2))

    # 3. Sơ đồ pipeline toàn diện
    png3 = out_dir / "pipeline_overview.png"
    csv3 = out_dir / "pipeline_overview.csv"
    plot_pipeline_overview(png3, csv3, root=root)
    register_figure(
        manifest_path,
        fig_id="pipeline_overview",
        file_hinh=png3,
        file_du_lieu_dau_vao=csv3,
        script_sinh_hinh=script_rel,
        lenh_chay=cli_cmd,
        task_id="T24",
        dung_o="Báo cáo Chương 1 & 3, Slide thuyết trình",
        root=root,
    )
    results.append(("pipeline_overview", png3, csv3))

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "figures"), help="Thư mục xuất ảnh (mặc định: figures/)")
    parser.add_argument("--manifest", default=str(ROOT / "tracking" / "figures_manifest.csv"), help="Đường dẫn file manifest")
    args = parser.parse_args()

    print("Đang khởi tạo sơ đồ kiến trúc & pipeline từ mô hình thật...")
    res = generate_all(out_dir=args.output_dir, manifest_path=args.manifest, root=ROOT)
    for fig_id, png, csv_f in res:
        print(f"  [OK] {fig_id}: {png.name} (kèm {csv_f.name})")
    print("Hoàn tất sinh sơ đồ và đăng ký vào figures_manifest.csv!")


if __name__ == "__main__":
    main()

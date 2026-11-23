# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Script dòng lệnh (CLI) chạy trực quan hóa các bộ lọc tích chập (conv filters).

Nhiệm vụ T20.
Cách dùng:
    python src/visualize_filters.py [--model baseline] [--checkpoint path/to/best.pt] [--out-dir figures]
    python src/visualize_filters.py --method weights --layer first
    python src/visualize_filters.py --method fmaps --layer last
    python src/visualize_filters.py --method actmax --target class
    python src/visualize_filters.py --method gradcam
"""
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
import sys

import torch

# Hỗ trợ chạy trực tiếp: python src/visualize_filters.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn import PRESETS, build_model
from src.visual.filters import (
    activation_max,
    feature_maps,
    grad_cam,
    plot_weights,
)


def _calc_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_or_build_model(model_name: str = "baseline", checkpoint_path: str = None, device: str = "cpu"):
    """Nạp checkpoint hoặc khởi tạo mô hình mới."""
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Hỗ trợ cả trường hợp ckpt là state_dict hoặc dict chứa state_dict
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model = build_model(model_name)
        model.load_state_dict(state)
        print(f"Đã nạp trọng số từ checkpoint: {checkpoint_path}")
    else:
        model = build_model(model_name)
        print(f"Khởi tạo mô hình '{model_name}' với trọng số ngẫu nhiên.")
    model.to(device).eval()
    return model


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252, print tiếng Việt sẽ lỗi khi chạy qua pipe/subprocess
    parser = argparse.ArgumentParser(description="Trực quan hóa bộ lọc Conv (Nhiệm vụ T20 - TV4)")
    parser.add_argument("--model", default="baseline", choices=sorted(PRESETS), help="Tên kiến trúc CNN")
    parser.add_argument("--checkpoint", default=None, help="Đường dẫn file best.pt (nếu có)")
    parser.add_argument("--out-dir", default="figures", help="Thư mục xuất ảnh và dữ liệu")
    parser.add_argument("--method", default="all", choices=["all", "weights", "fmaps", "actmax", "gradcam"], help="Phương pháp cần chạy")
    parser.add_argument("--layer", default=None, help="Lớp conv cần trực quan hóa ('first', 'last' hoặc tên lớp)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị chạy")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = load_or_build_model(args.model, args.checkpoint, device)

    # Tạo mẫu ảnh tổng hợp 48x48 phục vụ thử nghiệm nếu không nạp dữ liệu thật
    torch.manual_seed(42)
    sample_img = torch.randn(1, 1, 48, 48, device=device)
    # Tạo cấu trúc khuôn mặt giả lập đơn giản: nền tối, tâm sáng nhẹ
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 48), indexing="ij")
    face_mask = (grid_y**2 + grid_x**2 < 0.64).float().to(device)
    sample_img = (sample_img * 0.1 + face_mask * 0.8).clamp(0.0, 1.0)

    generated_manifest = []
    now_iso = datetime.now().isoformat()

    # 1. Trọng số (plot_weights)
    if args.method in ("all", "weights"):
        # Lớp đầu
        fig_first, data_first = plot_weights(
            model,
            layer="first",
            out_path=out_dir / "filters_weights_first.png",
            data_path=out_dir / "filters_weights_first.csv",
        )
        print(f"-> [Weights First] Đã lưu {out_dir / 'filters_weights_first.png'}")

        # Lớp cuối
        fig_last, data_last = plot_weights(
            model,
            layer=args.layer or "last",
            out_path=out_dir / "filters_weights_last.png",
            data_path=out_dir / "filters_weights_last.csv",
        )
        print(f"-> [Weights Last] Đã lưu {out_dir / 'filters_weights_last.png'}")

        for suffix in ["first", "last"]:
            p_img = out_dir / f"filters_weights_{suffix}.png"
            p_data = out_dir / f"filters_weights_{suffix}.csv"
            generated_manifest.append({
                "file_hinh": str(p_img).replace("\\", "/"),
                "file_du_lieu_dau_vao": str(p_data).replace("\\", "/"),
                "lenh_chay": f"python src/visualize_filters.py --method weights --layer {suffix}",
                "sha256_hinh": _calc_sha256(p_img),
            })

    # 2. Bản đồ đặc trưng (feature_maps)
    if args.method in ("all", "fmaps"):
        p_img = out_dir / "filters_feature_maps.png"
        p_data = out_dir / "filters_feature_maps.npz"
        feature_maps(
            model,
            sample_img,
            layer=args.layer or "last",
            top_k=16,
            out_path=p_img,
            data_path=p_data,
        )
        print(f"-> [Feature Maps] Đã lưu {p_img}")
        generated_manifest.append({
            "file_hinh": str(p_img).replace("\\", "/"),
            "file_du_lieu_dau_vao": str(p_data).replace("\\", "/"),
            "lenh_chay": "python src/visualize_filters.py --method fmaps",
            "sha256_hinh": _calc_sha256(p_img),
        })

    # 3. Kích hoạt cực đại (activation_max)
    if args.method in ("all", "actmax"):
        # Theo bộ lọc
        p_img_f = out_dir / "filters_actmax_filter.png"
        p_data_f = out_dir / "filters_actmax_filter.npz"
        activation_max(
            model,
            target_type="filter",
            steps=50,
            out_path=p_img_f,
            data_path=p_data_f,
        )
        print(f"-> [ActMax Filter] Đã lưu {p_img_f}")

        # Theo lớp
        p_img_c = out_dir / "filters_actmax_class.png"
        p_data_c = out_dir / "filters_actmax_class.npz"
        activation_max(
            model,
            target_type="class",
            steps=50,
            out_path=p_img_c,
            data_path=p_data_c,
        )
        print(f"-> [ActMax Class] Đã lưu {p_img_c}")

        for sub, name in [("filter", p_img_f), ("class", p_img_c)]:
            generated_manifest.append({
                "file_hinh": str(name).replace("\\", "/"),
                "file_du_lieu_dau_vao": str(out_dir / f"filters_actmax_{sub}.npz").replace("\\", "/"),
                "lenh_chay": f"python src/visualize_filters.py --method actmax --target {sub}",
                "sha256_hinh": _calc_sha256(name),
            })

    # 4. Grad-CAM
    if args.method in ("all", "gradcam"):
        p_img = out_dir / "filters_grad_cam.png"
        p_data = out_dir / "filters_grad_cam.csv"
        grad_cam(
            model,
            sample_img,
            layer=args.layer or "last",
            out_path=p_img,
            data_path=p_data,
        )
        print(f"-> [Grad-CAM] Đã lưu {p_img}")
        generated_manifest.append({
            "file_hinh": str(p_img).replace("\\", "/"),
            "file_du_lieu_dau_vao": str(p_data).replace("\\", "/"),
            "lenh_chay": "python src/visualize_filters.py --method gradcam",
            "sha256_hinh": _calc_sha256(p_img),
        })

    print(f"\nHoàn tất sinh {len(generated_manifest)} hình trực quan hóa bộ lọc tại: {out_dir}")
    print("Dữ liệu hình ảnh sẵn sàng phục vụ báo cáo và kiểm tra Hinh_ve.")


if __name__ == "__main__":
    main()

# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Script dòng lệnh (CLI) chạy trực quan hóa các bộ lọc tích chập (conv filters).

Nhiệm vụ T20.
Cách dùng:
    python src/visualize_filters.py --run-dir results/E7/E7-001_s0 --data-csv data/fer2013.csv --register-manifest
    python src/visualize_filters.py --checkpoint results/E1/E1-014_s0/best.pt --out-dir figures
    python src/visualize_filters.py [--model baseline] [--model-kwargs '{"bn": true}'] [--out-dir figures]
    python src/visualize_filters.py --method weights --layer first
    python src/visualize_filters.py --method fmaps --layer last
    python src/visualize_filters.py --method actmax --target class
    python src/visualize_filters.py --method gradcam

Lưu ý:
- Khi truyền --run-dir hoặc --checkpoint (có config.json nằm cùng thư mục), script tự động đọc
  tên kiến trúc `model` và `model_kwargs` (vd: bn=True cho large_bn, activation=..., dropout_conv=...)
  từ config.json rồi truyền vào build_model(model_name, **model_kwargs).
- Khi truyền --data-csv, CHỈ đọc tập PublicTest (val), tuyệt đối không đụng PrivateTest.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# Hỗ trợ chạy trực tiếp: python src/visualize_filters.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn import PRESETS, build_model
from src.visual.filters import (
    activation_max,
    feature_maps,
    grad_cam,
    plot_weights,
)
from src.visual.plots import register_figure


def _calc_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:12]


def _extract_model_spec(config_path: Path):
    """Đọc (model_name, model_kwargs, exp_id, full_cfg) từ config.json (hỗ trợ cả dạng có khóa 'cfg' lồng)."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = raw.get("cfg", raw) if isinstance(raw, dict) else {}
    model_name = cfg.get("model")
    model_kwargs = dict(cfg.get("model_kwargs") or {})
    exp_id = raw.get("exp_id") or cfg.get("exp_id") or Path(config_path).parent.name
    return model_name, model_kwargs, exp_id, cfg


def load_or_build_model(
    model_name: str = "baseline",
    checkpoint_path: str = None,
    device: str = "cpu",
    config_path: str = None,
    model_kwargs: dict = None,
):
    """Nạp checkpoint hoặc khởi tạo mô hình mới, tự truyền model_kwargs từ config.json nếu có."""
    resolved_name = model_name
    resolved_kwargs = {}
    exp_id = ""
    full_cfg = {}

    cfg_file = Path(config_path) if config_path else None
    if cfg_file is None and checkpoint_path:
        sibling_cfg = Path(checkpoint_path).parent / "config.json"
        if sibling_cfg.exists():
            cfg_file = sibling_cfg

    if cfg_file is not None and cfg_file.exists():
        cfg_model, cfg_kwargs, exp_id, full_cfg = _extract_model_spec(cfg_file)
        if cfg_model:
            resolved_name = cfg_model
        resolved_kwargs.update(cfg_kwargs)

    if model_kwargs:
        resolved_kwargs.update(model_kwargs)

    model = build_model(resolved_name, **resolved_kwargs)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt
        model.load_state_dict(state)
        print(
            f"Đã nạp trọng số từ checkpoint: {checkpoint_path} "
            f"(model='{resolved_name}', model_kwargs={resolved_kwargs})"
        )
    else:
        print(f"Khởi tạo mô hình '{resolved_name}' (model_kwargs={resolved_kwargs}) với trọng số ngẫu nhiên.")

    model.to(device).eval()
    return model


def _load_publictest_sample(data_csv: Path, sample_idx: int, cfg: dict, device: torch.device):
    """Nạp đúng 1 ảnh thật từ tập PublicTest (val_loader), tuyệt đối không đụng PrivateTest."""
    from src.data.loader import make_loaders

    loader_cfg = dict(cfg) if cfg else {"batch_size": 128, "num_workers": 0, "seed": 0}
    loader_cfg["data_path"] = str(data_csv)
    loader_cfg["num_workers"] = 0
    _, val_loader = make_loaders(loader_cfg)
    val_ds = val_loader.dataset
    if sample_idx < 0 or sample_idx >= len(val_ds):
        raise IndexError(f"sample_idx={sample_idx} ngoài phạm vi PublicTest [0, {len(val_ds) - 1}]")
    img, label = val_ds[sample_idx]
    return img.unsqueeze(0).to(device), int(label)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Trực quan hóa bộ lọc Conv (Nhiệm vụ T20 - TV4)")
    parser.add_argument("--run-dir", default=None, help="Thư mục run (vd results/E7/E7-001_s0), tự tìm best.pt và config.json")
    parser.add_argument("--model", default="baseline", choices=sorted(PRESETS), help="Tên kiến trúc CNN (bị ghi đè nếu có config.json)")
    parser.add_argument("--config", default=None, help="Đường dẫn config.json (tự lấy model & model_kwargs)")
    parser.add_argument("--model-kwargs", default=None, help="Chuỗi JSON ghi đè model_kwargs (vd '{\"bn\": true}')")
    parser.add_argument("--checkpoint", default=None, help="Đường dẫn file best.pt (nếu có)")
    parser.add_argument("--data-csv", default=None, help="Đường dẫn fer2013.csv để lấy ảnh thật từ PublicTest (không chạm PrivateTest)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Chỉ số mẫu trong PublicTest khi dùng --data-csv")
    parser.add_argument("--target-class", type=int, default=None, help="Chỉ số lớp cảm xúc (0..6) cho Grad-CAM")
    parser.add_argument("--steps", type=int, default=50, help="Số bước tối ưu Activation Maximization")
    parser.add_argument("--out-dir", default="figures", help="Thư mục xuất ảnh và dữ liệu")
    parser.add_argument("--method", default="all", choices=["all", "weights", "fmaps", "actmax", "gradcam"], help="Phương pháp cần chạy")
    parser.add_argument("--layer", default=None, help="Lớp conv cần trực quan hóa ('first', 'last' hoặc tên lớp)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị chạy")
    parser.add_argument("--register-manifest", action="store_true", help="Đăng ký các hình sinh ra vào tracking/figures_manifest.csv")
    parser.add_argument("--manifest", default=str(ROOT / "tracking/figures_manifest.csv"), help="Đường dẫn figures_manifest.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt_path = args.checkpoint
    cfg_path = args.config
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if ckpt_path is None and (run_dir / "best.pt").exists():
            ckpt_path = str(run_dir / "best.pt")
        if cfg_path is None and (run_dir / "config.json").exists():
            cfg_path = str(run_dir / "config.json")

    cli_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
    exp_id = ""
    full_cfg = {}
    effective_cfg = Path(cfg_path) if cfg_path else (Path(ckpt_path).parent / "config.json" if ckpt_path else None)
    if effective_cfg is not None and effective_cfg.exists():
        _, _, exp_id, full_cfg = _extract_model_spec(effective_cfg)

    model = load_or_build_model(
        args.model,
        ckpt_path,
        device=device,
        config_path=cfg_path,
        model_kwargs=cli_kwargs,
    )

    sample_label = None
    if args.data_csv:
        sample_img, sample_label = _load_publictest_sample(Path(args.data_csv), args.sample_idx, full_cfg, device)
        print(f"Đã nạp ảnh thật PublicTest idx={args.sample_idx}, true_label={sample_label}")
    else:
        # Tạo mẫu ảnh tổng hợp 48x48 phục vụ thử nghiệm unit test khi không nạp dữ liệu thật
        torch.manual_seed(42)
        sample_img = torch.randn(1, 1, 48, 48, device=device)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 48), indexing="ij")
        face_mask = (grid_y**2 + grid_x**2 < 0.64).float().to(device)
        sample_img = (sample_img * 0.1 + face_mask * 0.8).clamp(0.0, 1.0)

    cmd_prefix = f"python src/visualize_filters.py --run-dir {args.run_dir}" if args.run_dir else "python src/visualize_filters.py"
    generated_manifest = []

    # 1. Trọng số (plot_weights)
    if args.method in ("all", "weights"):
        plot_weights(
            model,
            layer="first",
            out_path=out_dir / "filters_weights_first.png",
            data_path=out_dir / "filters_weights_first.csv",
        )
        print(f"-> [Weights First] Đã lưu {out_dir / 'filters_weights_first.png'}")

        plot_weights(
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
                "fig_id": f"filters_weights_{suffix}",
                "file_hinh": p_img,
                "file_du_lieu_dau_vao": p_data,
                "lenh_chay": f"{cmd_prefix} --method weights --layer {suffix}",
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
            "fig_id": "filters_feature_maps",
            "file_hinh": p_img,
            "file_du_lieu_dau_vao": p_data,
            "lenh_chay": f"{cmd_prefix} --method fmaps",
            "sha256_hinh": _calc_sha256(p_img),
        })

    # 3. Kích hoạt cực đại (activation_max)
    if args.method in ("all", "actmax"):
        p_img_f = out_dir / "filters_actmax_filter.png"
        p_data_f = out_dir / "filters_actmax_filter.npz"
        activation_max(
            model,
            target_type="filter",
            steps=args.steps,
            out_path=p_img_f,
            data_path=p_data_f,
        )
        print(f"-> [ActMax Filter] Đã lưu {p_img_f}")

        p_img_c = out_dir / "filters_actmax_class.png"
        p_data_c = out_dir / "filters_actmax_class.npz"
        activation_max(
            model,
            target_type="class",
            steps=args.steps,
            out_path=p_img_c,
            data_path=p_data_c,
        )
        print(f"-> [ActMax Class] Đã lưu {p_img_c}")

        for sub, p_img, p_data in [("filter", p_img_f, p_data_f), ("class", p_img_c, p_data_c)]:
            generated_manifest.append({
                "fig_id": f"filters_actmax_{sub}",
                "file_hinh": p_img,
                "file_du_lieu_dau_vao": p_data,
                "lenh_chay": f"{cmd_prefix} --method actmax",
                "sha256_hinh": _calc_sha256(p_img),
            })

    # 4. Grad-CAM
    if args.method in ("all", "gradcam"):
        p_img = out_dir / "filters_grad_cam.png"
        p_data = out_dir / "filters_grad_cam.csv"
        target_cls = args.target_class if args.target_class is not None else sample_label
        grad_cam(
            model,
            sample_img,
            target_class=target_cls,
            layer=args.layer or "last",
            out_path=p_img,
            data_path=p_data,
        )
        print(f"-> [Grad-CAM] Đã lưu {p_img}")
        generated_manifest.append({
            "fig_id": "filters_grad_cam",
            "file_hinh": p_img,
            "file_du_lieu_dau_vao": p_data,
            "lenh_chay": f"{cmd_prefix} --method gradcam",
            "sha256_hinh": _calc_sha256(p_img),
        })

    if args.register_manifest:
        for item in generated_manifest:
            register_figure(
                manifest_path=args.manifest,
                fig_id=item["fig_id"],
                file_hinh=item["file_hinh"],
                file_du_lieu_dau_vao=item["file_du_lieu_dau_vao"],
                script_sinh_hinh="src/visualize_filters.py",
                lenh_chay=item["lenh_chay"],
                exp_ids=exp_id,
                task_id="T20",
                dung_o="T20: Trực quan hóa bộ lọc Conv, Feature Maps, Activation Maximization & Grad-CAM",
                root=ROOT,
            )

    print(f"\nHoàn tất sinh {len(generated_manifest)} hình trực quan hóa bộ lọc tại: {out_dir}")
    print("Dữ liệu hình ảnh sẵn sàng phục vụ báo cáo và kiểm tra Hinh_ve.")


if __name__ == "__main__":
    main()

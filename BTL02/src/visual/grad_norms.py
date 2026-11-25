"""Hình chuẩn gradient từng lớp cho E2 (activation, vanishing gradient).

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T13.

Đọc grad_norms.csv do trainer ghi khi cfg có log_grad_norms=true (mỗi epoch, mỗi lớp Conv/Linear một chuẩn L2 trung
bình của gradient). Vẽ 2 hình con:
  (a) chuẩn gradient theo vị trí lớp (từ lớp đầu vào tới lớp ra) ở một epoch, trục y log;
  (b) chuẩn gradient của lớp Conv đầu tiên theo epoch, trục y log.
Gradient càng nhỏ dần về phía lớp đầu (đường (a) dốc lên về bên phải) là dấu hiệu vanishing gradient.
Kèm CSV dữ liệu vẽ và một dòng figures_manifest.csv.

Chạy từ BTL02: python src/visual/grad_norms.py [--group E2] [--epoch 1] [--exp-ids E2-001_s0 ...]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BTL02 = Path(__file__).resolve().parents[2]
if str(BTL02) not in sys.path:
    sys.path.insert(0, str(BTL02))

from src.visual.plots import _apply_style, register_figure  # noqa: E402


def load_runs(root, group, exp_ids=None):
    """[(exp_id, variant, [dòng grad_norms.csv])] của các run Completed trong nhóm có grad_norms.csv, theo thứ tự csv."""
    with open(root / "tracking" / "experiments.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    runs = []
    for r in rows:
        if exp_ids and r["exp_id"] not in exp_ids:
            continue
        if not exp_ids and (not r["exp_id"].startswith(f"{group}-") or r["trang_thai_ky_thuat"] != "Completed"):
            continue
        path = root / r["run_dir"] / "grad_norms.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            g = list(csv.DictReader(f))
        runs.append((r["exp_id"], json.loads(r["hyperparams_json"]).get("variant", r["exp_id"]), g))
    return runs


def plot_grad_norms(runs, out_png, out_csv, epoch=1):
    _apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8))
    data = []
    for exp_id, variant, g in runs:
        layers = list(dict.fromkeys(r["layer"] for r in g))  # thứ tự lớp như trong mô hình
        at = {r["layer"]: float(r["grad_norm_mean"]) for r in g if int(r["epoch"]) == epoch}
        first = [(int(r["epoch"]), float(r["grad_norm_mean"])) for r in g if r["layer"] == layers[0]]
        label = f"{variant} ({exp_id})"
        ax_a.plot(range(len(layers)), [at.get(l, float("nan")) for l in layers], marker="o", markersize=3, label=label)
        ax_b.plot([e for e, _ in first], [v for _, v in first], label=label)
        data += [dict(exp_id=exp_id, variant=variant, epoch=r["epoch"], layer_idx=layers.index(r["layer"]),
                      layer=r["layer"], grad_norm_mean=r["grad_norm_mean"]) for r in g]
    if runs:
        layers = list(dict.fromkeys(r["layer"] for r in runs[0][2]))
        ax_a.set_xticks(range(len(layers)))
        fc = [l for l in layers if not l.split(".")[-1].startswith("conv")]  # Linear: classifier.1, classifier.4 ...
        ax_a.set_xticklabels([f"fc{fc.index(l) + 1}" if l in fc else l.split(".")[-1] for l in layers],
                             rotation=60, ha="right", fontsize=8)
    ax_a.set(yscale="log", xlabel="Lớp (đầu vào → đầu ra)", ylabel="Chuẩn L2 gradient (TB các batch)",
             title=f"(a) Chuẩn gradient từng lớp, epoch {epoch}")
    ax_b.set(yscale="log", xlabel="Epoch", ylabel="Chuẩn L2 gradient", title="(b) Lớp Conv đầu tiên theo epoch")
    ax_a.legend(fontsize=7)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp_id", "variant", "epoch", "layer_idx", "layer", "grad_norm_mean"])
        w.writeheader()
        w.writerows(data)
    return out_png


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Vẽ chuẩn gradient từng lớp từ grad_norms.csv (E2)")
    ap.add_argument("--group", default="E2")
    ap.add_argument("--exp-ids", nargs="*", help="chỉ vẽ các run này (mặc định: mọi run Completed của nhóm)")
    ap.add_argument("--epoch", type=int, default=1, help="epoch dùng cho hình (a)")
    ap.add_argument("--out", help="file PNG (mặc định figures/<group>_grad_norms.png)")
    ap.add_argument("--root", default=str(BTL02))
    a = ap.parse_args(argv)
    root = Path(a.root)
    runs = load_runs(root, a.group, a.exp_ids)
    if not runs:
        raise SystemExit(f"không có run nào của {a.group} có grad_norms.csv")
    out_png = Path(a.out) if a.out else root / "figures" / f"{a.group.lower()}_grad_norms.png"
    out_csv = out_png.with_suffix(".csv")
    plot_grad_norms(runs, out_png, out_csv, a.epoch)
    cmd = "python src/visual/grad_norms.py " + " ".join(argv if argv is not None else sys.argv[1:])
    register_figure(root / "tracking" / "figures_manifest.csv", out_png.stem, out_png, out_csv,
                    "src/visual/grad_norms.py", cmd.strip(), exp_ids=";".join(r[0] for r in runs),
                    task_id="T13", root=root)
    print(f"đã vẽ {len(runs)} run -> {out_png}")


if __name__ == "__main__":
    main()

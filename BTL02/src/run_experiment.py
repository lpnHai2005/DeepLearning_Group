"""Chạy một nhóm thực nghiệm từ file config JSON. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T07.

Dùng (từ thư mục BTL02):
    python src/run_experiment.py --config configs/E3_optim.json --member TV3 [--only adamw_1e-3 ...]

Config JSON:
    {"group": "E3", "task_id": "T14",
     "base": {"model": "baseline", "model_kwargs": {}, "epochs": 100, "optimizer": "sgd", "lr": 0.01, ...},
     "variants": {"adamw_1e-3": {"optimizer": "adamw", "lr": 0.001}},
     "seeds": [0]}
`base` + `variants[tên]` + seed là cfg của một run; cfg được đưa nguyên cho make_loaders(cfg) (TV1) và fit (khóa
của fit xem src/engine/trainer.py). Mô hình: "model" là tên cấu hình CNN (kèm "model_kwargs" ghi đè); transfer
learning dùng "transfer": {"name": "resnet18", "pretrained": true, "freeze_backbone": false}.

Chạy chia seed trên nhiều máy (ví dụ baseline T11: mỗi người một seed): thêm --seeds 1. Run mới dùng lại số NNN của
run cùng group, cùng cfg (trừ seed) đã Completed mà chưa có seed đó, nên 3 seed chạy ở 3 máy vẫn là NNN_s0/_s1/_s2
và make_tables gom được mean ± std. Nhớ pull experiments.csv mới nhất trước khi chạy.

Mỗi (variant, seed) là một run: results/<group>/<group>-NNN_s<seed>/ gồm config.json (ai chạy, máy nào, giờ nào,
commit, sha256 dữ liệu), run.log, history.csv, metrics.json, best.pt (chỉ run hợp lệ), curves.png + curves.csv (đường
cong Train/Val loss, acc vẽ từ history.csv bằng plots.py của T08, kèm 1 dòng tracking/figures_manifest.csv; run chưa
xong epoch nào thì không có). Sau mỗi run, code tự nối 1 dòng vào tracking/experiments.csv (kể cả run lỗi/bị ngắt). Chạy lại sau lỗi luôn lấy exp_id mới, không ghi đè.
Chưa có --eval-test: PrivateTest chạy đúng 1 lần ở nhiệm vụ T19 (TV3).
"""
import argparse
import csv
import getpass
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # chạy được cả dạng `python src/run_experiment.py`

import torch  # noqa: E402

from src.engine.trainer import fit, record_failure, scan_orphans  # noqa: E402
from src.models import build_model, build_transfer, n_params  # noqa: E402
from src.utils.io import append_csv_row, rel, sha256_file, write_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.visual.plots import plot_curves, register_figure  # noqa: E402  (T08, TV3)


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(root, *args):
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def _environment(root):
    data = root / "data" / "fer2013.csv"
    commit = _git(root, "rev-parse", "--short=8", "HEAD")
    if commit and _git(root, "status", "--porcelain", "--", "src", "configs"):
        commit += "-dirty"
    return dict(
        hostname=platform.node(), os_user=getpass.getuser(), git_user=_git(root, "config", "user.name"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
        cuda=torch.version.cuda, git_commit=commit, dataset_sha256=sha256_file(data) if data.exists() else "")


def _build(cfg):
    if "transfer" in cfg:
        return build_transfer(**cfg["transfer"])
    return build_model(cfg.get("model", "baseline"), **cfg.get("model_kwargs", {}))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_row(run_dir, root=ROOT):
    """Dựng 1 dòng experiments.csv từ chính file trong run_dir (config.json, metrics.json): nguồn duy nhất."""
    run_dir = Path(run_dir)
    meta, m = _read_json(run_dir / "config.json"), _read_json(run_dir / "metrics.json")
    kept = m["weights_quyet_dinh"] == "KEEP"
    return dict(
        exp_id=meta["exp_id"], task_id=meta["task_id"], thanh_vien=meta["member"], bat_dau=meta["started_at"],
        ket_thuc=meta["finished_at"] or "", tong_thoi_gian_s=meta["duration_s"] if meta["duration_s"] is not None else "",
        epoch_time_s_tb=m["epoch_time_s_tb"], may_chay=f"{meta['hostname']} | {meta['gpu']}",
        config_path=meta["config_path"], hyperparams_json=json.dumps(meta["cfg"], ensure_ascii=False, sort_keys=True),
        dataset_sha256=meta["dataset_sha256"], seed=meta["cfg"]["seed"], git_commit=meta["git_commit"],
        n_params=meta["n_params"], train_loss_cuoi=m["train_loss_cuoi"], train_acc_cuoi=m["train_acc_cuoi"],
        val_loss_cuoi=m["val_loss_cuoi"], best_val_loss=m["best_val_loss"], best_epoch=m["best_epoch"],
        val_acc=m["best_val_acc"], val_macro_f1=m["best_val_macro_f1"], trang_thai_ky_thuat=m["status"],
        weights_path=rel(run_dir / "best.pt", root) if kept else "", weights_sha256=m["weights_sha256"],
        weights_size_mb=m["weights_size_mb"], weights_quyet_dinh=m["weights_quyet_dinh"],
        weights_ma_ly_do=m["weights_ma_ly_do"], run_dir=rel(run_dir, root), log_path=rel(run_dir / "run.log", root),
        history_path=rel(run_dir / "history.csv", root),
        curve_path=rel(run_dir / "curves.png", root) if (run_dir / "curves.png").exists() else "")


def plot_run(run_dir, root=ROOT):
    """Vẽ curves.png + curves.csv từ history.csv của run và ghi/cập nhật 1 dòng figures_manifest.csv.

    Run chưa xong epoch nào thì bỏ qua (trả None). Lỗi khi vẽ chỉ in cảnh báo, không làm hỏng run đã train.
    """
    run_dir, root = Path(run_dir), Path(root)
    hist = run_dir / "history.csv"
    if not hist.exists() or len(hist.read_text(encoding="utf-8").splitlines()) < 2:
        return None
    meta = _read_json(run_dir / "config.json")
    try:
        png, _ = plot_curves(hist, run_dir / "curves.png", title=meta["exp_id"])
        register_figure(root / "tracking" / "figures_manifest.csv", f"curves_{meta['exp_id']}", png, hist,
                        "src/visual/plots.py", f"python src/visual/plots.py --run-dir {rel(run_dir, root)}",
                        exp_ids=meta["exp_id"], task_id=meta["task_id"], root=root)
    except Exception as e:
        print(f"[{meta['exp_id']}] không vẽ được curves.png: {e!r}", file=sys.stderr)
        return None
    return png


def _next_number(root, group, known_ids):
    pat = re.compile(rf"^{re.escape(group)}-(\d+)")
    ids = list(known_ids) + [p.name for p in (root / "results" / group).glob("*")]
    return 1 + max([int(m.group(1)) for m in map(pat.match, ids) if m] or [0])


def _cfg_key(cfg):
    """Khóa so sánh cấu hình: mọi khóa trừ seed và exp_id."""
    return json.dumps({k: v for k, v in cfg.items() if k not in ("seed", "exp_id")}, ensure_ascii=False, sort_keys=True)


def _pick_number(group, key, seed, prior, taken):
    """Số NNN để dùng lại cho (cfg, seed), hoặc None nếu phải lấy số mới.

    prior: {exp_id: (cfg_key, trang_thai)} các run đã có. Chỉ dùng lại số mà mọi run của nó cùng cfg và đều
    Completed (run lỗi thì chạy lại sang số mới), và chưa có seed này (không ghi đè run cũ).
    """
    pat = re.compile(rf"^{re.escape(group)}-(\d+)_s(\d+)$")
    by_num = {}
    for exp_id, (k, status) in prior.items():
        m = pat.match(exp_id)
        if m:
            by_num.setdefault(int(m.group(1)), []).append((int(m.group(2)), k, status))
    for num in sorted(by_num):
        runs = by_num[num]
        same_cfg_ok = all(k == key and st == "Completed" for _, k, st in runs)
        if same_cfg_ok and seed not in {sd for sd, _, _ in runs} and f"{group}-{num:03d}_s{seed}" not in taken:
            return num
    return None


def run(config_path, member, only=None, root=ROOT, seeds=None):
    """Chạy mọi (variant, seed) của file config; trả danh sách metrics của từng run.

    seeds: ghi đè danh sách seed trong config (chạy chia seed trên nhiều máy).
    """
    root = Path(root)
    spec = _read_json(config_path)
    with open(root / "tracking" / "members.csv", encoding="utf-8-sig", newline="") as f:
        members = {r["ma"] for r in csv.DictReader(f)}
    if member not in members:
        raise SystemExit(f"--member phải là một trong {sorted(members)} (mã trong tracking/members.csv), nhận '{member}'")
    unknown = set(only or []) - set(spec["variants"])
    if unknown:
        raise SystemExit(f"--only không có trong config: {sorted(unknown)}; có: {sorted(spec['variants'])}")
    csv_path = root / "tracking" / "experiments.csv"
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header, rows_csv = reader.fieldnames, list(reader)
    known = [r["exp_id"] for r in rows_csv]
    prior = {r["exp_id"]: (_cfg_key(json.loads(r["hyperparams_json"] or "{}")), r["trang_thai_ky_thuat"]) for r in rows_csv}
    from src.data.loader import make_loaders  # nhiệm vụ T04 (TV1)

    for d in scan_orphans(root / "results"):  # run bị kill từ lần trước: ghi dòng còn thiếu
        if d.name not in known:
            plot_run(d, root)
            row = collect_row(d, root)
            append_csv_row(csv_path, header, row)
            known.append(d.name)
            prior[d.name] = (_cfg_key(json.loads(row["hyperparams_json"])), row["trang_thai_ky_thuat"])
    env, group, results = _environment(root), spec["group"], []
    n = _next_number(root, group, known)
    taken = set(known) | {p.name for p in (root / "results" / group).glob("*")}
    for variant, over in spec["variants"].items():
        if only and variant not in only:
            continue
        fresh = None  # số mới dùng chung cho các seed của variant này trong lần chạy
        for seed in (seeds if seeds is not None else spec.get("seeds", [0])):
            cfg = {**spec["base"], **over, "seed": seed, "variant": variant}
            num = _pick_number(group, _cfg_key(cfg), seed, prior, taken)
            if num is None:
                if fresh is None or f"{group}-{fresh:03d}_s{seed}" in taken:
                    fresh, n = n, n + 1
                num = fresh
            exp_id = f"{group}-{num:03d}_s{seed}"
            taken.add(exp_id)
            run_dir = root / "results" / group / exp_id
            run_dir.mkdir(parents=True)  # đã tồn tại thì lỗi: không ghi đè
            cfg["exp_id"] = exp_id
            meta = dict(exp_id=exp_id, task_id=spec.get("task_id", ""), group=group, variant=variant, member=member,
                        **env, started_at=_now(), finished_at=None, duration_s=None, n_params=None,
                        config_path=rel(config_path, root), cfg=cfg)
            write_json(run_dir / "config.json", meta)
            t0 = time.perf_counter()
            try:
                set_seed(seed)
                train_loader, val_loader = make_loaders(cfg)
                model = _build(cfg)
                meta["n_params"] = n_params(model)
                write_json(run_dir / "config.json", meta)
                res = fit(model, train_loader, val_loader, cfg, run_dir)
            except Exception as e:  # lỗi trước khi vào fit: vẫn ghi run Failed
                record_failure(run_dir, exp_id, e)
                res = _read_json(run_dir / "metrics.json")
            meta.update(finished_at=_now(), duration_s=round(time.perf_counter() - t0, 3))
            write_json(run_dir / "config.json", meta)
            plot_run(run_dir, root)
            append_csv_row(csv_path, header, collect_row(run_dir, root))
            results.append(res)
            if res["status"] == "Aborted":  # Ctrl+C: dừng cả nhóm, không chạy tiếp
                return results
    return results


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # --help và log tiếng Việt không lỗi trên console Windows (cp1252)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="file JSON, ví dụ configs/E3_optim.json")
    ap.add_argument("--member", required=True, help="mã người chạy (TV1..TV4), ghi vào config.json và experiments.csv")
    ap.add_argument("--only", nargs="*", help="chỉ chạy các variant có tên này")
    ap.add_argument("--seeds", nargs="+", type=int, help="chỉ chạy các seed này (ghi đè 'seeds' trong config)")
    a = ap.parse_args(argv)
    run(a.config, a.member, a.only, seeds=a.seeds)


if __name__ == "__main__":
    main()

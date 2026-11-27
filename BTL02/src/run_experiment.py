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
Đánh giá PrivateTest đúng 1 lần duy nhất ở nhiệm vụ T19 (phần --eval-test phụ trách: TV3 – Võ Hoàng Bảo):
    python src/run_experiment.py --eval-test --member TV3 --run-dirs results/E7/E7-001_s0 results/E7/E7-001_s1 results/E7/E7-001_s2 --tta flip
Lệnh tự tạo results/E7/TEST_DONE.json và từ chối mọi lần chạy thứ hai nếu file khóa đã tồn tại.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.engine.evaluate import EMOTIONS, metrics, predict, save_predictions  # noqa: E402  (T08, TV3)
from src.engine.trainer import fit, record_failure, scan_orphans  # noqa: E402
from src.models import build_model, build_transfer, n_params  # noqa: E402
from src.utils.io import append_csv_row, rel, sha256_file, write_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.visual.plots import (  # noqa: E402  (T08, TV3)
    plot_confusion_matrix,
    plot_curves,
    plot_error_histogram,
    plot_sample_grid,
    register_figure,
)


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


def _load_member_set(root):
    with open(Path(root) / "tracking" / "members.csv", encoding="utf-8-sig", newline="") as f:
        return {r["ma"] for r in csv.DictReader(f)}


def _load_checkpoint_model(run_dir, device):
    """Nạp kiến trúc và trọng số best.pt từ một run_dir hợp lệ."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"
    if not cfg_path.is_file():
        raise SystemExit(f"Không tìm thấy config.json trong run_dir: {run_dir}")
    if not ckpt_path.is_file():
        raise SystemExit(f"Không tìm thấy best.pt trong run_dir: {run_dir}")
    meta = _read_json(cfg_path)
    cfg = meta.get("cfg", meta)
    model = _build(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, meta, cfg, ckpt_path


def _collect_loader_images(loader):
    """Gom ảnh đầu vào từ loader theo đúng thứ tự batch để vẽ lưới ca đúng/sai."""
    xs = []
    for x, _ in loader:
        xs.append(x.detach().cpu().numpy())
    if not xs:
        raise ValueError("test_loader không có mẫu nào")
    return np.concatenate(xs, axis=0)


def eval_test(
    run_dirs,
    member,
    tta="none",
    config_path=None,
    lock_file=None,
    out_dir=None,
    root=ROOT,
    device=None,
    benchmark_public=0.69,
    benchmark_private=0.71,
    register_figures=True,
    cli_command=None,
):
    """Đánh giá PrivateTest ĐÚNG 1 LẦN DUY NHẤT cho mô hình cuối E7 (nhiệm vụ T19, TV3).

    - Kiểm tra file khóa TEST_DONE.json ngay đầu hàm: nếu đã tồn tại thì từ chối chạy ngay.
    - Hỗ trợ 1 mô hình hoặc ensemble nhiều seed (trung bình xác suất softmax) kèm TTA (none/flip/ten_crop).
    - Ghi TEST_DONE.json nguyên tử, test_predictions.npz, test_per_class.csv, test_run.log,
      các biểu đồ kèm CSV, đăng ký figures_manifest.csv và nối 1 dòng vào tracking/experiments.csv.
    """
    root = Path(root)
    members = _load_member_set(root)
    if member not in members:
        raise SystemExit(f"--member phải là một trong {sorted(members)} (mã trong tracking/members.csv), nhận '{member}'")

    lock_path = Path(lock_file) if lock_file else (root / "results" / "E7" / "TEST_DONE.json")
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    if lock_path.exists():
        raise SystemExit(
            f"TỪ CHỐI CHẠY --eval-test: PrivateTest đã được đánh giá và khóa tại '{rel(lock_path, root)}'. "
            "Quy tắc BTL02 chỉ cho phép đánh giá PrivateTest đúng 1 lần duy nhất!"
        )

    if not run_dirs:
        raise SystemExit("Cần truyền ít nhất 1 thư mục run qua --run-dirs khi dùng --eval-test.")

    resolved_dirs = []
    for d in run_dirs:
        p = Path(d)
        resolved_dirs.append(p if p.is_absolute() else (root / p))

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir) if out_dir else lock_path.parent
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _now()
    t0 = time.perf_counter()

    loaded_models = []
    input_sig = None
    for rd in resolved_dirs:
        model, meta, cfg, ckpt_path = _load_checkpoint_model(rd, dev)
        sig = (int(cfg.get("resize", 48)), int(cfg.get("channels", 1)))
        if input_sig is None:
            input_sig = sig
        elif sig != input_sig:
            raise SystemExit(f"Các checkpoint không đồng nhất kích thước/kênh đầu vào: {input_sig} vs {sig} ({rd})")
        loaded_models.append((model, meta, cfg, ckpt_path, rd))

    loader_cfg = dict(loaded_models[0][2])
    if config_path:
        cfg_p = Path(config_path)
        if not cfg_p.is_absolute():
            cfg_p = root / cfg_p
        spec = _read_json(cfg_p)
        if "base" in spec:
            loader_cfg = {**spec["base"], **loader_cfg}
    loader_cfg["device"] = str(dev)

    from src.data.loader import make_test_loader  # nhiệm vụ T04 (TV1) — chỉ gọi trong --eval-test

    test_loader = make_test_loader(loader_cfg)
    images = _collect_loader_images(test_loader)

    use_amp = bool(loader_cfg.get("amp", True)) and dev.type == "cuda"
    y_true = None
    probs_list = []
    model_records = []
    for model, meta, cfg, ckpt_path, rd in loaded_models:
        yt, yp = predict(model, test_loader, tta=tta, device=dev, amp=use_amp)
        if y_true is None:
            y_true = yt
        elif not np.array_equal(y_true, yt):
            raise RuntimeError("Thứ tự nhãn PrivateTest giữa các lần duyệt loader không khớp!")
        probs_list.append(yp)
        single_m = metrics(yt, yp)
        model_records.append(
            {
                "exp_id": meta.get("exp_id", rd.name),
                "run_dir": rel(rd, root),
                "weights_path": rel(ckpt_path, root),
                "weights_sha256": sha256_file(ckpt_path),
                "n_params": int(meta.get("n_params") or n_params(model)),
                "seed": cfg.get("seed"),
                "test_acc": single_m["acc"],
                "test_macro_f1": single_m["macro_f1"],
            }
        )

    y_prob = np.mean(probs_list, axis=0).astype(np.float32)
    y_pred = y_prob.argmax(axis=1).astype(np.int64)
    final_m = metrics(y_true, y_prob)

    pred_path = save_predictions(out_path / "test_predictions.npz", y_true, y_prob)

    per_class_path = out_path / "test_per_class.csv"
    with open(per_class_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "precision", "recall", "f1", "support"], lineterminator="\n")
        w.writeheader()
        for label in EMOTIONS:
            st = final_m["per_class"][label]
            w.writerow(
                {
                    "label": label,
                    "precision": round(st["precision"], 6),
                    "recall": round(st["recall"], 6),
                    "f1": round(st["f1"], 6),
                    "support": st["support"],
                }
            )

    exp_ids = [r["exp_id"] for r in model_records]
    exp_ids_joined = ";".join(exp_ids)
    cmd_str = cli_command or (
        f"python src/run_experiment.py --eval-test --member {member} "
        f"--run-dirs {' '.join(r['run_dir'] for r in model_records)} --tta {tta}"
    )

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm_png, cm_csv = plot_confusion_matrix(
        final_m["cm"],
        fig_dir / "e7_test_confusion.png",
        title=f"Ma trận nhầm lẫn PrivateTest — E7 ({len(model_records)} seed, TTA={tta}, acc={final_m['acc']*100:.2f}%)",
    )
    err_png, err_csv = plot_error_histogram(
        y_true,
        y_pred,
        fig_dir / "e7_test_error_hist.png",
        title=f"Recall và tỷ lệ sai theo lớp trên PrivateTest — E7 (TTA={tta})",
    )
    ok_png, ok_csv = plot_sample_grid(
        images,
        y_true,
        y_pred,
        fig_dir / "e7_test_correct_grid.png",
        y_prob=y_prob,
        correct=True,
        max_samples=16,
        title="PrivateTest — Mẫu dự đoán ĐÚNG điển hình (E7)",
    )
    bad_png, bad_csv = plot_sample_grid(
        images,
        y_true,
        y_pred,
        fig_dir / "e7_test_wrong_grid.png",
        y_prob=y_prob,
        correct=False,
        max_samples=16,
        title="PrivateTest — Mẫu dự đoán SAI điển hình (E7)",
    )

    if register_figures:
        manifest_path = root / "tracking" / "figures_manifest.csv"
        for fid, png_p, csv_p, usage in [
            ("e7_test_confusion", cm_png, cm_csv, "Báo cáo Ch.4 & Slide E7"),
            ("e7_test_error_hist", err_png, err_csv, "Báo cáo Ch.4 phân tích lỗi"),
            ("e7_test_correct_grid", ok_png, ok_csv, "Báo cáo Ch.4 ca đúng"),
            ("e7_test_wrong_grid", bad_png, bad_csv, "Báo cáo Ch.4 ca sai"),
        ]:
            register_figure(
                manifest_path,
                fid,
                png_p,
                csv_p,
                "src/run_experiment.py",
                cmd_str,
                exp_ids=exp_ids_joined,
                task_id="T19",
                dung_o=usage,
                root=root,
            )

    finished_at = _now()
    duration_s = round(time.perf_counter() - t0, 3)
    env = _environment(root)
    log_path = out_path / "test_run.log"
    log_lines = [
        f"START | {started_at} | task=T19 | member={member} | host={env['hostname']} | gpu={env['gpu']}",
        f"CMD   | {cmd_str}",
        f"ENV   | commit={env['git_commit']} | dataset_sha256={env['dataset_sha256']} | tta={tta} | n_models={len(model_records)}",
    ]
    for r in model_records:
        log_lines.append(
            f"MODEL | {r['exp_id']} | sha256={r['weights_sha256'][:12]} | acc={r['test_acc']:.6f} | macro_f1={r['test_macro_f1']:.6f}"
        )
    log_lines.extend(
        [
            f"FINAL | n_samples={len(y_true)} | test_acc={final_m['acc']:.6f} | test_macro_f1={final_m['macro_f1']:.6f}",
            f"BENCH | vs_public_69={final_m['acc'] - benchmark_public:+.6f} | vs_private_71={final_m['acc'] - benchmark_private:+.6f}",
            f"END   | {finished_at} | duration_s={duration_s} | lock={rel(lock_path, root)}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    payload = {
        "locked": True,
        "task_id": "T19",
        "split": "PrivateTest",
        "evaluated_at": finished_at,
        "started_at": started_at,
        "duration_s": duration_s,
        "member": member,
        **env,
        "command": cmd_str,
        "tta": str(tta),
        "n_models": len(model_records),
        "n_samples": int(len(y_true)),
        "test_acc": final_m["acc"],
        "test_macro_f1": final_m["macro_f1"],
        "benchmark_public": benchmark_public,
        "benchmark_private": benchmark_private,
        "diff_vs_public_69": round(final_m["acc"] - benchmark_public, 6),
        "diff_vs_private_71": round(final_m["acc"] - benchmark_private, 6),
        "per_class": final_m["per_class"],
        "models": model_records,
        "predictions_path": rel(pred_path, root),
        "per_class_csv": rel(per_class_path, root),
        "log_path": rel(log_path, root),
        "figures": {
            "confusion": rel(cm_png, root),
            "error_hist": rel(err_png, root),
            "correct_grid": rel(ok_png, root),
            "wrong_grid": rel(bad_png, root),
        },
    }
    tmp_lock = lock_path.with_suffix(".json.tmp")
    write_json(tmp_lock, payload)
    tmp_lock.replace(lock_path)

    exp_csv = root / "tracking" / "experiments.csv"
    if exp_csv.exists():
        with open(exp_csv, encoding="utf-8", newline="") as f:
            header = csv.DictReader(f).fieldnames
        if header:
            test_row = {
                "exp_id": "E7-TEST",
                "task_id": "T19",
                "thanh_vien": member,
                "bat_dau": started_at,
                "ket_thuc": finished_at,
                "tong_thoi_gian_s": duration_s,
                "epoch_time_s_tb": "",
                "may_chay": f"{env['hostname']} | {env['gpu']}",
                "config_path": rel(config_path, root) if config_path and Path(config_path).exists() else (config_path or ""),
                "hyperparams_json": json.dumps(
                    {
                        "variant": f"private_test_{tta}",
                        "tta": str(tta),
                        "n_models": len(model_records),
                        "exp_ids": exp_ids,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "dataset_sha256": env["dataset_sha256"],
                "seed": ",".join(str(r["seed"]) for r in model_records if r["seed"] is not None),
                "git_commit": env["git_commit"],
                "n_params": model_records[0]["n_params"],
                "test_acc": final_m["acc"],
                "trang_thai_ky_thuat": "Completed",
                "weights_path": ";".join(r["weights_path"] for r in model_records),
                "weights_sha256": ";".join(r["weights_sha256"] for r in model_records),
                "weights_quyet_dinh": "KEEP",
                "weights_ma_ly_do": "OK",
                "run_dir": rel(out_path, root),
                "log_path": rel(log_path, root),
            }
            append_csv_row(exp_csv, header, test_row)

    print(
        f"[T19 --eval-test] PrivateTest acc={final_m['acc']:.4f} ({final_m['acc']*100:.2f}%), "
        f"macro-F1={final_m['macro_f1']:.4f} | Đã khóa tại {rel(lock_path, root)}"
    )
    return payload


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # --help và log tiếng Việt không lỗi trên console Windows (cp1252)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="file JSON, ví dụ configs/E3_optim.json (bắt buộc khi huấn luyện)")
    ap.add_argument("--member", required=True, help="mã người chạy (TV1..TV4), ghi vào config.json và experiments.csv")
    ap.add_argument("--only", nargs="*", help="chỉ chạy các variant có tên này")
    ap.add_argument("--seeds", nargs="+", type=int, help="chỉ chạy các seed này (ghi đè 'seeds' trong config)")
    ap.add_argument("--eval-test", action="store_true", help="đánh giá PrivateTest ĐÚNG 1 LẦN DUY NHẤT (nhiệm vụ T19)")
    ap.add_argument("--run-dirs", nargs="+", help="danh sách thư mục run chứa best.pt để đánh giá --eval-test")
    ap.add_argument("--tta", default="none", choices=["none", "flip", "ten_crop"], help="chế độ TTA cho --eval-test")
    ap.add_argument("--lock-file", default=None, help="đường dẫn file khóa (mặc định results/E7/TEST_DONE.json)")
    a = ap.parse_args(argv)
    if a.eval_test:
        cmd_str = "python src/run_experiment.py " + " ".join(argv if argv is not None else sys.argv[1:])
        eval_test(
            run_dirs=a.run_dirs,
            member=a.member,
            tta=a.tta,
            config_path=a.config,
            lock_file=a.lock_file,
            cli_command=cmd_str,
        )
        return
    if not a.config:
        ap.error("--config là bắt buộc khi không truyền --eval-test")
    run(a.config, a.member, a.only, seeds=a.seeds)


if __name__ == "__main__":
    main()

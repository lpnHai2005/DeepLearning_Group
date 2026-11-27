# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Trực quan hóa các bộ lọc tích chập (conv filters) và bản đồ đặc trưng cho FER2013.

Nhiệm vụ T20 (Mục 2.4 & Mục 4 trong PHAN_CONG_NHIEM_VU.md).
Hiện thực 4 phương pháp trực quan hóa:
1. plot_weights: Vẽ lưới trọng số bộ lọc conv đầu (3x3x1) và conv cuối (trung bình kênh vào).
2. feature_maps: Trích xuất và vẽ bản đồ đặc trưng đầu ra của lớp conv cuối cho ảnh.
3. activation_max: Tối ưu ảnh đầu vào (gradient ascent + L2 regularization) để kích hoạt cực đại
   từng bộ lọc conv cuối (filter activation) hoặc cực đại logit từng lớp cảm xúc (class activation).
4. grad_cam: Tính bản đồ nhiệt Grad-CAM tại lớp conv cuối cho ảnh đầu vào.

Mọi hàm đều hoạt động được trên cả mô hình chưa huấn luyện (ngẫu nhiên) lẫn mô hình đã huấn luyện (best.pt),
và hỗ trợ xuất hình ảnh (.png) kèm dữ liệu số (.csv hoặc .npz) phục vụ truy vết số liệu.
"""
import csv
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Chế độ headless, không phụ thuộc GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _model_device(model: nn.Module) -> torch.device:
    """Lấy thiết bị hiện tại của mô hình (CPU hoặc CUDA)."""
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _get_target_layer(model: nn.Module, layer: Optional[Union[str, nn.Module]] = None) -> Tuple[nn.Module, str]:
    """Tìm module lớp conv mục tiêu và tên của nó."""
    if layer is None or layer == "last":
        if hasattr(model, "last_conv"):
            name = getattr(model, "last_conv_name", "last_conv")
            return model.last_conv(), name
        # Tìm lớp Conv2d cuối cùng trong model
        last_conv = None
        last_name = ""
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
                last_name = name
        if last_conv is None:
            raise ValueError("Không tìm thấy lớp Conv2d nào trong mô hình.")
        return last_conv, last_name

    if layer == "first":
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                return mod, name
        raise ValueError("Không tìm thấy lớp Conv2d nào trong mô hình.")

    if isinstance(layer, nn.Module):
        return layer, layer.__class__.__name__

    if isinstance(layer, str):
        # Tìm theo tên thuộc tính trong model.features hoặc toàn model
        if hasattr(model, "features") and hasattr(model.features, layer):
            return getattr(model.features, layer), layer
        for name, mod in model.named_modules():
            if name == layer:
                return mod, name
        raise ValueError(f"Không tìm thấy layer có tên '{layer}' trong mô hình.")

    raise TypeError(f"layer không hợp lệ: {type(layer)}")


def _save_array_data(data: np.ndarray, data_path: Union[str, Path]):
    """Lưu mảng dữ liệu ra file .npz hoặc .csv."""
    p = Path(data_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".npz":
        np.savez_compressed(p, data=data)
    elif p.suffix == ".npy":
        np.save(p, data)
    else:
        # Mặc định lưu CSV (nếu 2D thì lưu trực tiếp, nếu >2D thì làm phẳng từng hàng)
        flat = data.reshape(data.shape[0], -1) if data.ndim > 2 else data
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"col_{i}" for i in range(flat.shape[1])])
            w.writerows(flat.tolist())


def plot_weights(
    model: nn.Module,
    layer: Optional[Union[str, nn.Module]] = "first",
    max_filters: int = 64,
    out_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """1. Trực quan hóa trọng số bộ lọc (Weights).

    - Nếu layer là conv đầu (C_in=1): vẽ trực tiếp lưới 3x3 của từng bộ lọc.
    - Nếu layer là conv cuối (C_in>1): tính trung bình theo kênh vào (weights.mean(dim=1))
      để tạo biểu diễn 2D 3x3 cho từng bộ lọc ra (C_out).

    Trả về:
        fig: matplotlib Figure.
        weights_repr: mảng numpy 2D/3D biểu diễn các bộ lọc đã vẽ.
    """
    conv_layer, layer_name = _get_target_layer(model, layer)
    w = conv_layer.weight.detach().cpu().float()  # (C_out, C_in, kH, kW)
    n_filters = min(w.shape[0], max_filters)
    w = w[:n_filters]

    if w.shape[1] == 1:
        # Conv đầu: kênh vào = 1 (ảnh xám) -> (N, 3, 3)
        repr_w = w[:, 0].numpy()
        title = f"Trọng số bộ lọc lớp đầu ({layer_name}, {n_filters} filters, 3x3x1)"
    else:
        # Conv sâu/cuối: trung bình theo các kênh vào -> (N, 3, 3)
        repr_w = w.mean(dim=1).numpy()
        title = f"Trọng số bộ lọc lớp {layer_name} ({n_filters} filters, trung bình theo C_in={w.shape[1]})"

    cols = int(np.ceil(np.sqrt(n_filters)))
    rows = int(np.ceil(n_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < n_filters:
            k = repr_w[i]
            # Chuẩn hóa min-max để hiển thị rõ
            k_min, k_max = k.min(), k.max()
            norm_k = (k - k_min) / (k_max - k_min + 1e-8)
            axes[i].imshow(norm_k, cmap="gray", interpolation="nearest")
            axes[i].set_title(f"F{i+1}", fontsize=7)
        axes[i].axis("off")

    fig.suptitle(title, fontsize=11, y=0.98)
    fig.tight_layout()

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    if data_path:
        _save_array_data(repr_w, data_path)

    return fig, repr_w


def feature_maps(
    model: nn.Module,
    x: torch.Tensor,
    layer: Optional[Union[str, nn.Module]] = "last",
    top_k: int = 16,
    out_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """2. Trực quan hóa bản đồ đặc trưng (Feature Maps).

    Trích xuất kích hoạt đầu ra tại lớp conv được chỉ định và vẽ `top_k` kênh
    có cường độ kích hoạt mạnh nhất (theo tổng giá trị kích hoạt).

    Args:
        model: Mô hình PyTorch.
        x: Tensor ảnh đầu vào (1, C, H, W) hoặc (N, C, H, W). Nếu batch > 1 thì lấy ảnh đầu tiên.
        layer: Tên lớp hoặc module (mặc định 'last' - conv cuối).
        top_k: Số kênh kích hoạt mạnh nhất cần hiển thị (mặc định 16).
    """
    model.eval()
    conv_layer, layer_name = _get_target_layer(model, layer)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    device = _model_device(model)
    x_in = x[:1].detach().to(device).clone()

    act_out = []

    def hook_fn(mod, inp, out):
        act_out.append(out.detach().cpu())

    h = conv_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(x_in)
    h.remove()

    if not act_out:
        raise RuntimeError("Hook không bắt được đầu ra của lớp conv.")

    fmap = act_out[0][0]  # (C, H, W)
    num_channels = fmap.shape[0]
    k = min(top_k, num_channels)

    # Chọn top-k kênh có kích hoạt trung bình cao nhất
    energy = fmap.mean(dim=(1, 2))  # (C,)
    top_indices = torch.topk(energy, k=k).indices.numpy()
    selected_fmaps = fmap[top_indices].numpy()  # (k, H, W)

    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < k:
            ch_idx = top_indices[i]
            fm = selected_fmaps[i]
            axes[i].imshow(fm, cmap="viridis", interpolation="nearest")
            axes[i].set_title(f"Ch {ch_idx}\n({energy[ch_idx]:.2f})", fontsize=8)
        axes[i].axis("off")

    fig.suptitle(f"Bản đồ đặc trưng lớp {layer_name} (Top {k}/{num_channels} kênh mạnh nhất)", fontsize=11, y=0.98)
    fig.tight_layout()

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    if data_path:
        _save_array_data(selected_fmaps, data_path)

    return fig, selected_fmaps


def activation_max(
    model: nn.Module,
    target_type: str = "filter",
    target_indices: Optional[list] = None,
    layer: Optional[Union[str, nn.Module]] = "last",
    steps: int = 80,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    input_shape: Tuple[int, int, int] = (1, 48, 48),
    out_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    seed: Optional[int] = 42,
) -> Tuple[plt.Figure, np.ndarray]:
    """3. Cực đại hóa kích hoạt (Activation Maximization).

    Tối ưu ảnh đầu vào bằng Gradient Ascent để:
    - `target_type='filter'`: Tối đa hóa kích hoạt trung bình của từng bộ lọc ở lớp conv.
    - `target_type='class'`: Tối đa hóa logit đầu ra cho từng lớp cảm xúc (0..6).

    Args:
        model: Mô hình PyTorch.
        target_type: 'filter' hoặc 'class'.
        target_indices: Danh sách chỉ số bộ lọc hoặc lớp (mặc định lấy 8 bộ lọc đầu hoặc 7 lớp).
        layer: Lớp conv mục tiêu nếu target_type='filter'.
        steps: Số bước tối ưu gradient ascent.
        lr: Tốc độ học (learning rate).
    """
    model.eval()
    conv_layer, layer_name = _get_target_layer(model, layer)

    if target_type == "class":
        if target_indices is None:
            target_indices = list(range(len(EMOTIONS)))
        titles = [f"{EMOTIONS[c]}" for c in target_indices]
        main_title = "Activation Maximization - Kích hoạt cực đại từng lớp cảm xúc"
    elif target_type == "filter":
        if target_indices is None:
            target_indices = list(range(min(8, conv_layer.out_channels)))
        titles = [f"Filter {idx}" for idx in target_indices]
        main_title = f"Activation Maximization - Kích hoạt bộ lọc lớp {layer_name}"
    else:
        raise ValueError("target_type phải là 'filter' hoặc 'class'.")

    device = _model_device(model)
    optimized_imgs = []

    for target in target_indices:
        # Khởi tạo ảnh ngẫu nhiên lá (leaf tensor) quanh giá trị trung bình 0.5 trên đúng device của model
        if seed is not None:
            torch.manual_seed(seed + int(target))
        x = (torch.randn(1, *input_shape, device=device) * 0.05 + 0.5).clamp(0.0, 1.0).detach().requires_grad_(True)

        act_holder = []

        def hook_fn(mod, inp, out):
            act_holder.append(out)

        h = conv_layer.register_forward_hook(hook_fn) if target_type == "filter" else None

        for _ in range(steps):
            if x.grad is not None:
                x.grad.zero_()
            act_holder.clear()
            logits = model(x)

            if target_type == "filter":
                if not act_holder:
                    break
                act = act_holder[0]  # (1, C, H, W)
                loss = act[0, target].mean()
            else:
                loss = logits[0, target]

            loss.backward()

            # Gradient ascent: cực đại hóa kích hoạt + điều chuẩn L2 + clamp
            with torch.no_grad():
                if x.grad is not None:
                    g = x.grad
                    g = g / (g.norm() + 1e-8)
                    x.add_(lr * g - weight_decay * x)
                    x.clamp_(0.0, 1.0)

        if h is not None:
            h.remove()

        img_np = x.detach().cpu().squeeze().numpy()
        optimized_imgs.append(img_np)

    optimized_imgs_arr = np.array(optimized_imgs)
    n = len(target_indices)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(optimized_imgs_arr[i], cmap="gray", vmin=0.0, vmax=1.0)
            axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")

    fig.suptitle(main_title, fontsize=11, y=0.98)
    fig.tight_layout()

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    if data_path:
        _save_array_data(optimized_imgs_arr, data_path)

    return fig, optimized_imgs_arr


def compute_grad_cam_map(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    layer: Optional[Union[str, nn.Module]] = "last",
) -> Tuple[np.ndarray, int, float, str]:
    """Tính bản đồ nhiệt Grad-CAM chuẩn hóa [0, 1] cùng (pred_class, pred_prob, layer_name)."""
    model.eval()
    conv_layer, layer_name = _get_target_layer(model, layer)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    device = _model_device(model)
    x_in = x[:1].detach().to(device).clone().requires_grad_(True)

    activations = []
    gradients = []

    def fwd_hook(mod, inp, out):
        activations.append(out)

    def bwd_hook(mod, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = conv_layer.register_forward_hook(fwd_hook)
    h2 = conv_layer.register_full_backward_hook(bwd_hook)

    logits = model(x_in)
    probs = torch.softmax(logits.detach(), dim=-1)[0]
    pred_class = int(logits.argmax(dim=-1).item())
    pred_prob = float(probs[pred_class].item())
    c = pred_class if target_class is None else target_class

    score = logits[0, c]
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Không thu được activation hoặc gradient cho Grad-CAM.")

    act = activations[0].detach()[0]  # (C, H, W)
    grad = gradients[0].detach()[0]  # (C, H, W)

    # Trọng số alpha: Global average pooling của gradient
    weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

    # Tổ hợp tuyến tính kích hoạt với ReLU
    cam = F.relu((weights * act).sum(dim=0, keepdim=True)).unsqueeze(0)  # (1, 1, H, W)

    # Phóng to bản đồ nhiệt lên kích thước ảnh gốc (H_in, W_in)
    cam_resized = F.interpolate(cam, size=(x_in.shape[2], x_in.shape[3]), mode="bilinear", align_corners=False)
    cam_np = cam_resized.squeeze().cpu().numpy()

    # Chuẩn hóa về [0, 1]
    cam_min, cam_max = cam_np.min(), cam_np.max()
    cam_norm = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
    return cam_norm, pred_class, pred_prob, layer_name


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    layer: Optional[Union[str, nn.Module]] = "last",
    out_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """4. Grad-CAM (Gradient-weighted Class Activation Mapping).

    Tính bản đồ nhiệt vùng chú ý tại lớp conv cuối cho ảnh đầu vào x đối với
    lớp dự đoán (hoặc `target_class` được chỉ định).

    Args:
        model: Mô hình PyTorch.
        x: Tensor ảnh (1, C, H, W) hoặc (C, H, W).
        target_class: Lớp cần giải thích (0..6). Nếu None, tự động lấy lớp có logit cao nhất.
        layer: Lớp conv để tính CAM (mặc định conv cuối).
    """
    cam_norm, pred_class, _pred_prob, layer_name = compute_grad_cam_map(
        model, x, target_class=target_class, layer=layer
    )
    c = pred_class if target_class is None else target_class

    # Tạo ảnh overlay
    raw_img = (x[:1] if x.ndim == 4 else x).detach().cpu().squeeze().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))

    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title(f"Ảnh gốc ({EMOTIONS[c] if c < len(EMOTIONS) else c})", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(cam_norm, cmap="jet")
    axes[1].set_title(f"Grad-CAM ({layer_name})", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(raw_img, cmap="gray")
    axes[2].imshow(cam_norm, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay (Dự đoán: {EMOTIONS[pred_class]})", fontsize=9)
    axes[2].axis("off")

    fig.tight_layout()

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    if data_path:
        _save_array_data(cam_norm, data_path)

    return fig, cam_norm


def grad_cam_grid(
    model: nn.Module,
    samples: list,
    layer: Optional[Union[str, nn.Module]] = "last",
    out_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, list]:
    """Vẽ lưới Grad-CAM (hàng trên: ảnh gốc, hàng dưới: overlay) cho nhiều ca mẫu PublicTest."""
    if not samples:
        raise ValueError("Danh sách samples cho grad_cam_grid không được rỗng.")

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(max(n * 2.1, 4.5), 5.3), squeeze=False)
    records = []
    cam_maps = []
    layer_name = ""

    for col, item in enumerate(samples):
        x = item["x"]
        sample_idx = item.get("sample_idx", col)
        true_idx = item.get("true_idx")
        true_label = item.get("true_label")
        if true_label is None and true_idx is not None and 0 <= int(true_idx) < len(EMOTIONS):
            true_label = EMOTIONS[int(true_idx)]

        cam_norm, pred_idx, pred_prob, layer_name = compute_grad_cam_map(
            model, x, target_class=item.get("target_class"), layer=layer
        )
        pred_label = EMOTIONS[pred_idx] if pred_idx < len(EMOTIONS) else str(pred_idx)
        raw_img = (x[:1] if x.ndim == 4 else x).detach().cpu().squeeze().numpy()

        peak_flat = int(np.argmax(cam_norm))
        peak_y, peak_x = divmod(peak_flat, cam_norm.shape[1])
        is_correct = true_idx is not None and int(true_idx) == pred_idx

        axes[0, col].imshow(raw_img, cmap="gray")
        top_title = f"#{sample_idx}\nTrue: {true_label}" if true_label is not None else f"#{sample_idx}"
        axes[0, col].set_title(top_title, fontsize=8.5)
        axes[0, col].axis("off")

        axes[1, col].imshow(raw_img, cmap="gray")
        axes[1, col].imshow(cam_norm, cmap="jet", alpha=0.5)
        pred_color = "#1b7a2f" if is_correct else ("#b71c1c" if true_idx is not None else "black")
        axes[1, col].set_title(f"Pred: {pred_label}\n({pred_prob:.1%})", fontsize=8.5, color=pred_color)
        axes[1, col].axis("off")

        cam_maps.append(cam_norm)
        records.append({
            "sample_idx": int(sample_idx),
            "true_idx": "" if true_idx is None else int(true_idx),
            "true_label": "" if true_label is None else str(true_label),
            "pred_idx": int(pred_idx),
            "pred_label": pred_label,
            "confidence": round(pred_prob, 6),
            "cam_mean": round(float(cam_norm.mean()), 6),
            "cam_max": round(float(cam_norm.max()), 6),
            "cam_peak_y": int(peak_y),
            "cam_peak_x": int(peak_x),
            "layer_name": layer_name,
        })

    suptitle = title or f"Grad-CAM tại lớp {layer_name} ({n} mẫu PublicTest)"
    fig.suptitle(suptitle, fontsize=11, y=0.97)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.03, wspace=0.12, hspace=0.40)

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")

    if data_path:
        dp = Path(data_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        if dp.suffix == ".npz":
            np.savez_compressed(dp, cams=np.array(cam_maps))
        else:
            fieldnames = list(records[0].keys())
            with open(dp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(records)

    return fig, records

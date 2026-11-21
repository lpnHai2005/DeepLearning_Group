"""Test cho src/models/transfer.py. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T06.

Chạy từ thư mục BTL02: python -m pytest tests/test_transfer.py
Không tải trọng số ImageNet (pretrained=False) để test chạy offline; tensor ngẫu nhiên chỉ dùng trong test.
"""
import pytest
import torch
from torch import nn

from src.models.cnn import n_params
from src.models.transfer import ARCHS, build_transfer, describe, n_trainable


@pytest.mark.parametrize("name", sorted(ARCHS))
def test_output_is_logits_n_by_7(name):
    m = build_transfer(name).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 7)


def test_head_is_new_linear_with_7_outputs():
    assert build_transfer("resnet18").fc.out_features == 7
    assert build_transfer("vgg16").classifier[-1].out_features == 7


def test_param_counts_by_hand():
    # ResNet18 gốc 11,689,512 tham số (fc 512*1000+1000 = 513,000); thay fc bằng 512*7+7 = 3,591
    assert n_params(build_transfer("resnet18")) == 11_689_512 - 513_000 + 3_591
    # VGG16 gốc 138,357,544 (lớp cuối 4096*1000+1000 = 4,097,000); thay bằng 4096*7+7 = 28,679
    assert n_params(build_transfer("vgg16")) == 138_357_544 - 4_097_000 + 28_679


@pytest.mark.parametrize("name", sorted(ARCHS))
def test_freeze_backbone_trains_only_head(name):
    m = build_transfer(name, freeze_backbone=True)
    head = m.fc if name == "resnet18" else m.classifier[-1]
    assert n_trainable(m) == sum(p.numel() for p in head.parameters())
    assert all(p.requires_grad for p in head.parameters())


def test_not_frozen_trains_everything():
    m = build_transfer("resnet18")
    assert n_trainable(m) == n_params(m)


def test_frozen_backbone_stays_eval_and_bn_stats_do_not_move():
    m = build_transfer("resnet18", freeze_backbone=True)
    m.train()
    before = m.layer1[0].bn1.running_mean.clone()
    m(torch.randn(4, 3, 64, 64))
    assert m.fc.training and not m.layer1[0].bn1.training
    assert torch.equal(before, m.layer1[0].bn1.running_mean)


def test_unfrozen_bn_stats_do_update():
    m = build_transfer("resnet18").train()
    before = m.layer1[0].bn1.running_mean.clone()
    m(torch.randn(4, 3, 64, 64))
    assert not torch.equal(before, m.layer1[0].bn1.running_mean)


def test_deepcopy_and_pickle_keep_freezing_independent():
    import copy, io
    m = build_transfer("resnet18", freeze_backbone=True)
    c = copy.deepcopy(m).train()
    assert c is not m
    c(torch.randn(4, 3, 64, 64))
    assert not c.layer1[0].bn1.training and m.layer1[0].bn1.running_mean.abs().sum() == 0  # bản gốc không bị đụng
    buf = io.BytesIO(); torch.save(m, buf); buf.seek(0)
    assert torch.load(buf, weights_only=False).fc.out_features == 7


def test_frozen_backbone_gets_no_gradient():
    m = build_transfer("resnet18", freeze_backbone=True)
    m(torch.randn(2, 3, 64, 64)).sum().backward()
    assert m.conv1.weight.grad is None and m.fc.weight.grad is not None


def test_vgg16_at_48px_collapses_to_1x1_at_last_pool():
    shapes = {l: s for l, s, _ in describe(build_transfer("vgg16"), (1, 3, 48, 48))}
    pools = [s for k, s in shapes.items() if k.startswith("features")]  # dict giữ thứ tự chạy
    assert [p[1] for p in pools] == [24, 12, 6, 3, 1]  # "fail case" của báo cáo mẫu
    assert shapes["avgpool"] == (512, 7, 7)  # torchvision phóng 1x1 lên 7x7 trước FC


def test_vgg16_at_96px_has_3x3_at_last_pool():
    shapes = describe(build_transfer("vgg16"), (1, 3, 96, 96))
    assert [s for l, s, _ in shapes if l.startswith("features")][-1] == (512, 3, 3)


def test_describe_params_sum_to_total_for_resnet():
    m = build_transfer("resnet18")
    assert sum(p for _, _, p in describe(m)) == n_params(m)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        build_transfer("khong_co")

"""Test cho src/models/cnn.py. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T06.

Chạy từ thư mục BTL02: python -m pytest tests/test_cnn.py
Tensor ngẫu nhiên chỉ dùng trong test này, không bao giờ xuất hiện trong kết quả hay hình của báo cáo.
"""
import pytest
import torch
from torch import nn

from src.models.cnn import ACTIVATIONS, PRESETS, build_model, describe, n_params


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_output_is_logits_n_by_7(name):
    m = build_model(name).eval()
    out = m(torch.randn(5, 1, 48, 48))
    assert out.shape == (5, 7)
    assert isinstance(m.classifier[-1], nn.Linear)  # lớp cuối là Linear: không có softmax trong model


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_n_params_equals_sum_of_numel(name):
    m = build_model(name)
    assert n_params(m) == sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_params(m) == sum(p for _, _, p in describe(m))  # tổng theo từng lớp cũng khớp


def test_n_params_by_hand_tiny_config():
    # 1 conv 1->2 kênh 3x3 có bias: 1*2*9+2=20; GAP; Linear 2->7: 2*7+7=21. Tổng 41, tính tay độc lập với code.
    m = build_model("baseline", blocks=[[2]], bn=False, head="gap", fc_units=0)
    assert n_params(m) == 41


def test_bn_adds_params_and_removes_conv_bias():
    # BN: +2 tham số (gamma, beta) mỗi kênh; conv không còn bias (-1 mỗi kênh): chênh = +1 mỗi kênh
    kw = dict(blocks=[[4]], head="gap", fc_units=0)
    assert n_params(build_model("baseline", bn=True, **kw)) - n_params(build_model("baseline", bn=False, **kw)) == 4


def test_feature_map_sizes_follow_pooling():
    shapes = {l: s for l, s, _ in describe(build_model("large"))}
    assert [shapes[f"pool{b}"][1] for b in range(1, 5)] == [24, 12, 6, 3]
    assert shapes["conv1_1"] == (64, 48, 48)  # padding=1 giữ nguyên kích thước


def test_presets_match_sample_keras_architecture():
    # Số conv mỗi khối và độ rộng FC lấy từ demo_code.html
    convs = lambda n: [len(b) for b in PRESETS[n]["blocks"]]
    assert convs("small") == [2, 4] and convs("medium") == [2, 3, 3] and convs("medium_01") == [3, 4, 4]
    assert convs("large") == [4, 4, 4, 4]
    assert PRESETS["large"]["fc_units"] == 2048 and PRESETS["medium"]["fc_units"] == 1024


def test_too_many_pools_for_48px_raises():
    with pytest.raises(ValueError, match="MaxPool"):
        build_model("baseline", blocks=[[4]] * 7)  # 48 -> 0 sau 7 lần pool: đây là "fail case" kiểu VGG16 ở 48 px


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        build_model("khong_co")
    with pytest.raises(ValueError):
        build_model("baseline", activation="khong_co")
    with pytest.raises(ValueError):
        build_model("baseline", head="khong_co")


@pytest.mark.parametrize("act", sorted(ACTIVATIONS))
def test_every_activation_builds_and_runs(act):
    out = build_model("baseline", activation=act).eval()(torch.randn(2, 1, 48, 48))
    assert out.shape == (2, 7) and torch.isfinite(out).all()


def test_last_conv_is_the_last_conv_layer():
    m = build_model("medium")
    assert m.last_conv_name == "conv3_3"
    assert isinstance(m.last_conv(), nn.Conv2d) and m.last_conv().out_channels == 256


def test_eval_mode_is_deterministic_train_mode_uses_dropout():
    x = torch.randn(4, 1, 48, 48)
    m = build_model("baseline").eval()
    assert torch.equal(m(x), m(x))
    m.train()
    assert not torch.equal(m(x), m(x))


def test_one_sgd_step_lowers_loss():
    torch.manual_seed(0)
    m = build_model("baseline", blocks=[[8], [16]], fc_units=32)
    x, y = torch.randn(16, 1, 48, 48), torch.randint(0, 7, (16,))
    opt, loss_fn = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9), nn.CrossEntropyLoss()
    m.eval()  # tắt dropout để so sánh loss công bằng trước và sau
    before = loss_fn(m(x), y).item()
    for _ in range(5):
        opt.zero_grad()
        loss_fn(m(x), y).backward()
        opt.step()
    assert loss_fn(m(x), y).item() < before


def test_default_init_unchanged_by_init_argument():
    """init="default" phải cho đúng trọng số như trước khi có tham số init (các run cũ không đổi)."""
    torch.manual_seed(0)
    a = build_model("large")
    torch.manual_seed(0)
    b = build_model("large", init="default")
    for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert na == nb and torch.equal(pa, pb)


def test_glorot_init_matches_keras_glorot_uniform():
    """Trọng số Conv/Linear nằm trong [-limit, limit], limit = sqrt(6/(fan_in+fan_out)) như Keras; bias = 0."""
    torch.manual_seed(0)
    m = build_model("large", init="glorot")
    layers = [x for x in m.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
    assert len(layers) == 16 + 2  # 16 Conv + 2 Dense của LARGE_CNN
    for layer in layers:
        w = layer.weight
        rf = w[0][0].numel() if w.dim() > 2 else 1
        fan_in, fan_out = w.shape[1] * rf, w.shape[0] * rf
        limit = (6 / (fan_in + fan_out)) ** 0.5
        assert w.abs().max() <= limit
        assert w.abs().max() > 0.9 * limit  # dùng gần hết khoảng, không phải phân phối hẹp hơn
        assert torch.count_nonzero(layer.bias) == 0


def test_glorot_init_keeps_gradient_in_deep_net_without_bn():
    """Lý do thêm init (E1): ở LARGE không BN, gradient tới conv1_1 với init mặc định nhỏ hơn Glorot nhiều bậc."""
    x, y = torch.randn(16, 1, 48, 48), torch.randint(0, 7, (16,))
    norms = {}
    for init in ("default", "glorot"):
        torch.manual_seed(0)
        m = build_model("large", init=init).eval()
        nn.CrossEntropyLoss()(m(x), y).backward()
        norms[init] = m.features.conv1_1.weight.grad.norm().item()
    assert norms["glorot"] > 100 * norms["default"]


def test_unknown_init_raises():
    with pytest.raises(ValueError):
        build_model("baseline", init="he")

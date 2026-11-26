"""Kiểm số liệu bản đồ đặc trưng VGG16 mà src/e6_analysis.py đưa vào bảng E6.3 (T17)."""
from src.e6_analysis import vgg_feature_maps


def test_vgg16_48px_collapses_and_avgpool_only_copies():
    g = vgg_feature_maps(48)
    assert [s[1] for s in g["pools"]] == [24, 12, 6, 3, 1]
    assert g["conv5_side"] == 3 and g["conv5_used"] == "4/9"  # pool5 bỏ hàng/cột thứ 3
    assert g["avgpool"] == (512, 7, 7) and g["distinct_per_channel"] == 1
    assert g["fc1_in"] == 25088 and g["fc1_effective_params"] == 512 * 4096 + 4096
    assert g["rf_conv5"] == 196 > 48


def test_vgg16_96px_keeps_3x3_information():
    g = vgg_feature_maps(96)
    assert g["pools"][-1] == (512, 3, 3) and g["conv5_used"] == "36/36"
    assert g["pool5_cells"] == 9 and g["fc1_effective_params"] == 9 * 512 * 4096 + 4096

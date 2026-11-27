**Bảng 4.7b: Đánh giá TTA và Ensemble mô hình cuối E7 trên PublicTest (Validation)**

| Chế độ TTA | Loại đánh giá | Số seed | Val acc (mean ± std) | Val macro-F1 (mean ± std) | Val NLL |
|---|---|---:|---|---|---:|
| `none` | Đơn lẻ (trung bình) | 3 | 0.7060 ± 0.0023 | 0.6922 ± 0.0027 | 1.2175 |
| `none` | **Ensemble** | 3 | 0.7236 | 0.7134 | 0.9780 |
| `flip` | Đơn lẻ (trung bình) | 3 | 0.7150 ± 0.0024 | 0.7015 ± 0.0014 | 1.1157 |
| `flip` | **Ensemble** | 3 | 0.7250 | 0.7106 | 0.9402 |
| `ten_crop` | Đơn lẻ (trung bình) | 3 | 0.6836 ± 0.0043 | 0.6686 ± 0.0061 | 1.1232 |
| `ten_crop` | **Ensemble** | 3 | 0.6991 | 0.6839 | 0.9716 |

**Chi tiết từng seed trên PublicTest**

| Mã run | Seed | Chế độ TTA | Val acc | Val macro-F1 | Val NLL |
|---|---:|---|---:|---:|---:|
| `E1-014_s0` | 0 | `none` | 0.7086 | 0.6953 | 1.2014 |
| `E1-014_s1` | 1 | `none` | 0.7047 | 0.6910 | 1.2593 |
| `E1-014_s2` | 2 | `none` | 0.7047 | 0.6903 | 1.1919 |
| `E1-014_s0` | 0 | `flip` | 0.7166 | 0.7022 | 1.1061 |
| `E1-014_s1` | 1 | `flip` | 0.7122 | 0.6999 | 1.1382 |
| `E1-014_s2` | 2 | `flip` | 0.7161 | 0.7024 | 1.1027 |
| `E1-014_s0` | 0 | `ten_crop` | 0.6882 | 0.6753 | 1.1069 |
| `E1-014_s1` | 1 | `ten_crop` | 0.6826 | 0.6668 | 1.1450 |
| `E1-014_s2` | 2 | `ten_crop` | 0.6799 | 0.6636 | 1.1179 |

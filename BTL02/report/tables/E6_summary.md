# Bảng 4.6: Kết quả thực nghiệm E6 – Transfer learning (VGG16 và ResNet18)

| group | task_id | variant | n_runs | n_completed | n_params | val_acc_mean | val_acc_std | val_acc_fmt | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_fmt | best_val_loss_mean | train_val_gap_mean | gpu | epoch_time_s_tb_mean | so_voi_baseline_tu_tinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E6 | T17 | vgg16_48_ft | 3 | 1 | 134289223 | 0.693787 |  | 0.6938 | 0.681212 |  | 0.6812 | 0.9373 | 0.2571 | RTX 3050 | 60.5 | +0.0123 (cao hơn) |
| E6 | T17 | resnet18_96_scratch | 1 | 1 | 11180103 | 0.677069 |  | 0.6771 | 0.66719 |  | 0.6672 | 1.0006 | 0.2437 | RTX 3050 | 55.5 | -0.0045 (< std baseline) |
| E6 | T17 | resnet18_96_ft | 1 | 1 | 11180103 | 0.687378 |  | 0.6874 | 0.669849 |  | 0.6698 | 0.9718 | 0.2983 | RTX 3060 | 79.4 | +0.0059 (< std baseline) |
| E6 | T17 | resnet18_96_frozen | 1 | 1 | 11180103 | 0.456952 |  | 0.4570 | 0.368315 |  | 0.3683 | 1.4507 | -0.0133 | RTX 3060 | 75.0 | -0.2246 (thấp hơn) |
| E6 | T17 | vgg16_96_ft | 1 | 1 | 134289223 | 0.697687 |  | 0.6977 | 0.681773 |  | 0.6818 | 0.922 | 0.1876 | RTX 4060 | 71.3 | +0.0162 (cao hơn) |

* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc thời gian không lấy trung bình qua nhiều GPU khác nhau).

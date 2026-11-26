# T14 – Bằng chứng phần C1 của TV1

Phạm vi theo tracking/tasks.csv: TV1 chạy lưới SGD, SGD+Nesterov và Adam. Seed 0; chạy bằng src/run_experiment.py với configs/E3_optim.json.

Đối chiếu config.json của từng run với configs/baseline.json: sgd_no_nesterov_1e-2 chỉ đổi nesterov; hai run SGD+Nesterov chỉ đổi lr; hai run Adam đổi optimizer và lr theo lưới đã giao. Các run giữ model, batch size 128, scheduler, augmentation, weight decay và thiết lập khác của baseline.

| Variant | Run | Trạng thái | Best epoch | Best val accuracy | Best val macro-F1 | Thay đổi so với baseline |
|---|---|---:|---:|---:|---:|---|
| sgd_no_nesterov_1e-2 | E3-011_s0 | Completed | 100 | 0.6843 | 0.6442 | nesterov=False |
| sgd_nesterov_3e-2 | E3-012_s0 | Completed | 2 | 0.2494 | 0.0570 | lr=0.03 |
| sgd_nesterov_3e-3 | E3-013_s0 | Completed | 96 | 0.6743 | 0.6376 | lr=0.003 |
| adam_1e-3 | E3-014_s0 | Completed | 99 | 0.6865 | 0.6592 | optimizer=adam, lr=0.001 |
| adam_3e-4 (run dở) | E3-015_s0 | Aborted | 44 | 0.6498 | 0.5784 | optimizer=adam, lr=0.0003; không tính run hoàn tất |
| adam_3e-4 | E3-016_s0 | Completed | 100 | 0.6879 | 0.6587 | optimizer=adam, lr=0.0003; chạy lại hoàn tất |

Năm cấu hình được giao cho TV1 hiện có run hoàn tất: E3-011_s0 đến E3-014_s0 và E3-016_s0. Run SGD LR 0.03 dừng sớm theo cấu hình; runner vẫn ghi Completed.
E3-015_s0 bị ngắt ở epoch 46, được ghi Aborted/DISCARD và vẫn giữ trong lịch sử; không tính là kết quả hoàn tất. E3-016_s0 chạy lại đúng biến thể `adam_3e-4`, hoàn tất 100 epoch, `best_epoch=100`, `best_val_acc=0.687935`, `best_val_macro_f1=0.658696`, `best_val_loss=0.901274`; runner giữ `best.pt`. Run mới dùng FER2013 SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`.

Run, history, metrics, đường cong và log nằm dưới `results/E3/E3-016_s0/`; tracking tự ghi `tracking/experiments.csv` và `tracking/figures_manifest.csv`. Chưa tổng hợp bảng E3 cuối cùng vì phần đó thuộc R (TV3). PrivateTest không được dùng.

Phần C1 của TV1 hoàn tất. T14 tổng thể vẫn In Progress; không tự đổi trạng thái toàn nhiệm vụ. Nghiệm thu T14 thuộc TV2.

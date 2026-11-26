# T14 – Bằng chứng phần C1 của TV1

Phạm vi theo tracking/tasks.csv: TV1 chạy lưới SGD, SGD+Nesterov và Adam. Seed 0; chạy bằng src/run_experiment.py với configs/E3_optim.json.

Đối chiếu config.json của từng run với configs/baseline.json: sgd_no_nesterov_1e-2 chỉ đổi nesterov; hai run SGD+Nesterov chỉ đổi lr; hai run Adam đổi optimizer và lr theo lưới đã giao. Các run giữ model, batch size 128, scheduler, augmentation, weight decay và thiết lập khác của baseline.

| Variant | Run | Trạng thái | Best epoch | Best val accuracy | Best val macro-F1 | Thay đổi so với baseline |
|---|---|---:|---:|---:|---:|---|
| sgd_no_nesterov_1e-2 | E3-011_s0 | Completed | 100 | 0.6843 | 0.6442 | nesterov=False |
| sgd_nesterov_3e-2 | E3-012_s0 | Completed | 2 | 0.2494 | 0.0570 | lr=0.03 |
| sgd_nesterov_3e-3 | E3-013_s0 | Completed | 96 | 0.6743 | 0.6376 | lr=0.003 |
| adam_1e-3 | E3-014_s0 | Completed | 99 | 0.6865 | 0.6592 | optimizer=adam, lr=0.001 |
| adam_3e-4 | E3-015_s0 | Aborted | 44 | 0.6498 | 0.5784 | optimizer=adam, lr=0.0003 |

Bốn run E3-011_s0 đến E3-014_s0 hoàn tất; run SGD LR 0.03 dừng sớm theo cấu hình và runner ghi Completed.
E3-015_s0 bị ngắt sau epoch 46 khi phiên làm việc dừng. src.make_tables.sync_orphans() đã ghi Aborted (INTERRUPTED@epoch46), DISCARD và xóa checkpoint dở. Chỉ số run này là kết quả dở dang, không dùng như kết quả hoàn tất; cần chạy lại adam_3e-4 để xong phần C1.

Run, history, metrics và đường cong từng run nằm dưới results/E3/; tracking ở tracking/experiments.csv và tracking/figures_manifest.csv. Chưa tổng hợp bảng E3 cuối cùng vì phần đó thuộc R (TV3). PrivateTest không được dùng.

T14 tổng thể vẫn In Progress; chưa đánh dấu phần C1 hoàn tất cho đến khi adam_3e-4 chạy xong. Nghiệm thu T14 thuộc TV2.

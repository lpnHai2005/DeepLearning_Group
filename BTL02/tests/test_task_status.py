"""Test luật tự suy trạng thái nhiệm vụ của tracking/task_status.py. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T01.

Chạy từ thư mục BTL02: python -m pytest tests/test_task_status.py
"""
import importlib.util
from datetime import datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location("task_status", Path(__file__).resolve().parents[1] / "tracking" / "task_status.py")
task_status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task_status)


def t(day, hour=12):
    return datetime.fromisoformat(f"2026-11-{day:02d}T{hour:02d}:00:00+07:00")


def test(tid, day, ket_qua, hour=12):
    return {"task_id": tid, "thoi_gian": t(day, hour).isoformat(), "ket_qua": ket_qua}


test.__test__ = False  # hàm dựng dữ liệu, không phải test


def test_luat_trang_thai():
    commits = [
        (t(20), "feat(T02): khung repo"),
        (t(21), "feat(T03): kiểm tra dữ liệu"),
        (t(22), "done(T03): xong, chờ kiểm thử"),
        (t(21), "feat(T04): loader"),
        (t(22), "feat(T05): eda"),
        (t(22), "done(T05): xong"),
        (t(23), "fix(T05): sửa theo Fail"),
        (t(24), "done(T05): sửa xong, chờ kiểm lại"),
        (t(22), "feat(T06): cnn"),
        (t(23), "done(T06): xong"),
    ]
    tests = [
        test("T04", 23, "Pass"),
        test("T05", 23, "Fail"),
        test("T06", 22, "Pass", hour=8),  # Pass rồi R lại done sau đó: chờ kiểm lại
        test("T06", 20, "Fail"),
        test("T08", 24, "Pass"),  # có dòng kiểm thử dù chưa thấy commit nào
        test("T09", 24, "Partial"),  # kết quả khác Pass/Fail không tính
    ]
    s = task_status.derive([f"T{i:02d}" for i in range(1, 11)], commits, tests)
    assert {k: v[0] for k, v in s.items()} == {
        "T01": "Pending", "T02": "In Progress", "T03": "Completed", "T04": "Verified",
        "T05": "Completed", "T06": "Completed", "T07": "Pending", "T08": "Verified",
        "T09": "Pending", "T10": "Pending",
    }
    assert s["T05"][1] == t(24) and s["T01"][1] is None


def test_kiem_lai_sau_done_thi_theo_ket_qua_moi():
    commits = [(t(22), "done(T07): xong"), (t(23), "done(T07): sửa xong")]
    assert task_status.derive(["T07"], commits, [test("T07", 22, "Fail", 20)])["T07"][0] == "Completed"
    assert task_status.derive(["T07"], commits, [test("T07", 22, "Fail", 20), test("T07", 24, "Pass")])["T07"][0] == "Verified"
    assert task_status.derive(["T07"], commits, [test("T07", 24, "Fail")])["T07"][0] == "Failed"


def test_ma_nhiem_vu_trong_than_thong_diep_khong_tinh_la_done():
    commits = [(t(22), "feat(T09): thêm make_tables, done(T08) chưa liên quan")]
    s = task_status.derive(["T08", "T09"], commits, [])
    assert s["T08"][0] == "In Progress" and s["T09"][0] == "In Progress"

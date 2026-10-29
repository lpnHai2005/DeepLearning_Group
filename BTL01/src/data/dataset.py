"""Đọc CSV, scale, tạo DataLoader. (TV1 – T1.4)

load_split trả về numpy ở thang GỐC. make_loader scale x và y trước khi đưa vào mô hình.
"""


def load_split(name):
    """name in {'train_pool', 'val', 'test'} -> (x: (N,2), y: (N,))."""
    raise NotImplementedError("TV1 - T1.4")


def subset(x, y, n):
    """Lấy n mẫu đầu của pool (tập con lồng nhau cho Thực nghiệm A)."""
    raise NotImplementedError("TV1 - T1.4")


def make_loader(x, y, batch_size, shuffle, seed):
    """x, y thang gốc -> DataLoader (đã scale), shuffle dùng generator có seed."""
    raise NotImplementedError("TV1 - T1.4")

"""Đo thời gian huấn luyện bằng time.perf_counter() chỉ quanh fit(). (TV2)"""
import time


class Timer:
    """with Timer() as t: ...   -> t.elapsed (giây)"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0

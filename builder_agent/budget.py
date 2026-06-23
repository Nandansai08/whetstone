from __future__ import annotations

import threading


class TokenBudget:
    def __init__(self, limit: int = 0):
        self._limit = limit
        self._input_tokens = 0
        self._output_tokens = 0
        self._lock = threading.Lock()

    def record(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

    @property
    def total(self) -> int:
        with self._lock:
            return self._input_tokens + self._output_tokens

    @property
    def input_tokens(self) -> int:
        with self._lock:
            return self._input_tokens

    @property
    def output_tokens(self) -> int:
        with self._lock:
            return self._output_tokens

    def exceeded(self) -> bool:
        if self._limit <= 0:
            return False
        return self.total >= self._limit

    def usage(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
            "limit": self._limit,
        }

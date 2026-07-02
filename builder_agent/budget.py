from __future__ import annotations

import threading


class TokenBudget:
    def __init__(self, limit: int = 0, max_cost: float = 0.0):
        self._limit = limit
        self._max_cost = max_cost
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        self._cost = 0.0
        self._cost_unknown = False
        self._lock = threading.Lock()

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cache_read_tokens += cache_read_tokens
            self._cache_creation_tokens += cache_creation_tokens
            if cost is not None:
                if not self._cost_unknown:
                    self._cost += cost
            else:
                self._cost_unknown = True

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

    @property
    def cache_read_tokens(self) -> int:
        with self._lock:
            return self._cache_read_tokens

    @property
    def cache_creation_tokens(self) -> int:
        with self._lock:
            return self._cache_creation_tokens

    @property
    def cost(self) -> float | None:
        with self._lock:
            if self._cost_unknown:
                return None
            return self._cost

    def exceeded(self) -> bool:
        with self._lock:
            total_tokens = self._input_tokens + self._output_tokens
            if self._limit > 0 and total_tokens >= self._limit:
                return True
            if (
                not self._cost_unknown
                and self._max_cost > 0.0
                and self._cost >= self._max_cost
            ):
                return True
            return False

    def exceeded_reason(self) -> str | None:
        with self._lock:
            total_tokens = self._input_tokens + self._output_tokens
            if self._limit > 0 and total_tokens >= self._limit:
                return "token_budget"
            if (
                not self._cost_unknown
                and self._max_cost > 0.0
                and self._cost >= self._max_cost
            ):
                return "max_cost"
            return None

    def usage(self) -> dict:
        with self._lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
                "limit": self._limit,
                "cost": None if self._cost_unknown else self._cost,
                "max_cost": self._max_cost,
                "cache_read_tokens": self._cache_read_tokens,
                "cache_creation_tokens": self._cache_creation_tokens,
            }

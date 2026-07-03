"""Token budget tracking and accounting module."""

from __future__ import annotations

import threading


class TokenBudget:
    """Accumulates and enforces limits on token consumption during builds."""

    def __init__(self, limit: int = 0, max_cost: float = 0.0):
        """Initialize TokenBudget with a specific token limit and max cost.

        Args:
            limit: The maximum total tokens allowed before budget is exceeded.
            max_cost: The maximum monetary cost allowed before budget is exceeded.
        """
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
        """Record input and output tokens consumed, and optionally cost.

        Args:
            input_tokens: Number of input tokens to add.
            output_tokens: Number of output tokens to add.
            cost: Optional monetary cost of the tokens.
            cache_read_tokens: Number of cached prompt tokens read.
            cache_creation_tokens: Number of cached prompt tokens created/written.
        """
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
        """Return the sum of input and output tokens consumed."""
        with self._lock:
            return self._input_tokens + self._output_tokens

    @property
    def input_tokens(self) -> int:
        """Return the total input tokens consumed."""
        with self._lock:
            return self._input_tokens

    @property
    def output_tokens(self) -> int:
        """Return the total output tokens consumed."""
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
        """Return the total monetary cost consumed, or None if cost is unknown."""
        with self._lock:
            if self._cost_unknown:
                return None
            return self._cost

    def exceeded(self) -> bool:
        """Check if total tokens or cost consumed exceed the configured budget limit.

        Returns:
            True if token budget or max cost limit is exceeded, False otherwise.
        """
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
        """Return the reason why the budget was exceeded ('token_budget' or 'max_cost').

        Returns:
            String representing the exceeded reason, or None if not exceeded.
        """
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
        """Return a dictionary summarizing the budget usage and limits.

        Returns:
            A dictionary containing consumed tokens, cost, and limit details.
        """
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

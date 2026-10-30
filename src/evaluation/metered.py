"""Metering wrapper: counts LLM calls, latency and (reported or estimated) tokens/cost."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from controller.llm_client import LLMClient

T = TypeVar("T", bound=BaseModel)


@dataclass
class PriceTable:
    """USD per million tokens. Optional: without it cost is reported as None (never guessed)."""
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1e6 * self.input_per_mtok + output_tokens / 1e6 * self.output_per_mtok


@dataclass
class CallRecord:
    schema: str
    latency_sec: float
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool
    error: Optional[str] = None


class MeteredClient(LLMClient):
    def __init__(self, inner: LLMClient, price: Optional[PriceTable] = None, is_mock: bool = False) -> None:
        self.inner = inner
        self.price = price
        self.is_mock = is_mock
        self.calls: List[CallRecord] = []

    def _record(self, schema: str, started: float, prompt: str, system: Optional[str], out_text: str, error: Optional[str]) -> None:
        usage: Dict[str, int] = getattr(self.inner, "last_usage", {}) or {}
        if usage:
            i, o, est = usage.get("input_tokens", 0), usage.get("output_tokens", 0), False
        else:
            i, o, est = (len(prompt) + len(system or "")) // 4, len(out_text) // 4, True
        self.inner.last_usage = {} if hasattr(self.inner, "last_usage") else None
        self.calls.append(CallRecord(schema, time.perf_counter() - started, i, o, est, error))

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        t0 = time.perf_counter()
        try:
            out = self.inner.generate(prompt, system_prompt)
        except Exception as e:
            self._record("text", t0, prompt, system_prompt, "", str(e))
            raise
        self._record("text", t0, prompt, system_prompt, out, None)
        return out

    def generate_structured(self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None) -> T:
        t0 = time.perf_counter()
        try:
            out = self.inner.generate_structured(prompt, schema, system_prompt)
        except Exception as e:
            self._record(schema.__name__, t0, prompt, system_prompt, "", str(e))
            raise
        self._record(schema.__name__, t0, prompt, system_prompt, out.model_dump_json(), None)
        return out

    def summary(self) -> Dict[str, Any]:
        i = sum(c.input_tokens for c in self.calls)
        o = sum(c.output_tokens for c in self.calls)
        return {
            "calls": len(self.calls), "errors": sum(1 for c in self.calls if c.error),
            "latency_sec": round(sum(c.latency_sec for c in self.calls), 4),
            "input_tokens": i, "output_tokens": o,
            "tokens_estimated": any(c.tokens_estimated for c in self.calls),
            "cost_usd": round(self.price.cost(i, o), 6) if self.price else None,
            "is_mock": self.is_mock,
        }

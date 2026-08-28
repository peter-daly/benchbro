from __future__ import annotations

import json
import re

from benchbro import Case, system


@system(scope="session")
def compiled_pattern() -> re.Pattern[str]:
    return re.compile(r"[a-z]+")


encoding = Case(
    name="json-encoding",
    tags=["realistic", "serialization"],
    adaptive=True,
    min_repeats=5,
    repeats=50,
    min_iterations=10,
    max_time_s=3.0,
    target_relative_margin_pct=3.0,
    warning_threshold_pct=10.0,
    regression_threshold_pct=20.0,
)


@encoding.benchmark()
@encoding.parametrize("size", [100, 10_000], ids=["small", "large"])
@encoding.parametrize("sort_keys", [False, True], ids=["unsorted", "sorted"])
def encode_payload(size: int, sort_keys: bool) -> str:
    payload = {str(index): index for index in range(size)}
    return json.dumps(payload, sort_keys=sort_keys)


parsing = Case(
    name="text-parsing",
    tags=["realistic", "parsing"],
    setup_timing="exclude",
    teardown_timing="exclude",
)


@parsing.input()
def document() -> str:
    return "benchbro makes repeatable benchmarks " * 100


@parsing.benchmark()
def regex_tokens(document: str, compiled_pattern: re.Pattern[str]) -> list[str]:
    return compiled_pattern.findall(document)

from __future__ import annotations

import asyncio
import hashlib
import os

from benchbro import Case


def _slow_factor() -> int:
    raw = os.getenv("BENCHBRO_EXAMPLE_SLOW_FACTOR", "1")
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(1, value)


SLOW_FACTOR = _slow_factor()


hashing = Case(
    name="hashing",
    case_type="cpu",
    metric_type="time",
    tags=["demo", "cpu"],
    warmup_iterations=1,
    min_iterations=500,
    repeats=20,
)


@hashing.input()
def hash_payload() -> bytes:
    return b"benchbro-end-to-end-example"


@hashing.benchmark()
def sha1_digest(hash_payload: bytes) -> str:
    digest = ""
    for _ in range(SLOW_FACTOR):
        digest = hashlib.sha1(hash_payload).hexdigest()
    return digest


@hashing.benchmark()
def sha256_digest(hash_payload: bytes) -> str:
    digest = ""
    for _ in range(SLOW_FACTOR):
        digest = hashlib.sha256(hash_payload).hexdigest()
    return digest


allocations = Case(
    name="allocations",
    case_type="memory",
    metric_type="memory",
    tags=["demo", "memory"],
    warmup_iterations=1,
    min_iterations=50,
    repeats=20,
)


@allocations.input()
def allocation_size() -> int:
    return 500 * SLOW_FACTOR


@allocations.benchmark()
def list_allocation(allocation_size: int) -> list[int]:
    return [i for i in range(allocation_size)]


@allocations.benchmark()
def dict_allocation(allocation_size: int) -> dict[int, int]:
    return {i: i for i in range(allocation_size)}


async_io = Case(
    name="async_io",
    case_type="cpu",
    metric_type="time",
    tags=["demo", "async"],
    warmup_iterations=1,
    min_iterations=100,
    repeats=20,
)


@async_io.input()
async def async_payload() -> bytes:
    await asyncio.sleep(0)
    return b"benchbro-async-example"


@async_io.benchmark()
async def async_sha256_digest(async_payload: bytes) -> str:
    digest = ""
    for _ in range(SLOW_FACTOR):
        await asyncio.sleep(0)
        digest = hashlib.sha256(async_payload).hexdigest()
    return digest

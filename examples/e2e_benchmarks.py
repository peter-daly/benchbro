from __future__ import annotations

import asyncio
import hashlib
import os
import sys

from benchbro import Case, CommandSpec, system


@system(scope="module")
def iteration_scale() -> int:
    raw = os.getenv("BENCHBRO_EXAMPLE_SLOW_FACTOR", "1")
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(1, value)


@system()
def hash_prefix() -> bytes:
    return b"benchbro-"


@system()
def allocation_step(iteration_scale: int) -> int:
    return 500 * iteration_scale


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
    return b"end-to-end-example"


@hashing.benchmark()
def sha1_digest(hash_payload: bytes, hash_prefix: bytes, iteration_scale: int) -> str:
    payload = hash_prefix + hash_payload
    digest = ""
    for _ in range(iteration_scale):
        digest = hashlib.sha1(payload).hexdigest()
    return digest


@hashing.benchmark()
def sha256_digest(hash_payload: bytes, hash_prefix: bytes, iteration_scale: int) -> str:
    payload = hash_prefix + hash_payload
    digest = ""
    for _ in range(iteration_scale):
        digest = hashlib.sha256(payload).hexdigest()
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
    return 1


@allocations.benchmark()
def list_allocation(allocation_size: int, allocation_step: int) -> list[int]:
    return [i for i in range(allocation_size * allocation_step)]


@allocations.benchmark()
def dict_allocation(allocation_size: int, allocation_step: int) -> dict[int, int]:
    size = allocation_size * allocation_step
    return {i: i for i in range(size)}


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
async def async_sha256_digest(async_payload: bytes, iteration_scale: int) -> str:
    digest = ""
    for _ in range(iteration_scale):
        await asyncio.sleep(0)
        digest = hashlib.sha256(async_payload).hexdigest()
    return digest


subprocess_cli = Case(
    name="subprocess_cli",
    case_type="cpu",
    metric_type="time",
    tags=["demo", "subprocess"],
    warmup_iterations=1,
    min_iterations=50,
    repeats=20,
)


@subprocess_cli.input()
def cli_payload() -> str:
    return "benchbro subprocess"


@subprocess_cli.subprocess()
def python_uppercase(cli_payload: str) -> CommandSpec:
    return CommandSpec(
        argv=[
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(sys.argv[1].upper())",
            cli_payload,
        ]
    )

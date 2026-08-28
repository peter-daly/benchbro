from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

from benchbro.models import (
    BenchmarkCase,
    CanonicalSystemScope,
    CommandSpec,
    SystemDefinition,
    normalize_system_scope,
)


@dataclass
class ScopeState:
    cache: dict[str, Any] = field(default_factory=dict)
    finalizers: list[Callable[[], Any]] = field(default_factory=list)


@dataclass
class InvocationContext:
    case: BenchmarkCase
    systems: dict[str, SystemDefinition]
    session_state: ScopeState
    benchmark_state: ScopeState
    event_loop: asyncio.AbstractEventLoop
    iteration_state: ScopeState = field(default_factory=ScopeState)
    active_stack: list[str] = field(default_factory=list)


def _run_awaitable(
    value: Any,
    event_loop: asyncio.AbstractEventLoop | None = None,
    timeout_s: float | None = None,
) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if timeout_s is not None:
            value = asyncio.wait_for(value, timeout=timeout_s)
        if event_loop is None:
            return asyncio.run(value)
        return event_loop.run_until_complete(value)
    close = getattr(value, "close", None)
    if callable(close):
        close()
    raise RuntimeError(
        "benchbro cannot execute async benchmarks from within an active event loop"
    )


def has_running_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _resolve_maybe_awaitable(
    value: Any,
    event_loop: asyncio.AbstractEventLoop | None = None,
    timeout_s: float | None = None,
) -> Any:
    if inspect.isawaitable(value):
        return _run_awaitable(value, event_loop, timeout_s)
    return value


def validate_injected_callable(
    func: Callable[..., Any],
    *,
    label: str,
    allow_empty: bool,
) -> inspect.Signature:
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    if not allow_empty and parameters:
        raise ValueError(f"{label} cannot declare dependencies")
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(f"{label} cannot use positional-only parameters")
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(f"{label} cannot use *args")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(f"{label} cannot use **kwargs")
    return signature


def _resolve_input(case: BenchmarkCase, event_loop: asyncio.AbstractEventLoop) -> Any:
    if case.input_func is None:
        raise ValueError(f"Benchmark '{case.benchmark_name}' does not define an input")
    return _resolve_maybe_awaitable(case.input_func(), event_loop)


def _consume_generator_teardown(generator: Any, system_name: str) -> None:
    try:
        next(generator)
    except StopIteration:
        return
    raise RuntimeError(f"System '{system_name}' yielded more than once")


async def _consume_async_generator_teardown(generator: Any, name: str) -> None:
    try:
        await anext(generator)
    except StopAsyncIteration:
        return
    raise RuntimeError(f"System '{name}' yielded more than once")


def _resolve_system_result(
    system_name: str,
    result: Any,
    state: ScopeState,
    event_loop: asyncio.AbstractEventLoop,
) -> Any:
    if inspect.isasyncgen(result):
        try:
            value = _run_awaitable(anext(result), event_loop)
        except StopAsyncIteration as exc:
            raise RuntimeError(
                f"System '{system_name}' must yield exactly one value"
            ) from exc
        state.finalizers.append(
            lambda result=result, name=system_name: _consume_async_generator_teardown(
                result, name
            )
        )
        return value
    if inspect.isgenerator(result):
        try:
            value = next(result)
        except StopIteration as exc:
            raise RuntimeError(
                f"System '{system_name}' must yield exactly one value"
            ) from exc
        state.finalizers.append(
            lambda result=result, name=system_name: _consume_generator_teardown(
                result, name
            )
        )
        return value
    return _resolve_maybe_awaitable(result, event_loop)


def _scope_state(scope: CanonicalSystemScope, context: InvocationContext) -> ScopeState:
    if scope == "session":
        return context.session_state
    if scope == "benchmark":
        return context.benchmark_state
    return context.iteration_state


def _scope_rank(scope: CanonicalSystemScope) -> int:
    return {"iteration": 0, "benchmark": 1, "session": 2}[scope]


def resolve_callable_kwargs(
    func: Callable[..., Any],
    context: InvocationContext,
    *,
    owner_name: str,
    owner_kind: str,
    owner_scope: CanonicalSystemScope | None = None,
    owner_is_async: bool | None = None,
) -> dict[str, Any]:
    signature = validate_injected_callable(
        func, label=f"{owner_kind} '{owner_name}'", allow_empty=True
    )
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        matching_sources = sum(
            (
                parameter.name in context.case.parameters,
                context.case.input_name == parameter.name,
                parameter.name in context.systems,
            )
        )
        if matching_sources > 1:
            raise ValueError(f"Dependency name '{parameter.name}' is ambiguous")
        if parameter.name in context.case.parameters:
            if owner_kind == "system":
                raise ValueError(
                    f"System '{owner_name}' cannot depend on parameter '{parameter.name}'"
                )
            kwargs[parameter.name] = context.case.parameters[parameter.name]
            continue
        if context.case.input_name == parameter.name:
            if owner_kind == "system":
                raise ValueError(
                    f"System '{owner_name}' cannot depend on input '{parameter.name}'"
                )
            kwargs[parameter.name] = _resolve_input(context.case, context.event_loop)
            continue
        if parameter.name in context.systems:
            dependency = context.systems[parameter.name]
            dependency_scope = normalize_system_scope(dependency.scope)
            if owner_scope is not None and _scope_rank(owner_scope) > _scope_rank(
                dependency_scope
            ):
                owner_label = "module" if owner_scope == "session" else owner_scope
                dependency_label = (
                    "function" if dependency_scope == "benchmark" else dependency_scope
                )
                raise ValueError(
                    f"{owner_label.capitalize()}-scoped system '{owner_name}' "
                    f"cannot depend on {dependency_label}-scoped system '{parameter.name}'"
                )
            if owner_is_async is False and dependency.is_async:
                raise ValueError(
                    f"Sync {owner_kind} '{owner_name}' cannot depend on async system '{parameter.name}'"
                )
            kwargs[parameter.name] = resolve_system(parameter.name, context)
            continue
        if parameter.default is not inspect.Signature.empty:
            continue
        raise ValueError(
            f"{owner_kind.capitalize()} '{owner_name}' has unknown dependency '{parameter.name}'"
        )
    return kwargs


def resolve_system(system_name: str, context: InvocationContext) -> Any:
    system = context.systems.get(system_name)
    if system is None:
        raise ValueError(f"Unknown system '{system_name}'")
    scope = normalize_system_scope(system.scope)
    state = _scope_state(scope, context)
    if system_name in state.cache:
        return state.cache[system_name]
    if system_name in context.active_stack:
        cycle = " -> ".join([*context.active_stack, system_name])
        raise ValueError(f"System dependency cycle detected: {cycle}")
    context.active_stack.append(system_name)
    try:
        kwargs = resolve_callable_kwargs(
            system.func,
            context,
            owner_name=system_name,
            owner_kind="system",
            owner_scope=scope,
            owner_is_async=system.is_async,
        )
        value = _resolve_system_result(
            system_name, system.func(**kwargs), state, context.event_loop
        )
        state.cache[system_name] = value
        return value
    finally:
        context.active_stack.pop()


def _stdio_target(mode: str) -> Any:
    return subprocess.DEVNULL if mode == "devnull" else None


def _build_command_spec(case: BenchmarkCase, context: InvocationContext) -> CommandSpec:
    kwargs = resolve_callable_kwargs(
        case.func,
        context,
        owner_name=case.benchmark_name,
        owner_kind="subprocess benchmark",
        owner_is_async=False,
    )
    result = case.func(**kwargs)
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise ValueError(
            f"Subprocess benchmark '{case.benchmark_name}' cannot be async; return CommandSpec directly"
        )
    if not isinstance(result, CommandSpec):
        raise ValueError(
            f"Subprocess benchmark '{case.benchmark_name}' must return CommandSpec"
        )
    return result


def _execute_command(case: BenchmarkCase, spec: CommandSpec) -> None:
    env = None
    if spec.env is not None:
        env = os.environ.copy()
        env.update(spec.env)
    timeout = spec.timeout_s if spec.timeout_s is not None else case.settings.timeout_s
    try:
        completed = subprocess.run(
            spec.argv,
            cwd=str(spec.cwd) if spec.cwd is not None else None,
            env=env,
            input=spec.stdin_bytes,
            stdout=_stdio_target(spec.stdout),
            stderr=_stdio_target(spec.stderr),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Subprocess benchmark '{case.benchmark_name}' timed out after {timeout} seconds"
        ) from exc
    if completed.returncode not in spec.expected_exit_codes:
        raise RuntimeError(
            f"Subprocess benchmark '{case.benchmark_name}' exited with code {completed.returncode}"
        )


def prepare_invocation(
    case: BenchmarkCase, context: InvocationContext
) -> Callable[[], Any]:
    if case.kind == "subprocess":
        spec = _build_command_spec(case, context)
        return lambda: _execute_command(case, spec)
    kwargs = resolve_callable_kwargs(
        case.func,
        context,
        owner_name=case.benchmark_name,
        owner_kind="benchmark",
        owner_is_async=(
            inspect.iscoroutinefunction(case.func)
            or inspect.isasyncgenfunction(case.func)
        ),
    )

    def execute() -> Any:
        started = perf_counter_ns()
        value = _resolve_maybe_awaitable(
            case.func(**kwargs), context.event_loop, case.settings.timeout_s
        )
        elapsed_s = (perf_counter_ns() - started) / 1_000_000_000
        if case.settings.timeout_s is not None and elapsed_s > case.settings.timeout_s:
            raise RuntimeError(
                f"Benchmark '{case.benchmark_name}' exceeded its "
                f"{case.settings.timeout_s} second in-process timeout"
            )
        return value

    return execute


def execute_iteration(case: BenchmarkCase, context: InvocationContext) -> float:
    context.iteration_state = ScopeState()
    execute = None
    finalized = False
    try:
        if case.settings.setup_timing == "exclude":
            execute = prepare_invocation(case, context)
        started = perf_counter_ns()
        if execute is None:
            execute = prepare_invocation(case, context)
        execute()
        if case.settings.teardown_timing == "include":
            finalized = True
            finalize_scope(context.iteration_state, context.event_loop)
        elapsed_s = (perf_counter_ns() - started) / 1_000_000_000
    finally:
        if not finalized:
            finalize_scope(context.iteration_state, context.event_loop)
    return elapsed_s


def finalize_scope(state: ScopeState, event_loop: asyncio.AbstractEventLoop) -> None:
    error: Exception | None = None
    while state.finalizers:
        finalizer = state.finalizers.pop()
        try:
            _resolve_maybe_awaitable(finalizer(), event_loop)
        except Exception as exc:  # noqa: BLE001
            if error is None:
                error = exc
    state.cache.clear()
    if error is not None:
        raise error

"""A tiny execution-graph engine for the resume pipeline.

The pipeline is a DAG, not a line: resume parsing and JD analysis are independent
and should run concurrently, while the evidence mapper needs both. Declaring
dependencies per node buys three things for free:

  * **Parallelism** — nodes at the same topological level run in a thread pool.
    LLM calls are IO-bound, so this is a straight wall-clock win.
  * **Caching / resume** — a node whose inputs haven't changed reads its cached
    result, so re-running generation doesn't re-parse the resume.
  * **Observability** — every node's duration, status and error is recorded in
    the trace, which the UI renders as a pipeline view.

Deliberately ~200 lines and dependency-free. This is not a workflow framework;
it is the smallest thing that models the actual shape of the problem.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

NodeFn = Callable[["Context"], Any]


class GraphError(RuntimeError):
    """Raised when a node fails and the graph cannot continue."""

    def __init__(self, node: str, original: BaseException) -> None:
        super().__init__(f"node '{node}' failed: {original}")
        self.node = node
        self.original = original


@dataclass
class NodeTrace:
    name: str
    status: str = "pending"          # pending | running | ok | cached | skipped | error
    started_at: float = 0.0
    duration_ms: float = 0.0
    error: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
            "note": self.note,
        }


@dataclass
class Context:
    """Shared, append-only-ish state passed to every node."""

    values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    trace: list[NodeTrace] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.trace]


@dataclass
class Node:
    name: str
    fn: NodeFn
    deps: tuple[str, ...] = ()
    produces: str = ""               # context key this node writes (defaults to name)
    optional: bool = False           # a failure here warns instead of aborting
    retries: int = 1                 # total attempts on failure
    note: str = ""

    def key(self) -> str:
        return self.produces or self.name


class Graph:
    """Topologically levelled DAG with per-level parallel execution."""

    def __init__(self, name: str = "pipeline", max_workers: int = 4) -> None:
        self.name = name
        self.max_workers = max_workers
        self._nodes: dict[str, Node] = {}

    # -- construction ------------------------------------------------------
    def add(
        self,
        name: str,
        fn: NodeFn,
        deps: Iterable[str] = (),
        *,
        produces: str = "",
        optional: bool = False,
        retries: int = 1,
        note: str = "",
    ) -> "Graph":
        if name in self._nodes:
            raise ValueError(f"duplicate node: {name}")
        self._nodes[name] = Node(
            name=name,
            fn=fn,
            deps=tuple(deps),
            produces=produces,
            optional=optional,
            retries=retries,
            note=note,
        )
        return self

    # -- planning ----------------------------------------------------------
    def levels(self) -> list[list[Node]]:
        """Kahn's algorithm, grouped into levels of mutually independent nodes."""
        remaining = dict(self._nodes)
        for node in remaining.values():
            for dep in node.deps:
                if dep not in self._nodes:
                    raise ValueError(f"node '{node.name}' depends on unknown node '{dep}'")

        done: set[str] = set()
        out: list[list[Node]] = []
        while remaining:
            ready = [n for n in remaining.values() if all(d in done for d in n.deps)]
            if not ready:
                cycle = ", ".join(sorted(remaining))
                raise ValueError(f"cycle detected among: {cycle}")
            ready.sort(key=lambda n: n.name)
            out.append(ready)
            for n in ready:
                done.add(n.name)
                remaining.pop(n.name)
        return out

    def plan(self) -> list[list[str]]:
        return [[n.name for n in level] for level in self.levels()]

    # -- execution ---------------------------------------------------------
    def run(self, ctx: Context | None = None, *, only: set[str] | None = None) -> Context:
        """Execute the graph.

        `only` restricts execution to the named nodes plus their transitive
        dependencies; anything already present in the context is treated as
        cached and skipped. That is what makes `/generate` cheap after
        `/analyze` — the parse and JD nodes read from the cached context.
        """
        ctx = ctx or Context()
        wanted = self._closure(only) if only else set(self._nodes)

        for level in self.levels():
            batch = [n for n in level if n.name in wanted]
            if not batch:
                continue

            pending: list[Node] = []
            for node in batch:
                if node.key() in ctx.values:
                    ctx.trace.append(
                        NodeTrace(name=node.name, status="cached", note=node.note)
                    )
                else:
                    pending.append(node)

            if not pending:
                continue
            if len(pending) == 1:
                self._execute(pending[0], ctx)
            else:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending))) as pool:
                    list(pool.map(lambda n: self._execute(n, ctx), pending))
        return ctx

    def _closure(self, only: set[str]) -> set[str]:
        wanted: set[str] = set()
        stack = list(only)
        while stack:
            name = stack.pop()
            if name in wanted:
                continue
            if name not in self._nodes:
                raise ValueError(f"unknown node: {name}")
            wanted.add(name)
            stack.extend(self._nodes[name].deps)
        return wanted

    def _execute(self, node: Node, ctx: Context) -> None:
        tr = NodeTrace(name=node.name, status="running", started_at=time.time(), note=node.note)
        ctx.trace.append(tr)
        last_error: BaseException | None = None

        for attempt in range(1, max(1, node.retries) + 1):
            try:
                result = node.fn(ctx)
                if result is not None:
                    ctx.set(node.key(), result)
                tr.status = "ok"
                tr.duration_ms = (time.time() - tr.started_at) * 1000
                if attempt > 1:
                    tr.note = (tr.note + f" (succeeded on attempt {attempt})").strip()
                return
            except Exception as exc:  # noqa: BLE001 - recorded and re-raised below
                last_error = exc
                if attempt < max(1, node.retries):
                    time.sleep(0.6 * attempt)

        tr.duration_ms = (time.time() - tr.started_at) * 1000
        tr.status = "error"
        tr.error = f"{type(last_error).__name__}: {last_error}"

        if node.optional:
            ctx.warn(f"Optional stage '{node.name}' failed: {tr.error}")
            tr.status = "skipped"
            return

        detail = "".join(traceback.format_exception_only(type(last_error), last_error)).strip()
        tr.error = detail
        raise GraphError(node.name, last_error or RuntimeError("unknown failure"))

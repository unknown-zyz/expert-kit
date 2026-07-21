import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class PipelineTracer:
    # EngineCore may skip atexit during signal-driven shutdown. Periodic
    # snapshots retain a usable trace, but keep them sparse because each
    # snapshot serializes all events collected so far.
    _FLUSH_INTERVAL = 4096

    def __init__(self) -> None:
        self._path = os.getenv("EK_PIPELINE_TRACE", "")
        self._events: list[dict] = []
        self._last_stage_end: dict[int, int] = {}
        self._lock = threading.Lock()
        if self._path:
            atexit.register(self.flush)

    @property
    def enabled(self) -> bool:
        return bool(self._path)

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def last_stage_end(self, microbatch_id: int, default: int) -> int:
        with self._lock:
            return self._last_stage_end.get(microbatch_id, default)

    def mark_stage_end(self, microbatch_id: int, timestamp_ns: int) -> None:
        with self._lock:
            self._last_stage_end[microbatch_id] = timestamp_ns

    def record(
        self,
        stage: str,
        start_ns: int,
        end_ns: int,
        request_id: str,
        microbatch_id: int,
        layer_id: int,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "name": stage,
            "cat": "expertkit_pipeline",
            "ph": "X",
            "ts": start_ns / 1000,
            "dur": max(0, end_ns - start_ns) / 1000,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "args": {
                "request_id": request_id,
                "microbatch_id": microbatch_id,
                "layer_id": layer_id,
            },
        }
        with self._lock:
            self._events.append(event)
            should_flush = len(self._events) % self._FLUSH_INTERVAL == 0
        # EngineCore is terminated through a signal and does not reliably run
        # Python atexit handlers. Periodic snapshots preserve an actionable
        # trace without putting file I/O on every stage transition.
        if should_flush:
            self.flush()

    @contextmanager
    def nvtx(
        self,
        stage: str,
        microbatch_id: int,
        layer_id: int,
        request_id: str = "",
    ):
        label = (
            f"EK:{stage}:req={request_id}:u={microbatch_id}:l={layer_id}"
        )
        pushed = False
        if torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(label)
                pushed = True
            except RuntimeError:
                pass
        try:
            yield
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            events = list(self._events)
        # The plugin is imported in both the parent and EngineCore process.
        # Never let an idle parent overwrite the child process's real trace.
        if not events:
            return
        path = Path(self._path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({"traceEvents": events}, indent=2))
        temporary.replace(path)


pipeline_tracer = PipelineTracer()

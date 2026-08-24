"""Bounded FIFO scheduler for serial tokenizer computation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter


class QueueFullError(RuntimeError):
    """Raised when no waiting slot is available."""


class QueueTimeoutError(RuntimeError):
    """Raised when a job does not start before its queue deadline."""


@dataclass(frozen=True)
class ExecutionResult:
    value: int
    queue_wait_ms: float
    process_ms: float


@dataclass
class _Job:
    call: Callable[[], int]
    enqueued_at: float
    started: asyncio.Future[None]
    completed: asyncio.Future[ExecutionResult]
    cancelled: bool = field(default=False)


class SerialScheduler:
    """Run synchronous jobs FIFO on exactly one executor thread."""

    def __init__(self, *, queue_size: int, queue_timeout_seconds: float) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        self.queue_timeout_seconds = queue_timeout_seconds
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=queue_size)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenizer-worker"
        )
        self._consumer: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._consume())

    async def submit(self, call: Callable[[], int]) -> ExecutionResult:
        if self._closed or self._consumer is None:
            raise RuntimeError("scheduler is not running")
        loop = asyncio.get_running_loop()
        job = _Job(
            call=call,
            enqueued_at=perf_counter(),
            started=loop.create_future(),
            completed=loop.create_future(),
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as error:
            raise QueueFullError("tokenizer queue is full") from error

        try:
            await asyncio.wait_for(
                asyncio.shield(job.started), timeout=self.queue_timeout_seconds
            )
        except TimeoutError as error:
            job.cancelled = True
            raise QueueTimeoutError("tokenizer queue wait timed out") from error
        except asyncio.CancelledError:
            job.cancelled = True
            raise
        return await asyncio.shield(job.completed)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass
        while not self._queue.empty():
            job = self._queue.get_nowait()
            job.cancelled = True
            if not job.started.done():
                job.started.cancel()
            if not job.completed.done():
                job.completed.cancel()
            self._queue.task_done()
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            try:
                if job.cancelled:
                    continue
                started_at = perf_counter()
                job.started.set_result(None)
                try:
                    value = await loop.run_in_executor(self._executor, job.call)
                except BaseException as error:
                    if not job.completed.done():
                        job.completed.set_exception(error)
                else:
                    if not job.completed.done():
                        job.completed.set_result(
                            ExecutionResult(
                                value=value,
                                queue_wait_ms=(started_at - job.enqueued_at) * 1000,
                                process_ms=(perf_counter() - started_at) * 1000,
                            )
                        )
            finally:
                self._queue.task_done()

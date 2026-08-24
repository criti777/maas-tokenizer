import asyncio
from threading import Event, Lock
import time

import pytest

from maas_tokenizer.scheduler import (
    QueueFullError,
    QueueTimeoutError,
    SerialScheduler,
)


def test_scheduler_executes_jobs_one_at_a_time_in_fifo_order() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=3, queue_timeout_seconds=1)
        await scheduler.start()
        release = Event()
        first_started = Event()
        order: list[int] = []
        active = 0
        maximum_active = 0
        guard = Lock()

        def work(number: int, block: bool = False) -> int:
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
                order.append(number)
            if block:
                first_started.set()
                release.wait(timeout=2)
            with guard:
                active -= 1
            return number

        try:
            first = asyncio.create_task(scheduler.submit(lambda: work(1, True)))
            await asyncio.to_thread(first_started.wait, 1)
            second = asyncio.create_task(scheduler.submit(lambda: work(2)))
            third = asyncio.create_task(scheduler.submit(lambda: work(3)))
            await asyncio.sleep(0.02)
            release.set()
            results = await asyncio.gather(first, second, third)
        finally:
            await scheduler.close()

        assert [result.value for result in results] == [1, 2, 3]
        assert order == [1, 2, 3]
        assert maximum_active == 1
        assert all(result.queue_wait_ms >= 0 for result in results)
        assert all(result.process_ms >= 0 for result in results)

    asyncio.run(scenario())


def test_scheduler_rejects_when_waiting_queue_is_full() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=1, queue_timeout_seconds=1)
        await scheduler.start()
        release = Event()
        started = Event()

        def blocking() -> int:
            started.set()
            release.wait(timeout=2)
            return 1

        try:
            running = asyncio.create_task(scheduler.submit(blocking))
            await asyncio.to_thread(started.wait, 1)
            waiting = asyncio.create_task(scheduler.submit(lambda: 2))
            await asyncio.sleep(0.02)
            with pytest.raises(QueueFullError):
                await scheduler.submit(lambda: 3)
            release.set()
            await asyncio.gather(running, waiting)
        finally:
            release.set()
            await scheduler.close()

    asyncio.run(scenario())


def test_timed_out_job_is_not_executed() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=2, queue_timeout_seconds=0.03)
        await scheduler.start()
        release = Event()
        started = Event()
        timed_out_executed = Event()

        def blocking() -> int:
            started.set()
            release.wait(timeout=2)
            return 1

        try:
            running = asyncio.create_task(scheduler.submit(blocking))
            await asyncio.to_thread(started.wait, 1)
            with pytest.raises(QueueTimeoutError) as captured:
                await scheduler.submit(lambda: timed_out_executed.set() or 2)
            assert captured.value.queue_wait_ms >= 20
            release.set()
            await running
            await asyncio.sleep(0.03)
        finally:
            release.set()
            await scheduler.close()

        assert not timed_out_executed.is_set()

    asyncio.run(scenario())


def test_failed_job_exposes_queue_and_processing_timings() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=1, queue_timeout_seconds=1)
        await scheduler.start()

        def fail() -> int:
            time.sleep(0.01)
            raise ValueError("boom")

        try:
            with pytest.raises(ValueError, match="boom") as captured:
                await scheduler.submit(fail)
        finally:
            await scheduler.close()

        assert captured.value.queue_wait_ms >= 0
        assert captured.value.process_ms >= 5

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("queue_size", "timeout"), [(0, 1), (1, 0)]
)
def test_scheduler_rejects_non_positive_configuration(
    queue_size: int, timeout: float
) -> None:
    with pytest.raises(ValueError):
        SerialScheduler(queue_size=queue_size, queue_timeout_seconds=timeout)


def test_scheduler_rejects_submit_before_start() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=1, queue_timeout_seconds=1)
        with pytest.raises(RuntimeError, match="not running"):
            await scheduler.submit(lambda: 1)
        await scheduler.close()
        await scheduler.close()

    asyncio.run(scenario())


def test_close_waits_for_running_job_and_cancels_waiting_job() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=1, queue_timeout_seconds=1)
        await scheduler.start()
        release = Event()
        started = Event()
        waiting_executed = Event()

        def blocking() -> int:
            started.set()
            release.wait(timeout=2)
            return 7

        running = asyncio.create_task(scheduler.submit(blocking))
        await asyncio.to_thread(started.wait, 1)
        waiting = asyncio.create_task(
            scheduler.submit(lambda: waiting_executed.set() or 8)
        )
        await asyncio.sleep(0.02)
        close_task = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0.02)
        assert not close_task.done()
        release.set()

        result = await running
        await close_task
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert result.value == 7
        assert not waiting_executed.is_set()

    asyncio.run(scenario())


def test_cancelled_waiting_submit_is_not_executed() -> None:
    async def scenario() -> None:
        scheduler = SerialScheduler(queue_size=1, queue_timeout_seconds=1)
        await scheduler.start()
        release = Event()
        started = Event()
        cancelled_executed = Event()

        def blocking() -> int:
            started.set()
            release.wait(timeout=2)
            return 1

        try:
            running = asyncio.create_task(scheduler.submit(blocking))
            await asyncio.to_thread(started.wait, 1)
            cancelled = asyncio.create_task(
                scheduler.submit(lambda: cancelled_executed.set() or 2)
            )
            await asyncio.sleep(0.02)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            release.set()
            await running
            await asyncio.sleep(0.02)
        finally:
            release.set()
            await scheduler.close()
        assert not cancelled_executed.is_set()

    asyncio.run(scenario())

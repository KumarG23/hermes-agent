"""SQLite completion contention and cancellation with the real agent/API/store.

The responsiveness case adapts the lead's exact-head PR1 review reproducer.
All state is disposable; the provider fixture is deterministic and local.
"""

import asyncio
import sqlite3
import threading

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_run_message_binding import db, local_agent_factory, client_for


@pytest.mark.asyncio
async def test_completion_write_keeps_api_loop_responsive(
    db, monkeypatch, local_agent_factory
):
    factory, calls = local_agent_factory
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", factory)
    real_bind = db.bind_run_messages
    started = threading.Event()
    finished = threading.Event()
    heartbeat_during_write = threading.Event()

    def contended_bind(*args, **kwargs):
        locker = sqlite3.connect(db.db_path, check_same_thread=False)
        locker.execute("BEGIN IMMEDIATE")

        def release():
            # Event ordering is the assertion; timeout only breaks the original
            # loop-stall deadlock so the failing regression can finish.
            heartbeat_during_write.wait(2)
            locker.rollback()
            locker.close()

        unlocker = threading.Thread(target=release)
        unlocker.start()
        started.set()
        try:
            return real_bind(*args, **kwargs)
        finally:
            unlocker.join()
            finished.set()

    monkeypatch.setattr(db, "bind_run_messages", contended_bind)

    async def heartbeat():
        while not finished.is_set():
            if started.is_set():
                heartbeat_during_write.set()
            await asyncio.sleep(0.005)

    pulse = asyncio.create_task(heartbeat())
    try:
        async with await client_for(adapter) as client:
            response = await client.post(
                "/v1/runs", json={"input": "same", "session_id": "owned"}
            )
            run_id = (await response.json())["run_id"]
            await adapter._active_run_tasks[run_id]
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "completed"
            assert "history_binding" in status
        await pulse
        assert heartbeat_during_write.is_set(), (
            "SQLite binding write stalled the API event loop"
        )
        assert len(calls) == 1
        assert len(db.get_messages("owned")) == 2
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)


async def wait_for_signal(signal):
    assert await asyncio.wait_for(asyncio.to_thread(signal.wait, 5), 6), (
        "fixture barrier not reached"
    )


async def assert_recovery(adapter, db, client, run_id, expected):
    status = await (await client.get(f"/v1/runs/{run_id}")).json()
    assert status["status"] == expected
    recovered = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    recovered._session_db = db
    async with await client_for(recovered) as restored:
        response = await restored.get(f"/v1/runs/{run_id}")
        if expected == "completed":
            assert response.status == 200
            saved = await response.json()
            assert saved["recovered"] is True
            assert saved["history_binding"] == status["history_binding"]
        else:
            assert "history_binding" not in status
            assert db.get_run_message_binding(run_id) is None
            assert response.status == 404
    assert len(db.get_messages("owned")) == 2
    assert run_id not in adapter._active_run_tasks
    assert run_id not in adapter._run_binding_completions


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "cancel"])
async def test_cancellation_while_sqlite_waits_has_no_completed_recovery(
    db,
    monkeypatch,
    local_agent_factory,
    control,
):
    factory, calls = local_agent_factory
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", factory)
    real_bind = db.bind_run_messages
    locked = threading.Event()
    locker_holder = []

    def contended_bind(*args, **kwargs):
        locker = sqlite3.connect(db.db_path, check_same_thread=False)
        locker.execute("BEGIN IMMEDIATE")
        locker_holder.append(locker)
        locked.set()
        return real_bind(*args, **kwargs)

    monkeypatch.setattr(db, "bind_run_messages", contended_bind)
    task = None
    async with await client_for(adapter) as client:
        try:
            response = await client.post(
                "/v1/runs", json={"input": "same", "session_id": "owned"}
            )
            run_id = (await response.json())["run_id"]
            task = adapter._active_run_tasks[run_id]
            await wait_for_signal(locked)
            # HTTP polling also stays available while the independent writer
            # still holds BEGIN IMMEDIATE, not merely an artificial sleep.
            status = await asyncio.wait_for(client.get(f"/v1/runs/{run_id}"), 5)
            assert (await status.json())["status"] == "running"
            if control == "stop":
                response = await asyncio.wait_for(
                    client.post(f"/v1/runs/{run_id}/stop"), 5
                )
                assert response.status == 200
            else:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()  # repeated cancellation cannot abandon the worker
                await asyncio.sleep(0)
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "stopping"
            assert "history_binding" not in status
            assert run_id in adapter._active_run_tasks
        finally:
            for locker in locker_holder:
                await asyncio.to_thread(locker.rollback)
                locker.close()
            if task is not None:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await assert_recovery(adapter, db, client, run_id, "cancelled")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "cancel"])
@pytest.mark.parametrize("boundary", ["claimed", "committed"])
async def test_claimed_completion_settles_even_if_wrapper_is_cancelled(
    db,
    monkeypatch,
    local_agent_factory,
    control,
    boundary,
):
    factory, calls = local_agent_factory
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", factory)
    real_bind = db.bind_run_messages
    paused = threading.Event()
    release = threading.Event()

    def paused_bind(*args, **kwargs):
        guard = kwargs["commit_guard"]

        def claim():
            won = guard()
            assert won
            if boundary == "claimed":
                paused.set()
                assert release.wait(5)
            return won

        kwargs["commit_guard"] = claim
        committed = real_bind(*args, **kwargs)
        if boundary == "committed":
            paused.set()
            assert release.wait(5)
        return committed

    monkeypatch.setattr(db, "bind_run_messages", paused_bind)
    task = None
    async with await client_for(adapter) as client:
        try:
            response = await client.post(
                "/v1/runs", json={"input": "same", "session_id": "owned"}
            )
            run_id = (await response.json())["run_id"]
            task = adapter._active_run_tasks[run_id]
            await wait_for_signal(paused)
            if control == "stop":
                response = await asyncio.wait_for(
                    client.post(f"/v1/runs/{run_id}/stop"), 5
                )
                assert response.status == 409
                assert (await response.json())["error"]["code"] == "run_completing"
            else:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
            # Neither transport events nor status may claim completion early.
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "running"
            assert "history_binding" not in status
            assert not any(
                event and event.get("event") == "run.completed"
                for event in adapter._run_streams[run_id]._queue
            )
            assert run_id in adapter._active_run_tasks
        finally:
            release.set()
            if task is not None:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await assert_recovery(adapter, db, client, run_id, "completed")
        # Repeated polling keeps the same pair and never calls the provider again.
        await assert_recovery(adapter, db, client, run_id, "completed")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["stop", "cancel"])
async def test_failed_claimed_commit_never_recovers_as_completed(
    db,
    monkeypatch,
    local_agent_factory,
    control,
):
    factory, calls = local_agent_factory
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", factory)
    db._execute_write(
        lambda conn: conn.execute(
            "CREATE TRIGGER fail_binding BEFORE UPDATE OF api_run_id ON messages "
            "BEGIN SELECT RAISE(ABORT, 'fixture completion failure'); END"
        )
    )
    real_bind = db.bind_run_messages
    claimed = threading.Event()
    release = threading.Event()

    def failed_bind(*args, **kwargs):
        guard = kwargs["commit_guard"]

        def claim():
            assert guard()
            claimed.set()
            assert release.wait(5)
            return True

        kwargs["commit_guard"] = claim
        return real_bind(*args, **kwargs)

    monkeypatch.setattr(db, "bind_run_messages", failed_bind)
    task = None
    async with await client_for(adapter) as client:
        try:
            response = await client.post(
                "/v1/runs", json={"input": "same", "session_id": "owned"}
            )
            run_id = (await response.json())["run_id"]
            task = adapter._active_run_tasks[run_id]
            await wait_for_signal(claimed)
            if control == "stop":
                response = await asyncio.wait_for(
                    client.post(f"/v1/runs/{run_id}/stop"), 5
                )
                assert response.status == 409
            else:
                task.cancel()
                await asyncio.sleep(0)
        finally:
            release.set()
            if task is not None:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await assert_recovery(adapter, db, client, run_id, "failed")
    assert len(calls) == 1

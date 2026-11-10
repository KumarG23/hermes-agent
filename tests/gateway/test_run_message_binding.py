"""Source-only exact run binding integration; no provider or live service."""

from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from run_agent import AIAgent


def flush_agent(db, session="owned"):
    # Real persistence implementation without initializing any provider/tools.
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session
    agent._session_db = db
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent._run_binding_rows = {}
    return agent


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session("owned", source="api_server")
    store.create_session("other", source="api_server")
    yield store
    store.close()


def committed_pair(db, session="owned"):
    agent = flush_agent(db, session)
    messages = [
        {"role": "user", "content": "same"},
        {"role": "assistant", "content": "same"},
    ]
    assert agent._flush_messages_to_session_db(messages, []) is True
    return agent, messages


def test_committed_ids_survive_reflush_without_duplicate(db):
    agent, messages = committed_pair(db)
    rows = db.get_messages("owned")
    assert [
        (agent._run_binding_rows[id(m)][1], agent._run_binding_rows[id(m)][2])
        for m in messages
    ] == [("owned", row["id"]) for row in rows]
    assert agent._flush_messages_to_session_db(messages, []) is True
    assert len(db.get_messages("owned")) == 2


def test_binding_is_durable_and_immutable(db):
    agent, first = committed_pair(db)
    pair = [agent._run_binding_rows[id(m)][2] for m in first]
    assert db.bind_run_messages("run_first", "owned", *pair)
    agent2, second = committed_pair(db)
    newer = [agent2._run_binding_rows[id(m)][2] for m in second]
    assert db.bind_run_messages("run_second", "owned", *newer)
    assert not db.bind_run_messages("run_first", "owned", *newer)
    assert db.get_run_message_binding("run_first")["history_binding"] == {
        "version": 1,
        "user_message_id": pair[0],
        "assistant_message_id": pair[1],
    }
    path = db.db_path
    reopened = SessionDB(db_path=Path(path))
    try:
        assert reopened.get_run_message_binding(
            "run_first"
        ) == db.get_run_message_binding("run_first")
    finally:
        reopened.close()


@pytest.fixture
def local_agent_factory(monkeypatch, db):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # No discovery, plugins, title model or real SDK transport. The agent
    # prologue, conversation loop, finalizer, SessionDB and API stay real.
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **kw: 8192
    )
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda **kw: {})
    monkeypatch.setattr(
        "agent.turn_context._maybe_title_session_at_turn_start", lambda *a: None
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [])
    calls = []

    def factory(**kwargs):
        with patch("run_agent.OpenAI"):
            agent = AIAgent(
                api_key="local-test-placeholder",
                base_url="http://127.0.0.1:1/v1",
                model="local-fixture",
                provider="openai",
                quiet_mode=True,
                skip_memory=True,
                skip_context_files=True,
                enabled_toolsets=[],
                session_id=kwargs["session_id"],
                session_db=db,
            )
        agent.client = MagicMock()
        msg = SimpleNamespace(
            content="same", reasoning_content=None, reasoning=None, tool_calls=None
        )
        answer = SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            model="local-fixture",
            usage=None,
        )

        def complete(**kw):
            calls.append(kw["messages"])
            return answer

        agent.client.chat.completions.create.side_effect = complete
        agent._cached_system_prompt = "Local deterministic fixture."
        agent._use_prompt_caching = False
        agent._disable_streaming = True
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.tool_delay = 0
        return agent

    return factory, calls


async def client_for(adapter):
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get(
        "/api/sessions/{session_id}/messages", adapter._handle_session_messages
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_real_agent_api_history_and_restart(db, monkeypatch, local_agent_factory):
    factory, calls = local_agent_factory
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", factory)
    async with await client_for(adapter) as client:
        saved = []
        for _ in range(2):
            response = await client.post(
                "/v1/runs",
                json={
                    "input": "same",
                    "session_id": "owned",
                    "conversation_history": db.get_messages_as_conversation("owned"),
                },
            )
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            await adapter._active_run_tasks[run_id]
            response = await client.get(f"/v1/runs/{run_id}")
            status = await response.json()
            assert status["status"] == "completed", status
            assert status["history_binding"]["version"] == 1
            saved.append((run_id, status["history_binding"]))
        assert saved[0][1] != saved[1][1]
        assert len(calls) == 2
        history_response = await client.get("/api/sessions/owned/messages")
        assert history_response.status == 200
        history = (await history_response.json())["data"]
        by_id = {row["id"]: row for row in history}
        for run_id, pair in saved:
            assert by_id[pair["user_message_id"]]["role"] == "user"
            assert by_id[pair["assistant_message_id"]]["role"] == "assistant"
            for _ in range(2):
                assert (await (await client.get(f"/v1/runs/{run_id}")).json())[
                    "history_binding"
                ] == pair
        assert len(db.get_messages("owned")) == 4
        assert len(calls) == 2

    # New API + new DB handle, with no agent capable of admitting/replaying.
    reopened = SessionDB(db_path=Path(db.db_path))
    restored = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    restored._session_db = reopened
    monkeypatch.setattr(
        restored, "_create_agent", lambda **kw: pytest.fail("replayed run")
    )
    try:
        async with await client_for(restored) as client:
            for run_id, pair in saved:
                status = await (await client.get(f"/v1/runs/{run_id}")).json()
                assert status["recovered"] is True
                assert status["history_binding"] == pair
                assert "usage" not in status
            assert (await client.get("/v1/runs/run_legacy")).status == 404
        assert len(reopened.get_messages("owned")) == 4
    finally:
        reopened.close()


def test_only_exact_final_object_can_bind(db):
    from agent.run_message_binding import committed_pair as candidate

    agent = flush_agent(db)
    user = {"role": "user", "content": "same"}
    intermediate = {
        "role": "assistant",
        "content": "same",
        "tool_calls": [{"id": "call"}],
    }
    tool = {"role": "tool", "content": "same", "tool_call_id": "call"}
    final = {"role": "assistant", "content": "same"}
    messages = [user, intermediate, tool, final]
    agent._run_binding_user = user
    assert agent._flush_messages_to_session_db(messages, []) is True
    result = {"completed": True, "final_response": "same"}
    exact = candidate(agent, final, result)
    rows = db.get_messages("owned")
    assert exact == [rows[0]["id"], rows[3]["id"]]
    assert candidate(agent, intermediate, result) is None
    assert candidate(agent, tool, result) is None
    # Compression/fork copies and identical text do not inherit an identity.
    assert candidate(agent, dict(final), result) is None
    agent._run_binding_user = dict(user)
    assert candidate(agent, final, result) is None
    agent._run_binding_user = user
    for bad in (
        {"failed": True},
        {"interrupted": True},
        {"completed": False},
        {"response_transformed": True},
        {"final_response": "changed"},
    ):
        assert candidate(agent, final, {**result, **bad}) is None
    agent.session_id = "other"
    assert candidate(agent, final, result) is None


def test_append_failure_never_retains_uncommitted_identity(db):
    agent = flush_agent(db)
    messages = [
        {"role": "user", "content": "same"},
        {"role": "assistant", "content": "same"},
    ]
    db._execute_write(
        lambda conn: conn.execute(
            "CREATE TRIGGER fail_answer BEFORE INSERT ON messages "
            "WHEN NEW.role = 'assistant' BEGIN SELECT RAISE(ABORT, 'fixture failure'); END"
        )
    )
    assert agent._flush_messages_to_session_db(messages, []) is False
    assert agent._run_binding_rows == {}
    assert db.get_messages("owned") == []
    assert not any(m.get("_db_persisted") for m in messages)


def test_binding_rejects_missing_conflicting_cross_session_and_inactive(db):
    agent, messages = committed_pair(db)
    user, answer = [agent._run_binding_rows[id(m)][2] for m in messages]
    foreign = db.append_message("other", "assistant", "same")
    for ids, session in (
        ((user, 999999), "owned"),
        ((user, foreign), "owned"),
        ((answer, user), "owned"),
        ((user, answer), "other"),
    ):
        assert not db.bind_run_messages("run_bad", session, *ids)
    assert db.get_run_message_binding("run_bad") is None
    assert db.bind_run_messages("run_good", "owned", user, answer)
    assert db.bind_run_messages(
        "run_good", "owned", user, answer
    )  # idempotent confirmation
    assert not db.bind_run_messages("run_steal", "owned", user, answer)
    db._execute_write(
        lambda conn: conn.execute("UPDATE messages SET active=0 WHERE id=?", (answer,))
    )
    assert db.get_run_message_binding("run_good") is None
    assert not db.bind_run_messages("run_good", "owned", user, answer)


def test_ambiguous_or_cross_session_durable_rows_fail_closed(db):
    agent, messages = committed_pair(db)
    ids = [agent._run_binding_rows[id(m)][2] for m in messages]
    assert db.bind_run_messages("run_ambiguous", "owned", *ids)
    extra = db.append_message("other", "assistant", "same")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET api_run_id=? WHERE id=?", ("run_ambiguous", extra)
        )
    )
    assert db.get_run_message_binding("run_ambiguous") is None
    db._execute_write(
        lambda conn: conn.execute("DELETE FROM messages WHERE id=?", (ids[1],))
    )
    assert db.get_run_message_binding("run_ambiguous") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["binding", "transcript"])
async def test_binding_commit_failure_is_failed_api_status(
    db, monkeypatch, local_agent_factory, failure
):
    factory, calls = local_agent_factory
    operation = "UPDATE OF api_run_id" if failure == "binding" else "INSERT"
    db._execute_write(
        lambda conn: conn.execute(
            f"CREATE TRIGGER fail_binding BEFORE {operation} ON messages "
            "BEGIN SELECT RAISE(ABORT, 'fixture persistence failure'); END"
        )
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    monkeypatch.setattr(adapter, "_create_agent", factory)
    async with await client_for(adapter) as client:
        response = await client.post(
            "/v1/runs", json={"input": "same", "session_id": "owned"}
        )
        run_id = (await response.json())["run_id"]
        await adapter._active_run_tasks[run_id]
        status = await (await client.get(f"/v1/runs/{run_id}")).json()
        assert status["status"] == "failed"
        assert "history_binding" not in status
        assert db.get_run_message_binding(run_id) is None
    assert len(calls) == 1
    assert len(db.get_messages("owned")) == (2 if failure == "binding" else 0)


@pytest.mark.asyncio
async def test_status_revalidates_rewind_and_profile(db, tmp_path, monkeypatch):
    agent, messages = committed_pair(db)
    ids = [agent._run_binding_rows[id(m)][2] for m in messages]
    assert db.bind_run_messages("run_exact", "owned", *ids)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db
    from hermes_constants import get_hermes_home

    adapter._run_binding_homes["run_exact"] = str(get_hermes_home())
    adapter._set_run_status("run_exact", "completed", session_id="owned")
    async with await client_for(adapter) as client:
        assert (
            "history_binding" in await (await client.get("/v1/runs/run_exact")).json()
        )
        # A selected other profile cannot receive an in-memory pair from the
        # first profile, nor recover it from its own unrelated store.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-other"))
        assert (
            "history_binding"
            not in await (await client.get("/v1/runs/run_exact")).json()
        )
        other = SessionDB(db_path=tmp_path / "profile-other.db")
        adapter._session_db = other
        adapter._run_statuses.clear()
        try:
            assert (await client.get("/v1/runs/run_exact")).status == 404
        finally:
            other.close()
        adapter._session_db = db
        # Restore and then retire the exact answer; no cached pair survives.
        adapter._run_binding_homes["run_exact"] = str(get_hermes_home())
        adapter._set_run_status("run_exact", "completed", session_id="owned")
        db.rewind_to_message("owned", ids[0])
        assert (
            "history_binding"
            not in await (await client.get("/v1/runs/run_exact")).json()
        )


def test_in_place_rewrite_invalidates_pair_without_rebinding(db):
    agent, messages = committed_pair(db)
    ids = [agent._run_binding_rows[id(m)][2] for m in messages]
    assert db.bind_run_messages("run_rewritten", "owned", *ids)
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content='compaction carrier' WHERE id=?", (ids[0],)
        )
    )
    assert db.get_run_message_binding("run_rewritten") is None
    assert not db.bind_run_messages("run_rewritten", "owned", *ids)


def test_replace_or_fork_copy_cannot_inherit_run_identity(db):
    agent, messages = committed_pair(db)
    ids = [agent._run_binding_rows[id(m)][2] for m in messages]
    assert db.bind_run_messages("run_original", "owned", *ids)
    original = db.get_messages("owned")
    # Exercise the actual native multi-row copy primitive: unknown provenance
    # columns on exported/read rows are deliberately not inserted.
    db.append_messages_batch("other", original)
    assert all(row["api_run_id"] is None for row in db.get_messages("other"))
    db.replace_messages("owned", original)
    assert db.get_run_message_binding("run_original") is None


@pytest.mark.asyncio
async def test_existing_auth_gate_protects_binding(db):
    agent, messages = committed_pair(db)
    ids = [agent._run_binding_rows[id(m)][2] for m in messages]
    assert db.bind_run_messages("run_protected", "owned", *ids)
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "local-test-api-key"})
    )
    adapter._session_db = db
    async with await client_for(adapter) as client:
        assert (await client.get("/v1/runs/run_protected")).status == 401
        status = await (
            await client.get(
                "/v1/runs/run_protected",
                headers={"Authorization": "Bearer local-test-api-key"},
            )
        ).json()
        assert status["history_binding"]["user_message_id"] == ids[0]


def test_rewrite_between_agent_commit_and_api_confirmation_is_unavailable(db):
    agent, messages = committed_pair(db)
    ids = [agent._run_binding_rows[id(m)][2] for m in messages]
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content='changed after turn lease' WHERE id=?",
            (ids[1],),
        )
    )
    assert not db.bind_run_messages(
        "run_race", "owned", *ids, expected_user="same", expected_answer="same"
    )
    assert db.get_run_message_binding("run_race") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
async def test_non_success_after_persistence_never_confirms_pair(
    db, monkeypatch, local_agent_factory, outcome
):
    factory, calls = local_agent_factory
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = db

    def terminal_factory(**kw):
        agent = factory(**kw)
        run = agent.run_conversation

        def finish(**kwargs):
            result = run(**kwargs)
            if outcome == "failed":
                result["failed"] = True
            else:
                run_id = next(
                    key
                    for key, value in adapter._active_run_agents.items()
                    if value is agent
                )
                adapter._stopping_run_ids.add(run_id)
            return result

        agent.run_conversation = finish
        return agent

    monkeypatch.setattr(adapter, "_create_agent", terminal_factory)
    async with await client_for(adapter) as client:
        response = await client.post(
            "/v1/runs", json={"input": "same", "session_id": "owned"}
        )
        run_id = (await response.json())["run_id"]
        await adapter._active_run_tasks[run_id]
        status = await (await client.get(f"/v1/runs/{run_id}")).json()
        assert status["status"] == outcome
        assert "history_binding" not in status
        assert db.get_run_message_binding(run_id) is None
        adapter._run_statuses.clear()
        assert (await client.get(f"/v1/runs/{run_id}")).status == 404
    assert len(calls) == 1
    assert len(db.get_messages("owned")) == 2


def test_existing_store_adds_nullable_columns_without_fabricating_legacy_binding(
    tmp_path,
):
    import sqlite3

    path = tmp_path / "legacy.db"
    old = SessionDB(db_path=path)
    old.create_session("legacy", source="api_server")
    old.append_message("legacy", "user", "same")
    old.append_message("legacy", "assistant", "same")
    old.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX idx_messages_api_run")
        conn.execute("ALTER TABLE messages DROP COLUMN api_run_id")
        conn.execute("ALTER TABLE messages DROP COLUMN api_run_content_hash")
    upgraded = SessionDB(db_path=path)
    try:
        assert upgraded.get_run_message_binding("run_legacy") is None
        assert all(row["api_run_id"] is None for row in upgraded.get_messages("legacy"))
    finally:
        upgraded.close()

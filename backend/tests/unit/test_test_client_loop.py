"""The ``test_client`` loop: one event loop per test, tasks owned per request.

Each test drives a private ``_TestLoop`` with a tiny app, so the counters and
the loop are its own; no database is involved.
"""
import asyncio
import threading
import time

import pytest
from fastapi import FastAPI

from tests.fixtures import client as client_mod
from tests.fixtures.client import (
    _LoopStats,
    _PersistentLoopTestClient,
    _ReleasingTestClient,
    _REQUEST_ID,
    _TestLoop,
    make_test_client,
)

app = FastAPI()
TASKS = {}
GATES = {}


async def _forever(name):
    TASKS[name] = asyncio.current_task()
    await asyncio.sleep(3600)


@app.get("/spawn/{name}")
async def spawn(name: str):
    asyncio.get_running_loop().create_task(_forever(name))
    await asyncio.sleep(0)
    return {"loop": id(asyncio.get_running_loop())}


@app.get("/hold/{name}")
async def hold(name: str):
    """Spawn a task, then stay in flight until the test opens the gate."""
    asyncio.get_running_loop().create_task(_forever(name))
    await asyncio.sleep(0)
    started, gate = GATES[name]
    started.set()
    while not gate.is_set():
        await asyncio.sleep(0.01)
    return {"alive": not TASKS[name].done()}


@app.get("/ownerless/{name}")
async def ownerless(name: str):
    """Schedule a task from an executor thread, which carries no request id."""
    loop = asyncio.get_running_loop()
    running = asyncio.Event()

    async def body():
        TASKS[name] = asyncio.current_task()
        running.set()
        await asyncio.sleep(3600)

    def from_thread():
        asyncio.run_coroutine_threadsafe(body(), loop)

    await loop.run_in_executor(None, from_thread)
    await running.wait()
    return {}


@app.get("/stubborn/{name}")
async def stubborn(name: str):
    """Spawn a task that swallows cancellation for half a second."""

    async def body():
        TASKS[name] = asyncio.current_task()
        give_up_at = None
        while True:
            try:
                await asyncio.sleep(3600 if give_up_at is None else 0.01)
            except asyncio.CancelledError:
                if give_up_at is None:
                    give_up_at = time.monotonic() + 0.5
            if give_up_at is not None and time.monotonic() > give_up_at:
                return

    asyncio.get_running_loop().create_task(body())
    await asyncio.sleep(0)
    return {}


LOOPS = []


@app.get("/loop")
async def loop_id():
    LOOPS.append(asyncio.get_running_loop())
    return {"loop": id(asyncio.get_running_loop()), "thread": threading.get_ident()}


@pytest.fixture
def loop():
    TASKS.clear()
    GATES.clear()
    test_loop = _TestLoop(_LoopStats())
    yield test_loop
    test_loop.stop()


@pytest.fixture
def client(loop):
    return _PersistentLoopTestClient(app, test_loop=loop)


def _gate(name):
    GATES[name] = (threading.Event(), threading.Event())
    return GATES[name]


def _in_thread(fn):
    result = {}

    def run():
        try:
            result["value"] = fn()
        except BaseException as exc:  # surfaced by the test
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def test_requests_of_a_test_share_one_loop(client, loop):
    LOOPS.clear()
    client.get("/loop")
    client.get("/loop")
    assert LOOPS[0] is LOOPS[1]
    assert not LOOPS[0].is_closed()
    loop.stop()
    assert LOOPS[0].is_closed()
    assert loop.stats.snapshot()["portal_starts"] == 1


def test_task_spawned_in_a_request_is_cancelled_at_request_end(client, loop):
    assert client.get("/spawn/a").status_code == 200
    assert TASKS["a"].cancelled()
    assert loop.stats.snapshot()["owned_cancelled"] == 1
    assert loop.in_flight == 0


def test_request_id_is_reset_in_the_calling_thread(client):
    client.get("/loop")
    assert _REQUEST_ID.get() is None


def test_concurrent_request_from_another_thread_survives_first_cleanup(client, loop):
    started, gate = _gate("b")
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)

    assert client.get("/spawn/a").status_code == 200
    assert TASKS["a"].cancelled()
    assert not TASKS["b"].done()  # owned by the request still in flight

    gate.set()
    thread.join(5)
    assert "error" not in result
    assert result["value"].json() == {"alive": True}
    assert TASKS["b"].cancelled()  # and cancelled at its own end
    assert loop.stats.snapshot()["owned_cancelled"] == 2


def test_ownerless_task_is_cancelled_when_no_request_is_in_flight(client, loop):
    assert client.get("/ownerless/o").status_code == 200
    assert TASKS["o"].cancelled()
    counts = loop.stats.snapshot()
    assert counts["ownerless_cancelled"] == 1
    assert counts["owned_cancelled"] == 0


def test_ownerless_task_waits_for_the_last_request_in_flight(client, loop):
    started, gate = _gate("b")
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)

    assert client.get("/ownerless/o").status_code == 200
    assert not TASKS["o"].done()  # can't tell whose it is while b runs
    assert loop.stats.snapshot()["ownerless_skipped_in_flight"] >= 1

    gate.set()
    thread.join(5)
    assert "error" not in result
    assert TASKS["o"].cancelled()
    assert loop.stats.snapshot()["ownerless_cancelled"] == 1


def test_request_racing_a_dirty_restart_completes(client, loop, monkeypatch):
    monkeypatch.setattr(client_mod, "_CANCEL_BOUND_SECONDS", 0.1)
    started, gate = _gate("b")
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)
    first_portal = loop.portal

    # Its task outlives the bound: the loop is dirty, but b is still on it.
    assert client.get("/stubborn/s").status_code == 200
    counts = loop.stats.snapshot()
    assert counts["bound_hits"] == 1
    assert loop.portal is first_portal

    gate.set()
    thread.join(5)
    assert "error" not in result
    assert result["value"].status_code == 200
    # The last request out stopped the dirty loop; the next one gets a new one.
    assert loop.portal is None
    assert TASKS["s"].done()
    assert client.get("/loop").status_code == 200
    assert loop.portal is not first_portal
    counts = loop.stats.snapshot()
    assert counts["dirty_restarts"] == 1
    assert counts["portal_starts"] == 2


def test_stop_ends_the_loop_and_its_thread(client, loop):
    before = set(threading.enumerate())
    client.get("/loop")
    portal_threads = set(threading.enumerate()) - before
    assert portal_threads and loop.portal is not None
    loop.stop()
    assert loop.portal is None
    assert not any(t.is_alive() for t in portal_threads)
    # The next request lazily starts a fresh loop.
    assert client.get("/loop").status_code == 200
    assert loop.stats.snapshot()["portal_starts"] == 2


def test_stop_waits_for_requests_in_flight(client, loop):
    started, gate = _gate("b")
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)
    threading.Timer(0.2, gate.set).start()
    loop.stop()
    thread.join(5)
    assert "error" not in result
    assert result["value"].status_code == 200
    assert loop.portal is None


def test_the_boundary_fixture_runs_for_every_test(request, test_client):
    assert "_stop_test_client_loop" in request.fixturenames
    if isinstance(test_client, _PersistentLoopTestClient):
        assert test_client._test_loop is client_mod._LOOP


def test_the_shared_loop_is_stopped_by_stop_test_loop():
    shared = _PersistentLoopTestClient(app)
    shared.get("/loop")
    assert client_mod._LOOP.portal is not None
    client_mod.stop_test_loop()
    assert client_mod._LOOP.portal is None


def test_kill_switch_gives_one_loop_per_request(monkeypatch):
    monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", "1")
    killed = make_test_client(app)
    assert type(killed) is _ReleasingTestClient
    LOOPS.clear()
    killed.get("/loop")
    killed.get("/loop")
    assert LOOPS[0] is not LOOPS[1]
    assert all(loop.is_closed() for loop in LOOPS)
    TASKS.clear()
    killed.get("/spawn/k")
    assert TASKS["k"].done()  # its loop was closed with it


def test_default_is_the_persistent_loop(monkeypatch):
    monkeypatch.delenv("BOW_TEST_PER_REQUEST_LOOP", raising=False)
    assert type(make_test_client(app)) is _PersistentLoopTestClient


def test_loop_stats_are_written(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    written = {}
    monkeypatch.setattr(
        Path, "write_text", lambda self, text: written.__setitem__(self.name, text)
    )
    monkeypatch.setattr(Path, "mkdir", lambda self, **kw: None)
    client_mod.write_loop_stats("gwX")
    counts = json.loads(written["loop-stats-gwX.json"])
    assert set(_LoopStats.FIELDS) <= set(counts)
    assert counts["mode"] in {"per-test", "per-request"}

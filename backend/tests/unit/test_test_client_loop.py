"""The ``test_client`` loop: one event loop per test, tasks owned per request.

Most tests drive a private ``_TestLoop`` with a tiny app, so the counters and
the loop are their own; the "wiring" tests use the real shared loop to check
it is stopped at the test boundary and before an engine dispose. No database.
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


@app.get("/boom/{name}")
async def boom(name: str):
    asyncio.get_running_loop().create_task(_forever(name))
    await asyncio.sleep(0)
    raise RuntimeError("boom")


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


def _portal_threads():
    return {t for t in threading.enumerate() if t.name.startswith("asyncio-portal-")}


def test_stop_ends_the_loop_and_its_thread(client, loop):
    before = _portal_threads()
    client.get("/loop")
    started = _portal_threads() - before
    assert len(started) == 1 and loop.portal is not None
    loop.stop()
    assert loop.portal is None
    assert not any(t.is_alive() for t in started)
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


def test_boundary_timeout_cancels_the_stuck_request_and_stops(client, loop, monkeypatch):
    monkeypatch.setattr(client_mod, "_BOUNDARY_WAIT_SECONDS", 0.2)
    started, gate = _gate("b")  # never opened
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)

    t0 = time.monotonic()
    loop.stop()
    assert time.monotonic() - t0 < 3
    assert loop.portal is None
    thread.join(5)
    assert not thread.is_alive()
    assert "error" in result  # the stuck request failed in its own thread
    assert loop.in_flight == 0
    counts = loop.stats.snapshot()
    assert counts["boundary_wait_timeouts"] == 1
    assert counts["cleanup_errors"] == 0  # its cleanup found the loop gone
    # Later stops don't wait on it, and requests get a fresh loop.
    assert client.get("/loop").status_code == 200
    t0 = time.monotonic()
    loop.stop()
    assert time.monotonic() - t0 < 1


def test_bound_hit_alone_replaces_the_loop_on_the_next_request(client, loop, monkeypatch):
    monkeypatch.setattr(client_mod, "_CANCEL_BOUND_SECONDS", 0.1)
    client.get("/loop")
    first_portal = loop.portal
    assert client.get("/stubborn/s").status_code == 200
    # Nothing else in flight: the dirty loop is stopped at once.
    assert loop.portal is None
    assert TASKS["s"].done()
    assert client.get("/loop").status_code == 200
    assert loop.portal is not None and loop.portal is not first_portal
    counts = loop.stats.snapshot()
    assert counts["bound_hits"] == 1
    assert counts["dirty_restarts"] == 1
    assert counts["portal_starts"] == 2


def test_app_exception_still_cleans_up(client, loop):
    with pytest.raises(RuntimeError, match="boom"):
        client.get("/boom/x")
    assert TASKS["x"].cancelled()
    assert loop.in_flight == 0
    assert client.get("/loop").status_code == 200


def test_request_id_reset_tolerates_another_context(client):
    """A request generator finished in another context must not raise."""
    import contextvars

    factory = client._portal_factory()
    contextvars.copy_context().run(factory.__enter__)
    factory.__exit__(None, None, None)  # token belongs to the copied context
    assert _REQUEST_ID.get() is None
    assert client._test_loop.in_flight == 0


# -- wiring into the suite: these use the real shared loop -----------------------

_RECORDED = {}


def test_dispose_async_engine_stops_a_running_loop():
    from tests.conftest import _dispose_async_engine

    _PersistentLoopTestClient(app).get("/loop")
    assert client_mod._LOOP.portal is not None
    _dispose_async_engine()
    assert client_mod._LOOP.portal is None


def test_teardown_hook_stops_the_loop_before_and_after_finalizers():
    from tests.conftest import pytest_runtest_teardown

    shared = _PersistentLoopTestClient(app)
    shared.get("/loop")
    hook = pytest_runtest_teardown(None, None)
    next(hook)  # before the fixture finalizers
    assert client_mod._LOOP.portal is None
    shared.get("/loop")  # a finalizer making a request
    assert client_mod._LOOP.portal is not None
    with pytest.raises(StopIteration):
        next(hook)  # after them
    assert client_mod._LOOP.portal is None


@pytest.fixture
def request_in_teardown():
    yield
    before = _portal_threads()
    _PersistentLoopTestClient(app).get("/loop")
    _RECORDED["teardown_threads"] = _portal_threads() - before


def test_wiring_1_loop_started_by_a_test_and_its_teardown(request_in_teardown):
    before = _portal_threads()
    _PersistentLoopTestClient(app).get("/loop")
    _RECORDED["test_threads"] = _portal_threads() - before
    _RECORDED["portal"] = client_mod._LOOP.portal
    assert _RECORDED["portal"] is not None


def test_wiring_2_loop_is_gone_at_the_start_of_the_next_test():
    if "portal" not in _RECORDED:
        pytest.skip("the previous test ran in another worker")
    assert client_mod._LOOP.portal is None
    threads = _RECORDED["test_threads"] | _RECORDED["teardown_threads"]
    assert len(threads) == 2
    assert not any(t.is_alive() for t in threads)


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


@pytest.mark.parametrize("value", ["1", "true", "YES", " True "])
def test_kill_switch_values(monkeypatch, value):
    monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", value)
    assert type(make_test_client(app)) is _ReleasingTestClient


@pytest.mark.parametrize("value", [None, "", "0", "no"])
def test_default_is_the_persistent_loop(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BOW_TEST_PER_REQUEST_LOOP", raising=False)
    else:
        monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", value)
    assert type(make_test_client(app)) is _PersistentLoopTestClient


def test_loop_stats_are_written(tmp_path, client, loop):
    import json

    client.get("/spawn/a")
    client_mod.write_loop_stats(tmp_path / "out", "gw7", stats=loop.stats)
    counts = json.loads((tmp_path / "out" / "loop-stats-gw7.json").read_text())
    assert set(_LoopStats.FIELDS) <= set(counts)
    assert counts["requests"] == 1 and counts["owned_cancelled"] == 1
    assert counts["mode"] in {"per-test", "per-request"}


def test_loop_stats_worker_id():
    from types import SimpleNamespace as NS

    worker = NS(workerinput={"workerid": "gw3"}, option=NS(dist="worksteal"))
    controller = NS(option=NS(dist="worksteal"))
    plain = NS(option=NS(dist="no"))
    assert client_mod.loop_stats_worker_id(worker) == "gw3"
    assert client_mod.loop_stats_worker_id(controller) is None
    assert client_mod.loop_stats_worker_id(plain) == "main"

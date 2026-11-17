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
async def ownerless(name: str, spawn: bool = False):
    """Schedule a task from an executor thread, which carries no request id;
    with ``spawn``, also one of the request's own."""
    loop = asyncio.get_running_loop()
    if spawn:
        loop.create_task(_forever(f"{name}-owned"))
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


@app.get("/raises-on-cancel/{name}")
async def raises_on_cancel(name: str):
    """Spawn a task that turns its cancellation into another exception."""

    async def body():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise ValueError(f"{name} failed while being cancelled")

    TASKS[name] = asyncio.get_running_loop().create_task(body())
    await asyncio.sleep(0)
    return {}


LINGERING = {}


@app.get("/linger/{name}")
async def linger(name: str):
    """Spawn a task that swallows every cancellation until the test lets it
    go, so whoever stops its loop is held up."""
    release = LINGERING[name] = threading.Event()

    async def body():
        TASKS[name] = asyncio.current_task()
        while not release.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass

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


def test_a_skipped_ownerless_task_is_counted_once_per_request(client, loop):
    started, gate = _gate("b")
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)

    # Its own task makes the cleanup sweep twice; the ownerless one is
    # skipped on both sweeps but is one task.
    assert client.get("/ownerless/o", params={"spawn": True}).status_code == 200
    assert TASKS["o-owned"].cancelled()
    assert not TASKS["o"].done()
    assert loop.stats.snapshot()["ownerless_skipped_in_flight"] == 1

    gate.set()
    thread.join(5)
    assert not thread.is_alive()
    assert "error" not in result


def test_a_task_failing_while_cancelled_is_logged_and_counted_once(client, loop, caplog):
    import gc
    import logging

    caplog.set_level(logging.WARNING)
    assert client.get("/raises-on-cancel/r").status_code == 200
    task = TASKS.pop("r")
    assert task.done() and not task.cancelled()
    del task
    gc.collect()
    assert loop.stats.snapshot()["cancel_errors"] == 1
    logged = [r for r in caplog.records if r.exc_info and isinstance(r.exc_info[1], ValueError)]
    assert len(logged) == 1
    assert logged[0].levelno == logging.WARNING
    assert "r failed while being cancelled" in str(logged[0].exc_info[1])
    # Logged once with its traceback, not again as "never retrieved" by GC.
    assert "exception was never retrieved" not in caplog.text


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

    loop.stop()
    assert loop.portal is None
    assert TASKS["b"].cancelled()  # cancelled by the stop, not left to Runner
    thread.join(5)
    assert not thread.is_alive()
    assert "error" in result  # the stuck request failed in its own thread
    assert loop.in_flight == 0
    counts = loop.stats.snapshot()
    assert counts["boundary_wait_timeouts"] == 1
    assert counts["cleanup_errors"] == 0  # its cleanup found the loop gone
    # Requests get a fresh loop, and a later stop doesn't wait on the old one.
    assert client.get("/loop").status_code == 200
    loop.stop()
    assert loop.portal is None
    assert loop.stats.snapshot()["boundary_wait_timeouts"] == 1


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


def test_stop_waits_for_a_loop_another_thread_is_still_stopping(client, loop, monkeypatch):
    monkeypatch.setattr(client_mod, "_CANCEL_BOUND_SECONDS", 0.1)
    # The request hits the bound; its own thread then detaches the dirty loop
    # and is held up stopping it by the task that swallows cancellation.
    thread, result = _in_thread(lambda: client.get("/linger/l"))
    deadline = time.monotonic() + 5
    while loop.portal is None or "l" not in TASKS:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    while loop.portal is not None:  # detached by the request's thread
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert thread.is_alive() and not TASKS["l"].done()

    threading.Timer(0.3, LINGERING["l"].set).start()
    loop.stop()
    assert TASKS["l"].done()  # stop returned only once that loop was gone
    thread.join(5)
    assert not thread.is_alive()
    assert "error" not in result


def test_app_exception_still_cleans_up(client, loop):
    with pytest.raises(RuntimeError, match="boom"):
        client.get("/boom/x")
    assert TASKS["x"].cancelled()
    assert loop.in_flight == 0
    assert client.get("/loop").status_code == 200


def test_request_from_another_thread_while_one_is_in_flight_is_counted(client, loop):
    loop.test_thread = threading.get_ident()
    started, gate = _gate("b")
    # Alone in flight: not counted, even from another thread.
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)
    assert loop.stats.snapshot()["cross_thread_requests"] == 0
    # The test's own thread joining in is not counted either.
    assert client.get("/loop").status_code == 200
    assert loop.stats.snapshot()["cross_thread_requests"] == 0
    # Another thread while b is in flight is.
    other, other_result = _in_thread(lambda: client.get("/loop"))
    other.join(5)
    assert not other.is_alive() and "error" not in other_result
    gate.set()
    thread.join(5)
    assert not thread.is_alive() and "error" not in result
    assert loop.stats.snapshot()["cross_thread_requests"] == 1


def test_no_test_thread_means_no_cross_thread_check(client, loop):
    started, gate = _gate("b")
    thread, result = _in_thread(lambda: client.get("/hold/b"))
    assert started.wait(5)
    other, _ = _in_thread(lambda: client.get("/loop"))
    other.join(5)
    gate.set()
    thread.join(5)
    assert not thread.is_alive() and "error" not in result
    assert loop.stats.snapshot()["cross_thread_requests"] == 0


def test_request_id_reset_tolerates_another_context(client):
    """A request generator finished in another context must not raise."""
    import contextvars

    factory = client._portal_factory()
    contextvars.copy_context().run(factory.__enter__)
    factory.__exit__(None, None, None)  # token belongs to the copied context
    assert _REQUEST_ID.get() is None
    assert client._test_loop.in_flight == 0


# -- wiring into the suite: these use the real shared loop -----------------------

_LEFT_IN_TEARDOWN = set()


@pytest.fixture(autouse=True)
def _no_shared_loop_survives_a_test():
    """Order-independent check of the root conftest's teardown hook: every
    test of this module starts with no shared loop, and every one ends by
    starting one in a fixture teardown (after the hook's first stop), which
    the hook's second stop must end. Any test that follows another of this
    module on the same worker fails if either stop is missing."""
    assert client_mod._LOOP.portal is None
    assert not any(t.is_alive() for t in _LEFT_IN_TEARDOWN)
    yield
    before = _portal_threads()
    _PersistentLoopTestClient(app).get("/loop")
    _LEFT_IN_TEARDOWN.clear()
    _LEFT_IN_TEARDOWN.update(_portal_threads() - before)


@pytest.fixture
def loop_already_stopped_in_teardown():
    """Teardown runs after the hook's first stop: the loop the test body
    started must already be gone."""
    yield
    assert client_mod._LOOP.portal is None


def test_loop_of_the_test_body_is_stopped_before_the_finalizers(
    loop_already_stopped_in_teardown,
):
    _PersistentLoopTestClient(app).get("/loop")
    assert client_mod._LOOP.portal is not None


@pytest.fixture
def threaded_request_ended_in_teardown():
    """A fixture that ends its own threaded request in its teardown: the
    hook's first stop must leave the loop running for it."""
    TASKS.clear()
    started, gate = _gate("t")
    thread, result = _in_thread(lambda: _PersistentLoopTestClient(app).get("/hold/t"))
    assert started.wait(5)
    yield
    gate.set()
    thread.join(5)
    assert "error" not in result
    assert result["value"].json() == {"alive": True}


def test_teardown_may_end_its_own_request_in_flight(threaded_request_ended_in_teardown):
    assert client_mod._LOOP.in_flight == 1


def test_dispose_async_engine_stops_a_running_loop():
    from tests.conftest import _dispose_async_engine

    _PersistentLoopTestClient(app).get("/loop")
    assert client_mod._LOOP.portal is not None
    _dispose_async_engine()
    assert client_mod._LOOP.portal is None


def test_dispose_async_engine_disposes_even_if_the_loop_stop_fails(monkeypatch):
    from tests import conftest

    cleaned = []

    def failing_stop(if_idle=False):
        raise RuntimeError("stop failed")

    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "stop_test_loop", failing_stop)
        patch.setattr(conftest, "_stop_abandoned_sqlite_threads", lambda: cleaned.append(1))
        with pytest.raises(RuntimeError, match="stop failed"):
            conftest._dispose_async_engine()
    assert cleaned == [1]


def test_teardown_hook_runs_finalizers_even_if_its_first_stop_fails(monkeypatch):
    from tests import conftest

    calls = []

    def flaky_stop(if_idle=False):
        calls.append(if_idle)
        if if_idle:
            raise RuntimeError("first stop failed")

    # Patched only inside, since the real hook runs before monkeypatch's undo.
    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "stop_test_loop", flaky_stop)
        hook = conftest.pytest_runtest_teardown(_fake_item(), None)
        next(hook)  # the finalizers run here
        with pytest.raises(RuntimeError, match="first stop failed"):
            hook.send(None)
    assert calls == [True, False]


def _fake_item():
    from types import SimpleNamespace

    return SimpleNamespace(stash={}, get_closest_marker=lambda name: None)


def test_a_boundary_timeout_fails_the_test_it_happened_in(monkeypatch):
    from tests import conftest

    stats = _LoopStats()

    def stop_that_times_out(if_idle=False):
        if not if_idle:
            stats.add("boundary_wait_timeouts")

    item = _fake_item()
    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "_STATS", stats)
        conftest.pytest_runtest_setup(item)
        patch.setattr(client_mod, "stop_test_loop", stop_that_times_out)
        hook = conftest.pytest_runtest_teardown(item, None)
        next(hook)
        with pytest.raises(pytest.fail.Exception, match="still running"):
            hook.send(None)
        # The next test starts from the new count: it does not fail for this one.
        conftest.pytest_runtest_setup(item)
        patch.setattr(client_mod, "stop_test_loop", lambda if_idle=False: None)
        hook = conftest.pytest_runtest_teardown(item, None)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(None)


def test_teardown_hook_reports_a_failed_first_stop_when_a_finalizer_also_fails(monkeypatch):
    from tests import conftest

    def flaky_stop(if_idle=False):
        if if_idle:
            raise RuntimeError("first stop failed")

    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "stop_test_loop", flaky_stop)
        hook = conftest.pytest_runtest_teardown(_fake_item(), None)
        next(hook)
        with pytest.raises(KeyError) as raised:
            hook.throw(KeyError("finalizer failed"))
    assert "first stop failed" in "\n".join(getattr(raised.value, "__notes__", []))


@pytest.mark.parametrize(
    "problem", ["cleanup_errors", "cancel_errors", "cross_thread_requests"]
)
def test_problems_that_fail_the_test_they_happened_in(monkeypatch, problem):
    from tests import conftest

    stats = _LoopStats()
    item = _fake_item()
    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "_STATS", stats)
        conftest.pytest_runtest_setup(item)
        patch.setattr(
            client_mod, "stop_test_loop", lambda if_idle=False: stats.add(problem)
        )
        hook = conftest.pytest_runtest_teardown(item, None)
        next(hook)
        with pytest.raises(pytest.fail.Exception, match=problem):
            hook.send(None)


def test_bound_hits_and_dirty_restarts_do_not_fail_the_test(monkeypatch):
    from tests import conftest

    stats = _LoopStats()
    item = _fake_item()

    def stop(if_idle=False):
        stats.add("bound_hits")
        stats.add("dirty_restarts")

    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "_STATS", stats)
        conftest.pytest_runtest_setup(item)
        patch.setattr(client_mod, "stop_test_loop", stop)
        hook = conftest.pytest_runtest_teardown(item, None)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(None)


def test_setup_watches_the_test_thread_and_teardown_stops(monkeypatch):
    from tests import conftest

    item = _fake_item()
    conftest.pytest_runtest_setup(item)
    assert client_mod._LOOP.test_thread == threading.get_ident()
    with monkeypatch.context() as patch:
        patch.setattr(client_mod, "stop_test_loop", lambda if_idle=False: None)
        hook = conftest.pytest_runtest_teardown(item, None)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(None)
    assert client_mod._LOOP.test_thread is None


@pytest.fixture
def loop_health(monkeypatch):
    from tests import conftest

    health = {}
    monkeypatch.setattr(conftest, "_LOOP_HEALTH", health)
    return health


def _fake_session(*, controller=False, workeroutput=None):
    from types import SimpleNamespace

    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(has_plugin=lambda name: controller and name == "dsession")
    )
    if workeroutput is not None:
        config.workeroutput = workeroutput
    return SimpleNamespace(config=config)


def test_xdist_controller_reports_every_expected_worker(loop_health):
    from types import SimpleNamespace

    from tests import conftest

    specs = [SimpleNamespace(id=f"gw{i}") for i in range(3)]
    conftest.pytest_xdist_setupnodes(None, specs)
    reported = SimpleNamespace(
        gateway=SimpleNamespace(id="gw1"),
        workeroutput={conftest._LOOP_HEALTH_KEY: {}},
    )
    conftest.pytest_testnodedown(reported, None)
    conftest.pytest_sessionfinish(_fake_session(controller=True), 0)
    assert client_mod.loop_health_lines(loop_health, "shared") == [
        "test_client loop (shared) gw0: no report",
        "test_client loop (shared) gw1: no problems",
        "test_client loop (shared) gw2: no report",
    ]


def test_xdist_controller_never_reports_itself(loop_health):
    from tests import conftest

    conftest.pytest_sessionfinish(_fake_session(controller=True), 0)
    assert loop_health == {}


def test_a_run_without_workers_reports_main(loop_health):
    from tests import conftest

    conftest.pytest_sessionfinish(_fake_session(), 0)
    assert loop_health == {"main": {}}


def test_a_worker_sends_its_problems_to_the_controller(loop_health):
    from tests import conftest

    workeroutput = {}
    conftest.pytest_sessionfinish(_fake_session(workeroutput=workeroutput), 0)
    assert workeroutput == {conftest._LOOP_HEALTH_KEY: {}}
    assert loop_health == {}


def test_loop_problems_are_the_non_zero_problem_counters():
    stats = _LoopStats()
    assert client_mod.loop_problems(stats) == {}
    stats.add("owned_cancelled", 3)  # routine, not a problem
    stats.add("bound_hits", 2)
    stats.add("boundary_wait_timeouts")
    assert client_mod.loop_problems(stats) == {"bound_hits": 2, "boundary_wait_timeouts": 1}


def test_loop_health_has_one_line_per_worker_with_the_mode():
    lines = client_mod.loop_health_lines({
        "gw1": {"bound_hits": 2, "dirty_restarts": 2},
        "gw0": {},
        "gw2": None,  # the worker sent no report (it crashed)
    }, "per-request")
    assert lines == [
        "test_client loop (per-request) gw0: no problems",
        "test_client loop (per-request) gw1: bound_hits=2 dirty_restarts=2",
        "test_client loop (per-request) gw2: no report",
    ]


@pytest.mark.parametrize("value, mode", [("1", "per-request"), ("", "shared"), ("0", "shared")])
def test_loop_mode_names_the_kill_switch_setting(monkeypatch, value, mode):
    monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", value)
    assert client_mod.loop_mode() == mode


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


@pytest.fixture
def released(monkeypatch):
    """Record each release of pooled connections (none open here)."""
    from app.settings import database

    calls = []
    real = database.release_test_connections

    async def recording():
        calls.append(asyncio.get_running_loop())
        await real()

    monkeypatch.setattr(database, "release_test_connections", recording)
    return calls


@pytest.mark.per_request_loop
def test_marked_test_gets_a_loop_and_a_release_per_request(released):
    shared = _PersistentLoopTestClient(app)  # what the session test_client is
    LOOPS.clear()
    shared.get("/loop")
    shared.get("/loop")
    assert LOOPS[0] is not LOOPS[1]
    assert all(loop.is_closed() for loop in LOOPS)
    assert released == LOOPS  # released on each request's own loop
    assert client_mod._LOOP.portal is None  # the shared loop is never started
    TASKS.clear()
    shared.get("/spawn/k")
    assert TASKS["k"].done()  # its loop was closed with it


def test_unmarked_test_shares_one_loop_across_requests(released):
    shared = _PersistentLoopTestClient(app)
    LOOPS.clear()
    shared.get("/loop")
    shared.get("/loop")
    assert LOOPS[0] is LOOPS[1]
    assert not LOOPS[0].is_closed()
    assert released == []  # released when the loop stops, not per request
    assert client_mod._LOOP.portal is not None


@pytest.mark.parametrize("value", ["1", "true", "YES", " True "])
def test_kill_switch_values(monkeypatch, value):
    monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", value)
    assert type(make_test_client(app)) is _ReleasingTestClient


@pytest.mark.parametrize("value", [None, "", "0", "no", " False "])
def test_default_is_the_persistent_loop(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BOW_TEST_PER_REQUEST_LOOP", raising=False)
    else:
        monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", value)
    assert type(make_test_client(app)) is _PersistentLoopTestClient


@pytest.mark.parametrize("value", ["2", "on", "per-request", "ture"])
def test_unrecognised_kill_switch_value_is_an_error(monkeypatch, value):
    monkeypatch.setenv("BOW_TEST_PER_REQUEST_LOOP", value)
    with pytest.raises(ValueError, match="BOW_TEST_PER_REQUEST_LOOP"):
        make_test_client(app)

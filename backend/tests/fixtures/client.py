"""The ``test_client`` fixture and the event loop its requests run on.

By default every request of a test runs on one event loop (a blocking portal)
that starts at the test's first request and is stopped at the test boundary
(see ``stop_test_loop``), so within a test pooled DB connections are reused
instead of opened per request. Each request still ends the way it did when it
had a loop of its own: the tasks it spawned are cancelled when it returns.

``BOW_TEST_PER_REQUEST_LOOP=1`` (or true/yes) restores the old behaviour, one
loop per request (``_ReleasingTestClient``), for A/B runs and debugging.
``BOW_TEST_LOOP_STATS=<dir>`` writes each process's loop counters there at
session end (``loop-stats-<worker>.json``).
"""
import asyncio
import contextlib
import contextvars
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from main import app


class _ReleasingTestClient(TestClient):
    """Each request runs on its own event loop (one blocking portal per
    request). Close the pooled DB connections that loop opened while it is
    still running — asyncpg can only close a connection on its own loop —
    so they neither leak nor get offered to the next request's loop."""

    @contextlib.contextmanager
    def _portal_factory(self):
        with super()._portal_factory() as portal:
            try:
                yield portal
            finally:
                if self.portal is None:
                    from app.settings.database import release_test_connections

                    portal.call(release_test_connections)


# The request a task belongs to. Set in the calling thread for the duration of
# a request; anyio's portal schedules its call with call_soon_threadsafe,
# which copies the calling thread's context, and tasks inherit their parent's
# context, so everything a request spawns carries its id. Work handed to the
# loop from threads that did not copy the context (run_in_executor callbacks,
# raw threads) carries none: "ownerless".
_REQUEST_ID = contextvars.ContextVar("bow_test_request_id", default=None)

# How long a request's cleanup keeps cancelling before it gives up and marks
# the loop dirty (stopped and replaced once no request is in flight).
_CANCEL_BOUND_SECONDS = 5.0
# How long the test boundary waits for requests still running in other threads
# before it cancels every task left on the loop, and how long it then waits for
# those cancellations. Neither bounds the loop's final stop (asyncio.Runner
# shutdown): a task that swallows cancellation, or executor work, still delays
# it, exactly as the per-request loop's shutdown always has.
_BOUNDARY_WAIT_SECONDS = 60.0
_BOUNDARY_CANCEL_SECONDS = 5.0


class _LoopStats:
    """Per-process counters, written at session end for measurement."""

    FIELDS = (
        "portal_starts",
        "portal_stops",
        "requests",
        "owned_cancelled",
        "ownerless_cancelled",
        "ownerless_skipped_in_flight",
        "bound_hits",
        "dirty_restarts",
        "cleanup_errors",
        "boundary_wait_timeouts",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.counts = dict.fromkeys(self.FIELDS, 0)

    def add(self, field, n=1):
        if n:
            with self._lock:
                self.counts[field] += n

    def snapshot(self):
        with self._lock:
            return dict(self.counts)


class _TestLoop:
    """One lazily started blocking portal shared by concurrent requests.

    A single condition guards lazy start, the in-flight requests (tracked per
    portal), stop and restart. Nothing holds it while waiting on a loop that a
    request's cleanup could be running on, so the cleanup (which takes it
    briefly, on the loop thread) cannot deadlock against a stop.
    """

    def __init__(self, stats):
        self.stats = stats
        self._cond = threading.Condition()
        self._cm = None
        self._portal = None
        self._in_flight = {}  # request id -> portal it runs on
        self._dirty = False
        self._ids = itertools.count(1)

    # -- introspection (tests) ------------------------------------------------
    @property
    def portal(self):
        return self._portal

    @property
    def in_flight(self):
        with self._cond:
            return len(self._in_flight)

    def _busy_locked(self, portal):
        return any(p is portal for p in self._in_flight.values())

    # -- lifecycle --------------------------------------------------------------
    def _start_locked(self):
        # SelectorEventLoop explicitly: neither a uvloop policy a uvicorn test
        # installed nor any other policy decides the shared loop's type.
        cm = anyio.from_thread.start_blocking_portal(
            "asyncio", backend_options={"loop_factory": asyncio.SelectorEventLoop}
        )
        portal = cm.__enter__()
        # The portal's own tasks (its main task), never cancelled by a sweep.
        portal._bow_own_tasks = frozenset(portal.call(_all_tasks_now))
        self._cm, self._portal, self._dirty = cm, portal, False
        self.stats.add("portal_starts")

    def _detach_locked(self):
        cm, portal = self._cm, self._portal
        self._cm = self._portal = None
        self._dirty = False
        return cm, portal

    def _stop_detached(self, cm, portal):
        """Release the loop's pooled connections on the loop, then stop it the
        normal way (asyncio.Runner cancels and awaits what is left, closes async
        generators and joins the default executor)."""
        if cm is None:
            return
        from app.settings.database import release_test_connections

        try:
            portal.call(release_test_connections)
        except Exception as exc:  # the loop is going away regardless
            self.stats.add("cleanup_errors")
            print(f"Warning: release_test_connections failed: {exc!r}", file=sys.stderr)
        cm.__exit__(None, None, None)
        self.stats.add("portal_stops")

    def stop(self):
        """Test boundary: wait (at most _BOUNDARY_WAIT_SECONDS) for requests in
        flight on the loop, then release its connections and stop it. On a
        timeout every task left on the loop is cancelled first (waiting at most
        _BOUNDARY_CANCEL_SECONDS); the stuck request then fails in its own
        thread and its cleanup finds the loop gone."""
        with self._cond:
            portal = self._portal
            if portal is None:
                return
            timed_out = not self._cond.wait_for(
                lambda: self._portal is not portal or not self._busy_locked(portal),
                timeout=_BOUNDARY_WAIT_SECONDS,
            )
            if self._portal is not portal:  # stopped meanwhile (dirty restart)
                return
            detached = self._detach_locked()
        if timed_out:
            self.stats.add("boundary_wait_timeouts")
            print(
                "Warning: test_client request(s) still running at the test "
                "boundary; cancelling every task on their loop and stopping it",
                file=sys.stderr,
            )
            try:
                portal.call(_cancel_all_but_own, portal)
            except Exception as exc:
                self.stats.add("cleanup_errors")
                print(f"Warning: boundary cancel failed: {exc!r}", file=sys.stderr)
        self._stop_detached(*detached)

    # -- per request ----------------------------------------------------------------
    def enter(self):
        """Register a request; return (request id, portal). A dirty loop is
        replaced first, but only while no request is in flight on it."""
        while True:
            with self._cond:
                if not (self._dirty and not self._busy_locked(self._portal)):
                    if self._portal is None:
                        self._start_locked()
                    rid = next(self._ids)
                    self._in_flight[rid] = self._portal
                    self.stats.add("requests")
                    return rid, self._portal
                detached = self._detach_locked()
                self.stats.add("dirty_restarts")
            self._stop_detached(*detached)

    def exit(self, rid, portal):
        """End a request: cancel what it left behind, then deregister it. If
        its loop was already detached (a boundary timeout stopped it), there is
        nothing left to clean."""
        bound_hit = False
        try:
            with self._cond:
                current = portal is self._portal
            if current:
                try:
                    bound_hit = portal.call(self._cleanup, rid, portal)
                except Exception as exc:
                    with self._cond:
                        current = portal is self._portal
                    if current:  # not just stopped under us
                        self.stats.add("cleanup_errors")
                        print(
                            f"Warning: test_client request cleanup failed: {exc!r}",
                            file=sys.stderr,
                        )
                        bound_hit = True
        finally:
            detached = None
            with self._cond:
                del self._in_flight[rid]
                if portal is self._portal:
                    if bound_hit:
                        self._dirty = True
                    if self._dirty and not self._busy_locked(portal):
                        detached = self._detach_locked()
                        self.stats.add("dirty_restarts")
                self._cond.notify_all()
            if detached is not None:
                self._stop_detached(*detached)

    async def _cleanup(self, rid, portal):
        """Runs on the loop. Cancel the request's own tasks, and, when no other
        request is in flight on this loop, every task no request owns, except
        the portal's own and this one. Repeat until nothing is left (tasks
        spawned while cancelling, shielded ones, ones that catch
        CancelledError), within the bound. Returns True if the bound was hit."""
        me = asyncio.current_task()
        own_tasks = portal._bow_own_tasks
        deadline = time.monotonic() + _CANCEL_BOUND_SECONDS
        while True:
            with self._cond:
                others = {
                    other
                    for other, p in self._in_flight.items()
                    if other != rid and p is portal
                }
                targets = []
                owned = ownerless = skipped = 0
                for task in asyncio.all_tasks():
                    if task is me or task in own_tasks or task.done():
                        continue
                    owner = task.get_context().get(_REQUEST_ID)
                    if owner == rid:
                        owned += 1
                    elif owner in others:
                        continue
                    elif others:
                        skipped += 1
                        continue
                    else:
                        ownerless += 1
                    targets.append(task)
                # Cancel while holding the lock, so no request can start (and
                # spawn work this sweep would wrongly take for ownerless) in
                # between the decision and the cancel.
                for task in targets:
                    task.cancel()
            self.stats.add("owned_cancelled", owned)
            self.stats.add("ownerless_cancelled", ownerless)
            self.stats.add("ownerless_skipped_in_flight", skipped)
            if not targets:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _, pending = await asyncio.wait(targets, timeout=remaining)
            if pending:
                break
        self.stats.add("bound_hits")
        print(
            f"Warning: test_client request {rid} left tasks that did not settle "
            f"within {_CANCEL_BOUND_SECONDS}s; its loop will be replaced",
            file=sys.stderr,
        )
        return True


def _all_tasks_now():
    return set(asyncio.all_tasks())


async def _cancel_all_but_own(portal):
    me = asyncio.current_task()
    tasks = asyncio.all_tasks() - portal._bow_own_tasks - {me}
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.wait(tasks, timeout=_BOUNDARY_CANCEL_SECONDS)


_STATS = _LoopStats()
_LOOP = _TestLoop(_STATS)


def stop_test_loop():
    """Stop the shared test loop if one is running (test teardown, before an
    engine dispose, session end). Cheap when none is."""
    if _LOOP.portal is not None:
        _LOOP.stop()


def loop_stats_worker_id(config):
    """The name this process's stats file goes under, or None for an xdist
    controller (it runs no tests)."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        return workerinput.get("workerid", "gw")
    if getattr(config.option, "dist", "no") != "no":
        return None
    return "main"


def write_loop_stats(out_dir, worker_id, stats=None):
    """Write the loop counters to <out_dir>/loop-stats-<worker>.json."""
    counts = (stats or _STATS).snapshot()
    counts["mode"] = "per-request" if per_request_loop_enabled() else "per-test"
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"loop-stats-{worker_id}.json").write_text(json.dumps(counts, indent=2))
    except OSError as exc:
        print(f"Warning: could not write loop stats: {exc!r}", file=sys.stderr)
    if counts["bound_hits"] or counts["cleanup_errors"] or counts["boundary_wait_timeouts"]:
        print(f"test_client loop stats ({worker_id}): {json.dumps(counts)}", file=sys.stderr)


def per_request_loop_enabled():
    value = os.environ.get("BOW_TEST_PER_REQUEST_LOOP", "")
    return value.strip().lower() in {"1", "true", "yes"}


class _PersistentLoopTestClient(TestClient):
    """Requests run on the shared per-test loop (``_LOOP``). Each request owns
    the tasks it spawns (``_REQUEST_ID``), and they are cancelled when it ends.
    Pooled connections are released when the loop stops, not per request."""

    def __init__(self, *args, test_loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_loop = test_loop if test_loop is not None else _LOOP

    @contextlib.contextmanager
    def _portal_factory(self):
        if self.portal is not None:  # inside `with client:` (lifespan)
            yield self.portal
            return
        rid, portal = self._test_loop.enter()
        token = _REQUEST_ID.set(rid)
        try:
            yield portal
        finally:
            try:
                self._test_loop.exit(rid, portal)
            finally:
                try:
                    _REQUEST_ID.reset(token)
                except ValueError:
                    # The generator was finished in another context than the
                    # one it started in (e.g. closed by GC): clear the id there
                    # rather than fail; the original context is not reachable.
                    _REQUEST_ID.set(None)


def make_test_client(asgi_app=app):
    if per_request_loop_enabled():
        return _ReleasingTestClient(asgi_app)
    return _PersistentLoopTestClient(asgi_app)


@pytest.fixture(scope="session")
def test_client():
    return make_test_client()

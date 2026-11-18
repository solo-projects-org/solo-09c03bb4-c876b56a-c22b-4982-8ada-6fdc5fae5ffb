"""Incremental map-publication behaviour.

The recompute pipeline must scale with the number of coordinates affected by
unpublished batches, not with the wafer size, and an up-to-date wafer must
cost no cell work at all.  These tests pin both the invariants that make the
incremental path observable (per-cell ``computed_seq`` markers, affected-set
derivation) and its exact equivalence with a from-scratch rebuild.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.main import Settings, create_app
from app.scheduler import RecomputeScheduler
from app.schemas import BatchRequest
from app.service import WaferService
from tests.conftest import batch, cell, wait_until


def as_request(payload: dict) -> BatchRequest:
    return BatchRequest(**payload)


def raw_map_cells(db_path: str, wafer_id: str) -> dict[tuple[int, int], sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM map_cells WHERE wafer_id = ?", (wafer_id,)
        ).fetchall()
    finally:
        conn.close()
    return {(row["full_row"], row["full_col"]): row for row in rows}


def brute_force_winners(db_path: str, wafer_id: str) -> dict[tuple[int, int], sqlite3.Row]:
    """What a full rebuild would publish, derived straight from the log."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT o.full_row, o.full_col, o.verdict, o.retest_no, o.machine_id,
                   o.batch_id, b.seq AS batch_seq, o.local_row, o.local_col
              FROM observations AS o
              JOIN batches AS b ON b.batch_id = o.batch_id
             WHERE o.wafer_id = ?
            """,
            (wafer_id,),
        ).fetchall()
    finally:
        conn.close()
    winners: dict[tuple[int, int], sqlite3.Row] = {}
    for row in rows:
        key = (row["full_row"], row["full_col"])
        current = winners.get(key)
        if current is None or (row["retest_no"], row["batch_seq"]) > (
            current["retest_no"],
            current["batch_seq"],
        ):
            winners[key] = row
    return winners


def assert_map_matches_full_rebuild(service: WaferService, db_path: str, wafer_id: str) -> None:
    """The incrementally published map must equal a from-scratch rebuild."""
    served = service.get_map(wafer_id)
    assert served["status"] == "ready"
    assert served["accepted_seq"] == served["map_seq"]
    expected = brute_force_winners(db_path, wafer_id)
    actual = {(c["row"], c["col"]): c for c in served["cells"]}
    assert set(actual) == set(expected)
    for pos, payload in actual.items():
        want = expected[pos]
        assert payload["verdict"] == want["verdict"]
        assert payload["retest_no"] == want["retest_no"]
        assert payload["source"]["machine_id"] == want["machine_id"]
        assert payload["source"]["batch_id"] == want["batch_id"]
        assert payload["source"]["batch_seq"] == want["batch_seq"]
        assert payload["source"]["local_row"] == want["local_row"]
        assert payload["source"]["local_col"] == want["local_col"]
        # Every served source belongs to an already-published batch.
        assert payload["source"]["batch_seq"] <= served["map_seq"]


# --------------------------------------------------------------- write set
def test_small_batch_rewrites_only_affected_cells(db_path: str) -> None:
    service = WaferService(db_path)
    grid = [[cell("PASS", 0) for _ in range(6)] for _ in range(6)]
    service.accept_batch("W", as_request(batch(machine="M1", rows=6, cols=6, grid=grid)))
    service.recompute("W")
    before = raw_map_cells(db_path, "W")
    assert len(before) == 36
    assert {row["computed_seq"] for row in before.values()} == {1}

    # A two-cell update: one higher retest on a measured cell, one new cell
    # that was previously untested (null in the first grid is not needed --
    # (5,5) gets a higher retest, (0,5) is re-reported identically).
    service.accept_batch(
        "W",
        as_request(
            batch(
                machine="M2",
                origin=(5, 4),
                rows=6,
                cols=6,
                grid=[[cell("FAIL", 1), cell("PASS", 0)]],
            )
        ),
    )
    service.recompute("W")

    after = raw_map_cells(db_path, "W")
    assert len(after) == 36  # no cell dropped, none added
    rewritten = {pos for pos, row in after.items() if row["computed_seq"] == 2}
    assert rewritten == {(5, 4), (5, 5)}
    untouched = {pos for pos, row in after.items() if row["computed_seq"] == 1}
    assert len(untouched) == 34
    # The 34 untouched rows are byte-identical to before.
    for pos in untouched:
        assert dict(after[pos]) == dict(before[pos])
    # Higher retest wins on (5,4); identical re-report keeps verdict but the
    # source moves to the newest upload.
    assert after[(5, 4)]["verdict"] == "FAIL"
    assert after[(5, 4)]["source_retest_no"] == 1
    assert after[(5, 5)]["source_machine_id"] == "M2"
    assert_map_matches_full_rebuild(service, db_path, "W")


def test_new_measurements_on_blank_positions_enter_the_map(db_path: str) -> None:
    service = WaferService(db_path)
    service.accept_batch(
        "W", as_request(batch(machine="M1", grid=[[cell("PASS", 0), None]]))
    )
    service.recompute("W")
    assert service.get_map("W")["tested_cells"] == 1

    service.accept_batch(
        "W", as_request(batch(machine="M2", grid=[[None, cell("FAIL", 0)]]))
    )
    service.recompute("W")
    cells = raw_map_cells(db_path, "W")
    assert set(cells) == {(0, 0), (0, 1)}
    assert cells[(0, 1)]["verdict"] == "FAIL"
    assert cells[(0, 1)]["computed_seq"] == 2
    assert cells[(0, 0)]["computed_seq"] == 1  # untouched by the delta


def test_lower_retest_never_displaces_higher(db_path: str) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 5)]])))
    service.recompute("W")
    service.accept_batch("W", as_request(batch(machine="M2", grid=[[cell("FAIL", 2)]])))
    service.recompute("W")
    cells = raw_map_cells(db_path, "W")
    assert cells[(0, 0)]["verdict"] == "PASS"
    assert cells[(0, 0)]["source_retest_no"] == 5
    assert cells[(0, 0)]["source_machine_id"] == "M1"
    # The losing upload still leaves an audit trail: the cell was re-evaluated.
    assert cells[(0, 0)]["computed_seq"] == 2


def test_rereport_same_reading_recomputes_source_only(db_path: str) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 1)]])))
    service.recompute("W")
    second = service.accept_batch(
        "W", as_request(batch(machine="M2", grid=[[cell("PASS", 1)]]))
    )
    service.recompute("W")
    stored = raw_map_cells(db_path, "W")[(0, 0)]
    assert stored["source_machine_id"] == "M2"
    assert stored["source_batch_id"] == second["batch_id"]
    assert stored["source_batch_seq"] == 2
    assert stored["computed_seq"] == 2


def test_affected_coordinates_are_derived_from_unpublished_batches(db_path: str) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")
    service.accept_batch(
        "W", as_request(batch(machine="M2", origin=(1, 1), grid=[[cell("PASS", 1)]]))
    )
    service.accept_batch(
        "W", as_request(batch(machine="M3", origin=(2, 2), grid=[[cell("FAIL", 0)]]))
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        affected = service._affected_coordinates(conn, "W", 1, 3)
    finally:
        conn.close()
    assert affected == [(1, 1), (2, 2)]
    # One recompute round publishes both pending batches at once.
    service.recompute("W")
    cells = raw_map_cells(db_path, "W")
    assert {pos for pos, row in cells.items() if row["computed_seq"] == 3} == {
        (1, 1),
        (2, 2),
    }
    assert cells[(0, 0)]["computed_seq"] == 1


# ------------------------------------------------------------- up to date
def test_recompute_on_current_map_does_no_cell_work(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")
    assert service.get_map("W")["status"] == "ready"

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("cell work performed on an up-to-date wafer")

    monkeypatch.setattr(WaferService, "_affected_coordinates", forbidden)
    monkeypatch.setattr(WaferService, "_winning_observations", forbidden)
    result = service.recompute("W")
    assert result["status"] == "ready"
    assert result["scheduled"] is False
    # Nothing was republished either.
    assert {row["computed_seq"] for row in raw_map_cells(db_path, "W").values()} == {1}


def test_snapshot_of_current_map_is_empty(db_path: str) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")
    snapshot = service._build_snapshot("W")
    assert snapshot.up_to_date is True
    assert snapshot.cells == []
    assert snapshot.seq == snapshot.committed_seq == 1


# -------------------------------------------- consecutive small-batch stream
def test_consecutive_small_batches_interleaved_with_recompute(db_path: str) -> None:
    """A stream of small uploads, some landing mid-recompute, must converge to
    exactly what a full rebuild of the observation log would produce."""
    service = WaferService(db_path, recompute_delay_seconds=0.05)
    scheduler = RecomputeScheduler(service, workers=1)
    try:
        # Deterministic stream: verdict depends only on the retest number, so
        # no upload can conflict with an earlier one at the same position.
        # Origins stay in [1, 4] so every rotation of the 2x2 grid lands
        # inside the 6x6 wafer.
        uploads = []
        for i in range(12):
            origin = (1 + (i % 4), 1 + ((i * 2) % 4))
            retest = i % 3
            verdict = "PASS" if retest % 2 == 0 else "FAIL"
            grid = [
                [cell(verdict, retest), None],
                [cell(verdict, retest), cell("PASS", 0)],
            ]
            uploads.append(
                batch(
                    machine=f"M{i % 4}",
                    origin=origin,
                    rotation=(i % 4) * 90,
                    rows=6,
                    cols=6,
                    grid=grid,
                )
            )
        for i, payload in enumerate(uploads):
            service.accept_batch("W", as_request(payload))
            scheduler.submit("W")
            if i % 3 == 0:
                time.sleep(0.02)  # let some recompute rounds interleave
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = service.get_wafer_state("W")
            if state["committed_seq"] == state["accepted_seq"] == 12:
                break
            time.sleep(0.02)
        else:  # pragma: no cover
            raise AssertionError("recompute never caught up with the upload stream")
    finally:
        scheduler.shutdown()
    assert_map_matches_full_rebuild(service, db_path, "W")
    # The final publish carried the last batch's sequence.
    assert service.get_map("W")["map_seq"] == 12


def test_uploads_stay_accepted_while_incremental_recompute_runs(db_path: str) -> None:
    service = WaferService(db_path, recompute_delay_seconds=0.3)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")

    scheduler = RecomputeScheduler(service, workers=1)
    try:
        service.accept_batch(
            "W", as_request(batch(machine="M2", origin=(1, 1), grid=[[cell("PASS", 1)]]))
        )
        scheduler.submit("W")
        time.sleep(0.05)  # the delta computation is now sleeping before publish
        started = time.monotonic()
        service.accept_batch(
            "W", as_request(batch(machine="M3", origin=(2, 2), grid=[[cell("FAIL", 0)]]))
        )
        assert time.monotonic() - started < 0.2
        scheduler.submit("W")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = service.get_wafer_state("W")
            if state["committed_seq"] == state["accepted_seq"] == 3:
                break
            time.sleep(0.02)
        else:  # pragma: no cover
            raise AssertionError("recompute never caught up")
    finally:
        scheduler.shutdown()
    assert_map_matches_full_rebuild(service, db_path, "W")


# ------------------------------------------------------------------ recovery
def test_restart_with_pending_batches_then_more_uploads(db_path: str) -> None:
    writer = WaferService(db_path)
    writer.accept_batch(
        "W", as_request(batch(machine="M1", grid=[[cell("PASS", 0), cell("FAIL", 0)]]))
    )
    writer.recompute("W")
    # Two batches accepted but never published before the "crash".
    writer.accept_batch(
        "W", as_request(batch(machine="M2", origin=(1, 0), grid=[[cell("PASS", 2)]]))
    )
    writer.accept_batch(
        "W", as_request(batch(machine="M3", origin=(1, 1), grid=[[cell("FAIL", 1)]]))
    )

    revived = WaferService(db_path)
    pending = revived.get_map("W")
    assert pending["status"] == "pending_update"
    assert pending["map_seq"] == 1 and pending["accepted_seq"] == 3
    assert pending["tested_cells"] == 2  # old map still served
    assert revived.pending_wafer_ids() == ["W"]

    # More uploads land after the restart, before any recovery recompute.
    revived.accept_batch(
        "W", as_request(batch(machine="M4", origin=(0, 1), grid=[[cell("PASS", 3)]]))
    )
    revived.recompute("W")
    final = revived.get_map("W")
    assert final["status"] == "ready"
    assert final["map_seq"] == 4
    assert_map_matches_full_rebuild(revived, db_path, "W")
    # Cells from the first publish that nothing touched keep their marker.
    cells = raw_map_cells(db_path, "W")
    assert cells[(0, 0)]["computed_seq"] == 1
    assert cells[(0, 1)]["computed_seq"] == 4  # retest 3 from M4 won


def test_failed_delta_publish_keeps_old_map_and_recovers(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")
    service.accept_batch(
        "W", as_request(batch(machine="M2", origin=(1, 1), grid=[[cell("FAIL", 1)]]))
    )

    attempts = {"count": 0}
    real_publish = service._publish

    def flaky_publish(snapshot):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated publish crash")
        return real_publish(snapshot)

    monkeypatch.setattr(service, "_publish", flaky_publish)
    with pytest.raises(RuntimeError):
        service.recompute("W")
    broken = service.get_map("W")
    assert broken["status"] == "pending_update"
    assert broken["recompute_failed"] is True
    assert broken["map_seq"] == 1
    assert broken["tested_cells"] == 1  # old map preserved

    # A restart with a healthy service resumes purely from persisted state.
    revived = WaferService(db_path)
    assert revived.pending_wafer_ids() == ["W"]
    revived.recompute("W")
    healed = revived.get_map("W")
    assert healed["status"] == "ready"
    assert healed["recompute_failed"] is False
    assert healed["map_seq"] == 2
    assert_map_matches_full_rebuild(revived, db_path, "W")


def test_app_restart_recovers_pending_delta(restart_db_path: str) -> None:
    """Lifespan recovery publishes the delta left behind by the crash."""
    db = restart_db_path
    service = WaferService(db)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")
    service.accept_batch(
        "W", as_request(batch(machine="M2", origin=(1, 1), grid=[[cell("FAIL", 1)]]))
    )

    app = create_app(Settings(db_path=db, scheduler_workers=1))
    with TestClient(app) as client:
        state = wait_until(client, "W", lambda s: s["status"] == "ready")
        assert state["map_seq"] == 2
        body = client.get("/wafers/W/map").json()
        by_pos = {(c["row"], c["col"]): c for c in body["cells"]}
        assert by_pos[(0, 0)]["verdict"] == "PASS"
        assert by_pos[(1, 1)]["verdict"] == "FAIL"


@pytest.fixture
def restart_db_path(tmp_path) -> str:
    from app.db import init_db

    path = str(tmp_path / "restart.db")
    init_db(path)
    return path


# ------------------------------------------------- publication race control
def test_stale_delta_cannot_overwrite_newer_batch(db_path: str) -> None:
    service = WaferService(db_path)
    service.accept_batch("W", as_request(batch(machine="M1", grid=[[cell("PASS", 0)]])))
    service.recompute("W")
    stale = service._build_snapshot("W")  # up-to-date: no cells
    assert stale.up_to_date

    service.accept_batch("W", as_request(batch(machine="M2", grid=[[cell("FAIL", 1)]])))
    delta = service._build_snapshot("W")
    assert [ (c["full_row"], c["full_col"]) for c in delta.cells ] == [(0, 0)]

    # A newer acceptance makes the in-flight delta stale before it publishes.
    service.accept_batch("W", as_request(batch(machine="M3", grid=[[cell("PASS", 2)]])))
    assert service._publish(delta) is False
    state = service.get_wafer_state("W")
    assert state["committed_seq"] == 1  # nothing half-written
    service.recompute("W")
    final = raw_map_cells(db_path, "W")[(0, 0)]
    assert final["source_retest_no"] == 2
    assert final["source_machine_id"] == "M3"
    assert final["computed_seq"] == 3


def test_concurrent_recompute_workers_converge(db_path: str) -> None:
    """Two workers racing on one wafer must not corrupt the published map."""
    service = WaferService(db_path, recompute_delay_seconds=0.02)
    for i in range(4):
        service.accept_batch(
            "W",
            as_request(
                batch(machine=f"M{i}", origin=(i, 0), grid=[[cell("PASS", i)]])
            ),
        )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            service.recompute("W")
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    # At least one of them may have to catch up afterwards.
    while service.get_wafer_state("W")["committed_seq"] < 4:
        service.recompute("W")
    assert_map_matches_full_rebuild(service, db_path, "W")

"""End-to-end tests for the whole slice, offline.

These are the tests that stand behind the project's central claim: a fault
experiment is recorded once and reproduces afterwards from committed files, with
no endpoint, no key and no network. Everything here runs against the scripted stub
and a temporary store.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import evals.study as study_module
from evals.study import StudyConfig, run_study
from misfeed.store import Store


@pytest.fixture
def config(tmp_path: Path) -> StudyConfig:
    return StudyConfig(
        store_root=tmp_path / "cassettes",
        report_path=tmp_path / "report.json",
        faults=("empty_success", "partial_list"),
    )


class TestStudyRuns:
    async def test_produces_a_report_marked_synthetic(self, config: StudyConfig) -> None:
        summary = await run_study(config)
        assert summary["synthetic"] is True
        assert config.report_path.exists()
        assert config.report_path.with_suffix(".md").exists()
        assert "not a finding" in config.report_path.with_suffix(".md").read_text()

    async def test_clean_runs_pass_the_precondition(self, config: StudyConfig) -> None:
        # If this fails, nothing downstream means anything.
        summary = await run_study(config)
        assert summary["results"]["precondition_failed"] == []
        assert summary["results"]["totals"]["withheld_precondition"] == 0
        assert summary["results"]["totals"]["clean_runs"] == 4  # 2 policies x 2 tasks

    async def test_harness_separates_a_careless_agent_from_a_careful_one(
        self, config: StudyConfig
    ) -> None:
        # The gap is scripted, not observed. What is being asserted is that the
        # harness detects a difference in tool-result handling at all.
        summary = await run_study(config)
        naive = summary["results"]["by_variant"]["stub/naive / baseline"]
        careful = summary["results"]["by_variant"]["stub/careful / baseline"]
        assert naive["silent_corruption_rate"] is not None
        assert careful["silent_corruption_rate"] is not None
        assert naive["silent_corruption_rate"] > careful["silent_corruption_rate"]

    async def test_plausible_truncation_defeats_validation(self, config: StudyConfig) -> None:
        # partial_list leaves a well-formed, shorter payload. Checking that a
        # result is present and well-shaped cannot catch it, so even the careful
        # policy returns a confident wrong number.
        summary = await run_study(config)
        silent = [
            run
            for run in summary["results"]["runs"]
            if run["model"] == "stub/careful" and run.get("outcome") == "silent_corruption"
        ]
        assert silent
        assert {run["fault"] for run in silent} == {"partial_list"}

    async def test_inapplicable_faults_are_reported_not_scored(self, config: StudyConfig) -> None:
        # partial_list against the customer lookup has no list to shorten.
        summary = await run_study(config)
        assert summary["results"]["totals"]["skipped_inapplicable"] > 0
        reasons = {run["skipped"] for run in summary["results"]["runs"] if run.get("skipped")}
        assert "no list in content" in reasons

    async def test_every_scored_run_names_a_replayable_trace(self, config: StudyConfig) -> None:
        summary = await run_study(config)
        store = Store(config.store_root)
        scored = [r for r in summary["results"]["runs"] if r.get("outcome")]
        assert scored
        for run in scored:
            assert store.resolve(run["trace_id"])  # resolves to a full entry list


class TestReproducibility:
    async def test_results_are_byte_identical_on_rerun(self, config: StudyConfig) -> None:
        # The reproducibility guarantee. `provenance` is excluded by design: a
        # first run makes live calls and a replay makes none, so including it
        # would force the guarantee to be hedged.
        first = await run_study(config)
        second = await run_study(config)
        assert json.dumps(first["results"], sort_keys=True) == json.dumps(
            second["results"], sort_keys=True
        )
        assert first["synthetic"] == second["synthetic"]

    async def test_provenance_records_that_the_replay_was_free(self, config: StudyConfig) -> None:
        first = await run_study(config)
        assert first["provenance"]["live_calls"] > 0
        assert first["provenance"]["live_calls_by_trace"]
        second = await run_study(config)
        assert second["provenance"]["live_calls"] == 0
        assert second["provenance"]["live_calls_by_trace"] == {}

    async def test_rerun_makes_no_live_calls_at_all(self, config: StudyConfig) -> None:
        # The property that lets someone else reproduce a published number with no
        # endpoint: once recorded, every request is served from a cassette.
        first = await run_study(config)
        assert first["provenance"]["live_calls"] > 0

        second = await run_study(config)
        assert second["provenance"]["live_calls"] == 0

    async def test_recorded_study_replays_with_the_upstream_removed(
        self, config: StudyConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stronger than counting live calls: break the upstream entirely and show
        # the numbers still come out the same.
        first = await run_study(config)

        def refuse(*_: object, **__: object) -> None:
            raise AssertionError("the study contacted an upstream during replay")

        monkeypatch.setattr(study_module, "create_stub_model", refuse)
        second = await run_study(config)

        assert second["results"]["by_variant"] == first["results"]["by_variant"]
        assert second["results"]["by_fault"] == first["results"]["by_fault"]


class TestCassetteShape:
    async def test_trunks_and_branches_are_named_readably(self, config: StudyConfig) -> None:
        await run_study(config)
        traces = Store(config.store_root).list_traces()
        assert any(t.startswith("trunk--stub-naive--baseline--T1") for t in traces)
        assert any("--s1--empty_success" in t for t in traces)

    async def test_branches_store_only_their_continuation(self, config: StudyConfig) -> None:
        await run_study(config)
        store = Store(config.store_root)
        branches = [t for t in store.list_traces() if t.startswith("branch--")]
        assert branches
        for branch_id in branches:
            branch = store.load_trace(branch_id)
            assert branch.kind == "branch"
            # A branch holds fewer entries than the run it resolves to, because it
            # inherits the trunk prefix rather than copying it.
            assert len(branch.entries) < len(store.resolve(branch_id))

    async def test_branches_cost_less_than_whole_runs(self, config: StudyConfig) -> None:
        # The storage claim: N fault experiments against an M-step run cost far
        # less than N full runs, because each branch keeps only its continuation.
        # (Blob-level dedup is exercised directly in test_store.py; this study
        # gives it nothing to collapse, since every response has a distinct
        # conversation prefix.)
        await run_study(config)
        store = Store(config.store_root)
        branches = [t for t in store.list_traces() if t.startswith("branch--")]
        stored = sum(len(store.load_trace(t).entries) for t in branches)
        resolved = sum(len(store.resolve(t)) for t in branches)
        assert stored < resolved

    async def test_no_credentials_in_any_cassette(self, config: StudyConfig) -> None:
        await run_study(config)
        for path in Path(config.store_root).rglob("*.json"):
            assert "authorization" not in path.read_text().lower()


class TestReproducibilityCheck:
    """The gate CI runs. It has to fail when the cassettes stop reproducing.

    These are synchronous: `main` calls `asyncio.run` itself, which cannot be
    entered from inside a running loop.
    """

    def test_passes_on_a_freshly_recorded_study(self, config: StudyConfig) -> None:
        asyncio.run(run_study(config))
        assert (
            study_module.main(
                [
                    "--store",
                    str(config.store_root),
                    "--report",
                    str(config.report_path),
                    "--check",
                ]
            )
            == 0
        )

    def test_fails_when_a_recorded_response_is_tampered_with(self, config: StudyConfig) -> None:
        summary = asyncio.run(run_study(config))
        store = Store(config.store_root)
        trace_id = next(
            run["trace_id"]
            for run in summary["results"]["runs"]
            if run.get("outcome") == "silent_corruption"
        )
        last = store.resolve(trace_id)[-1]
        blob = store.get_blob(last.blob)
        blob["choices"][0]["message"]["content"] = "ANSWER: 6725"
        path = store.blob_path(last.blob)
        path.write_text(json.dumps(blob, sort_keys=True), encoding="utf-8")

        assert (
            study_module.main(
                [
                    "--store",
                    str(config.store_root),
                    "--report",
                    str(config.report_path),
                    "--check",
                ]
            )
            == 1
        )

    def test_fails_when_there_is_no_committed_report(self, config: StudyConfig) -> None:
        assert (
            study_module.main(
                [
                    "--store",
                    str(config.store_root),
                    "--report",
                    str(config.report_path),
                    "--check",
                ]
            )
            == 1
        )


class TestProvenanceFlag:
    """`synthetic` must survive a replay, which needs no endpoint.

    The real-model path itself cannot be exercised here: recording against a real
    endpoint needs an endpoint. What is tested is that replaying a study does not
    silently relabel its provenance, which is the part that could go wrong without
    anyone noticing.
    """

    def test_derived_from_whether_an_upstream_was_given(self, tmp_path: Path) -> None:
        stub = StudyConfig(store_root=tmp_path, report_path=tmp_path / "r.json")
        assert stub.synthetic is True
        real = StudyConfig(
            store_root=tmp_path,
            report_path=tmp_path / "r.json",
            upstream_base_url="https://api.example/v1",
            model="some-model",
        )
        assert real.synthetic is False

    def test_override_wins_so_a_replay_keeps_its_provenance(self, tmp_path: Path) -> None:
        replayed = StudyConfig(
            store_root=tmp_path,
            report_path=tmp_path / "r.json",
            upstream_base_url=None,
            model="some-model",
            synthetic_override=False,
        )
        assert replayed.synthetic is False

    def test_a_named_model_displaces_the_stub(self, tmp_path: Path) -> None:
        real = StudyConfig(
            store_root=tmp_path,
            report_path=tmp_path / "r.json",
            upstream_base_url="https://api.example/v1",
            model="some-model",
        )
        assert real.agents == (("some-model", None),)

    def test_stub_policies_become_the_agent_dimension(self, tmp_path: Path) -> None:
        stub = StudyConfig(store_root=tmp_path, report_path=tmp_path / "r.json")
        assert stub.agents == (("stub/naive", "naive"), ("stub/careful", "careful"))

"""Tests for the command line interface.

The `serve` command's own server loop is not exercised here -- binding a port in a
test suite is flaky and the app it serves is tested directly in test_proxy.py. What is
tested is everything up to that point: argument validation, config construction, and
the inspection commands, since a broken entry point is the first thing a reviewer
hits.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evals.study import StudyConfig, run_study
from misfeed.cli import build_parser, main
from misfeed.faults import FAULT_IDS
from misfeed.verdict import Outcome


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small recorded study, shared across the read-only inspection tests."""
    root = tmp_path_factory.mktemp("cli")
    asyncio.run(
        run_study(
            StudyConfig(
                store_root=root / "cassettes",
                report_path=root / "report.json",
                faults=("empty_success",),
                stub_policies=("naive",),
            )
        )
    )
    return root / "cassettes"


class TestEntryPoint:
    def test_the_declared_console_script_is_importable(self) -> None:
        # pyproject declares `misfeed = "misfeed.cli:main"`. If this import breaks,
        # `pip install misfeed && misfeed` fails for every user.
        from misfeed.cli import main as entry

        assert callable(entry)

    def test_help_exits_cleanly(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        assert exit_info.value.code == 0

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code == 2

    def test_every_subcommand_is_wired_to_a_function(self) -> None:
        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        subcommands = [c for a in actions for c in (a.choices or {})]
        assert set(subcommands) >= {"serve", "traces", "show", "faults", "outcomes"}


class TestTaxonomyCommands:
    def test_faults_lists_every_class(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["faults"]) == 0
        out = capsys.readouterr().out
        for name in FAULT_IDS:
            assert name in out

    def test_faults_shows_a_description_for_each(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["faults"])
        for line in capsys.readouterr().out.strip().splitlines():
            name, _, description = line.partition("  ")
            assert description.strip(), f"{name} has no description"

    def test_outcomes_lists_every_outcome(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["outcomes"]) == 0
        out = capsys.readouterr().out
        for outcome in Outcome:
            assert str(outcome) in out


class TestTraces:
    def test_lists_trunks_and_branches(
        self, recorded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--store", str(recorded), "traces"]) == 0
        out = capsys.readouterr().out
        assert "trunk" in out
        assert "branch" in out
        assert "resolved" in out  # branches report their resolved length

    def test_reports_an_empty_store(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--store", str(tmp_path / "nothing"), "traces"]) == 1
        assert "no traces" in capsys.readouterr().out


class TestShow:
    def test_prints_each_step_with_what_the_model_saw(
        self, recorded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trace = "trunk--stub-naive--baseline--T1_shipped_total"
        assert main(["--store", str(recorded), "show", trace]) == 0
        out = capsys.readouterr().out
        assert "step 0" in out
        assert "MODEL" in out
        assert "usage" in out
        assert "Alcott Foods" in out

    def test_a_branch_reports_its_parent_and_fault(
        self, recorded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trace = "branch--stub-naive--baseline--T1_shipped_total--s2--empty_success"
        assert main(["--store", str(recorded), "show", trace]) == 0
        out = capsys.readouterr().out
        assert "forked at step 2" in out
        assert "empty_success" in out

    def test_truncates_by_default_and_not_with_full(
        self, recorded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trace = "trunk--stub-naive--baseline--T1_shipped_total"
        main(["--store", str(recorded), "show", trace])
        short = capsys.readouterr().out
        main(["--store", str(recorded), "show", trace, "--full"])
        full = capsys.readouterr().out
        assert "..." in short
        assert len(full) > len(short)

    def test_missing_trace_reports_an_error(
        self, recorded: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--store", str(recorded), "show", "nope"]) == 1
        assert "not in" in capsys.readouterr().err


class TestServeValidation:
    """`serve` must reject a bad configuration before binding a port."""

    def test_inject_without_a_parent_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "--store",
                str(tmp_path),
                "serve",
                "--mode",
                "inject",
                "--trace",
                "b",
                "--fault",
                "empty_success",
                "--at-step",
                "1",
                "--upstream",
                "http://example/v1",
            ]
        )
        assert code == 2
        assert "INJECT mode requires" in capsys.readouterr().err

    def test_record_without_an_upstream_is_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--store", str(tmp_path), "serve", "--mode", "record", "--trace", "t"])
        assert code == 2
        assert "requires upstream_base_url" in capsys.readouterr().err

    def test_unknown_fault_is_rejected_by_the_parser(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "--store",
                    str(tmp_path),
                    "serve",
                    "--mode",
                    "inject",
                    "--trace",
                    "b",
                    "--fault",
                    "not_a_fault",
                ]
            )
        assert exit_info.value.code == 2

    def test_api_key_can_come_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Passing a key on the command line puts it in the shell history and the
        # process table, so the environment variable is the documented path.
        monkeypatch.setenv("MISFEED_UPSTREAM_API_KEY", "from-env")
        captured: dict[str, object] = {}

        def fake_run(app: object, **kwargs: object) -> None:
            captured["engine"] = app.state.engine  # type: ignore[attr-defined]

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", fake_run)
        code = main(
            [
                "--store",
                str(tmp_path),
                "serve",
                "--mode",
                "record",
                "--trace",
                "t",
                "--upstream",
                "http://example/v1",
            ]
        )
        assert code == 0
        engine = captured["engine"]
        assert engine.config.upstream_api_key == "from-env"  # type: ignore[attr-defined]

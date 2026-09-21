"""Tests for `r3con.pipeline.run_pipeline` — orchestration under a RunConfig (offline).

The four stage functions (`surface_relevance`, `propose_schema`, `parse_documents`,
`reasoning.reason`) are monkeypatched so no LLM is called; the fakes capture the
kwargs they receive. Verifies the relevance → schema → parsing → reasoning flow, that
the relevant context reaches every downstream stage, that the config's
`params` + seed + per-stage prompt versions reach every stage, and that a reasoning
failure writes a discoverable `error.txt` before re-raising.

Run with:  uv run python tests/test_pipeline.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from r3con import pipeline as pipeline_mod
from r3con.config import RunConfig
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger
from r3con.stages import reasoning as reasoning_mod

_PROMPTS = {
    "relevance": "v3", "structuring/schema": "v3", "structuring/parsing": "v2",
    "reasoning": "v1",
}


def _cfg(**over: Any) -> RunConfig:
    """A minimal RunConfig for the orchestration tests (no LLM is actually called)."""
    base: dict[str, Any] = dict(name="test", model="m", seed=0, relevance_rounds=3,
                                prompts=dict(_PROMPTS), params={})
    base.update(over)
    return RunConfig(**base)


@contextlib.contextmanager
def _temp_logs_dir() -> Iterator[Path]:
    """Point the logs dir at a temp directory via the env var the resolver reads — there
    is deliberately no import-time LOGS_DIR attribute to monkeypatch."""
    import os

    with tempfile.TemporaryDirectory() as tmp:
        original = os.environ.get("R3CON_LOGS_DIR")
        os.environ["R3CON_LOGS_DIR"] = tmp
        try:
            yield Path(tmp)
        finally:
            if original is None:
                del os.environ["R3CON_LOGS_DIR"]
            else:
                os.environ["R3CON_LOGS_DIR"] = original


@contextlib.contextmanager
def _patched_stages(captured: dict) -> Iterator[None]:
    orig = (pipeline_mod.surface_relevance, pipeline_mod.propose_schema, pipeline_mod.parse_documents, reasoning_mod.reason)

    def fake_surface_relevance(*, task, documents, model, rounds, run, **kw):
        captured["relevance"] = {"rounds": rounds, "kw": kw}
        return SimpleNamespace(snippets=["s0", "s1"], rounds=[["a", "b"], ["s0", "s1"]])

    def fake_propose(*, task, relevance_snippets, model, run, **kw):
        captured["structuring/schema"] = {"relevance": relevance_snippets, "kw": kw}
        return SimpleNamespace(
            schema_code="class Parse: pass", parse_cls=object,
            attempts=[SimpleNamespace(thought="t", schema_code="x", error=None)],
        )

    def fake_extract(*, documents, schema_code, parse_cls, task, relevance_snippets, model, run, **kw):
        captured["structuring/parsing"] = {"relevance": relevance_snippets, "kw": kw}
        return SimpleNamespace(parse={"records": []}, source_docs={"records": []})

    def fake_reason(*, task, schema_code, parsed, relevance_snippets, model, max_turns, timeout_s, run, **kw):
        captured["reasoning"] = {"relevance": relevance_snippets, "kw": kw}
        return SimpleNamespace(answer="the-answer", terminated_by="final_answer", turns=[])

    pipeline_mod.surface_relevance = fake_surface_relevance
    pipeline_mod.propose_schema = fake_propose
    pipeline_mod.parse_documents = fake_extract
    reasoning_mod.reason = fake_reason
    try:
        yield
    finally:
        (pipeline_mod.surface_relevance, pipeline_mod.propose_schema,
         pipeline_mod.parse_documents, reasoning_mod.reason) = orig


def test_runs_all_stages_into_the_run_folder() -> None:
    captured: dict = {}
    with _temp_logs_dir() as root, _patched_stages(captured):
        res = run_pipeline(
            task="q", documents=["d1", "d2"], config=_cfg(), task_logger=TaskLogger("t1"),
        )
        # the pipeline hands back the answer AND the views it was derived from
        assert res.answer == "the-answer"
        assert str(res) == "the-answer"
        assert res.relevant_context == ["s0", "s1"]              # final-round snippets, per document
        assert res.structured_context == {"records": []}         # the merged parse
        assert res.schema_code == "class Parse: pass"     # the per-question schema
        assert res.source_docs == {"records": []}
        assert res.run_dir == root / "t1"                 # where to find the artifacts
        t = root / "t1"
        for rel in ("relevance/result.json", "structuring/schema/result.json", "structuring/parsing/result.json",
                    "reasoning/result.json", "reasoning/result.json"):
            assert (t / rel).is_file(), rel
        summ = json.loads((t / "relevance/result.json").read_text())
        assert summ["n_rounds"] == 3 and summ["n_docs"] == 2
        assert len(summ["rounds"]) == 2
        assert summ["rounds"][0]["round"] == 1
        assert summ["rounds"][-1]["snippets"] == ["s0", "s1"]


def test_relevance_states_flow_into_every_downstream_stage() -> None:
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=_cfg(), task_logger=TaskLogger("t1"))
    for stage in ("structuring/schema", "structuring/parsing", "reasoning"):
        assert captured[stage]["relevance"] == ["s0", "s1"], stage


def test_source_docs_reach_reasoning() -> None:
    """The parsing stage's per-record source-document provenance reaches reasoning, so it
    can tag each record with the document it came from."""
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=_cfg(), task_logger=TaskLogger("t1"))
    assert captured["reasoning"]["kw"]["source_docs"] == {"records": []}


def test_relevance_rounds_come_from_config() -> None:
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=_cfg(relevance_rounds=2), task_logger=TaskLogger("t1"))
    assert captured["relevance"]["rounds"] == 2
    captured.clear()
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=_cfg(relevance_rounds=5), task_logger=TaskLogger("t1"))
    assert captured["relevance"]["rounds"] == 5


def test_schema_proposal_uses_settings_max_attempts() -> None:
    """run_pipeline must not override the schema proposal's retry budget — it passes no
    ``max_attempts``, so ``propose_schema`` uses ``settings.SCHEMA_MAX_ATTEMPTS``."""
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=_cfg(), task_logger=TaskLogger("t1"))
    assert "max_attempts" not in captured["structuring/schema"]["kw"]


def test_config_params_and_seed_reach_every_stage() -> None:
    captured: dict = {}
    cfg = _cfg(seed=0, params={"temperature": 0.7, "extra_body": {"top_k": 20}})
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=cfg,
                  task_logger=TaskLogger("t1"))
    for stage in ("relevance", "structuring/schema", "structuring/parsing", "reasoning"):
        kw = captured[stage]["kw"]
        assert kw["temperature"] == 0.7, stage
        assert kw["extra_body"] == {"top_k": 20}, stage
        assert kw["seed"] == 0, stage  # transport kwarg, from config.seed


def test_per_stage_prompt_versions_reach_each_stage() -> None:
    captured: dict = {}
    cfg = _cfg(prompts={"relevance": "vA", "structuring/schema": "vB",
                        "structuring/parsing": "vC", "reasoning": "vD"})
    with _temp_logs_dir(), _patched_stages(captured):
        run_pipeline(task="q", documents=["d"], config=cfg,
                  task_logger=TaskLogger("t1"))
    assert captured["relevance"]["kw"]["prompt_version"] == "vA"
    assert captured["structuring/schema"]["kw"]["prompt_version"] == "vB"
    assert captured["structuring/parsing"]["kw"]["prompt_version"] == "vC"
    assert captured["reasoning"]["kw"]["prompt_version"] == "vD"


def test_a_reasoning_failure_raises_but_leaves_the_traceback_on_disk() -> None:
    """The earlier stages' artifacts are already written, so the run folder should also
    say why the run ended — while the caller still gets the exception."""
    captured: dict = {}
    with _temp_logs_dir() as root, _patched_stages(captured):
        def boom(**_):
            raise ValueError("ctx too long")

        reasoning_mod.reason = boom  # restored by _patched_stages' finally
        raised = None
        try:
            run_pipeline(task="q", documents=["d"], config=_cfg(), task_logger=TaskLogger("t1"))
        except ValueError as e:
            raised = e
        assert raised is not None and "ctx too long" in str(raised)
        t = root / "t1"
        err = t / "reasoning" / "error.txt"
        assert err.is_file() and "ValueError" in err.read_text()
        assert not (t / "reasoning" / "result.json").is_file()
        # the upstream stages still left their work behind
        assert (t / "relevance" / "result.json").is_file()
        assert (t / "manifest.json").is_file()


def test_no_logger_returns_the_answer_without_writing() -> None:
    captured: dict = {}
    with _temp_logs_dir() as root, _patched_stages(captured):
        res = run_pipeline(task="q", documents=["d"], config=_cfg())
        assert res.answer == "the-answer"
        assert res.run_dir is None      # nothing was written, so there is nowhere to point
        assert res.relevant_context == ["s0", "s1"]  # the views are returned regardless
        assert not any(root.iterdir())  # nothing written without a task_logger


if __name__ == "__main__":
    tests = [
        test_runs_all_stages_into_the_run_folder,
        test_relevance_states_flow_into_every_downstream_stage,
        test_source_docs_reach_reasoning,
        test_relevance_rounds_come_from_config,
        test_schema_proposal_uses_settings_max_attempts,
        test_config_params_and_seed_reach_every_stage,
        test_per_stage_prompt_versions_reach_each_stage,
        test_a_reasoning_failure_raises_but_leaves_the_traceback_on_disk,
        test_no_logger_returns_the_answer_without_writing,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")

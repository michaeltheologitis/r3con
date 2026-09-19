"""Tests for `r3con.runs.TaskLogger` and `StageRun` (bare-folder logging).

Run with:  uv run python tests/test_runs.py
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from r3con.runs import StageRun, TaskLogger, new_run_folder, normalize_model_name


class _Sample(BaseModel):
    name: str
    count: int = Field(description="some count")


def _msgs_assistant(content: str = "ok", finish: str = "stop") -> dict:
    return {"role": "assistant", "content": content, "finish_reason": finish}


# ----- TaskLogger -----


def test_creates_per_task_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("task-1", root=Path(tmp))
        assert log.dir == Path(tmp) / "task-1"
        assert log.dir.is_dir()


def test_new_run_folder_is_unique_timestamped_and_opaque() -> None:
    a = new_run_folder()
    b = new_run_folder()
    # <UTC-timestamp>_<hex> — opaque (no identity), flat, and unique per call.
    assert a != b
    assert "/" not in a
    ts, sep, hexpart = a.partition("_")
    assert sep == "_"
    assert ts.endswith("Z") and len(ts) == len("20260613T142233Z")
    assert len(hexpart) == 8 and all(ch in "0123456789abcdef" for ch in hexpart)


def test_task_logger_flat_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("gpt__r3__abc123", task_id="t1", root=Path(tmp))
        assert log.dir == Path(tmp) / "gpt__r3__abc123"  # one flat level, no nesting
        assert log.dir.is_dir()
        assert log.folder == "gpt__r3__abc123" and log.task_id == "t1"
        # task_id defaults to the folder name when not given
        assert TaskLogger("solo", root=Path(tmp)).task_id == "solo"


def test_write_text_and_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_text("answer", "final\n")
        log.write_json("score", {"score": 1, "gold": "yes"})
        assert (log.dir / "answer.txt").read_text() == "final\n"
        assert json.loads((log.dir / "score.json").read_text()) == {"score": 1, "gold": "yes"}


def test_write_json_pydantic_instance_and_class() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_json("parser", _Sample(name="x", count=3))
        log.write_json("schema", _Sample)
        assert json.loads((log.dir / "parser.json").read_text()) == {"name": "x", "count": 3}
        assert "properties" in json.loads((log.dir / "schema.json").read_text())


def test_write_json_nested_pydantic_and_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_json("p", {"attempt": 1, "result": _Sample(name="x", count=3)})
        assert json.loads((log.dir / "p.json").read_text())["result"] == {"name": "x", "count": 3}
        log.write_text("a", "first")
        log.write_text("a", "second")
        assert (log.dir / "a.txt").read_text() == "second"


def test_unserializable_fallback_to_repr() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_json("oddity", {"obj": Opaque()})
        assert json.loads((log.dir / "oddity.json").read_text()) == {"obj": "<Opaque>"}


def test_write_json_keeps_unicode_readable() -> None:
    """Non-ASCII (e.g. CJK) is written literally, not as \\uXXXX escapes, and still
    round-trips through json.loads."""
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        p = log.write_json("r", {"label": "应付账款"})
        raw = p.read_text(encoding="utf-8")
        assert "应付账款" in raw and "\\u" not in raw
        assert json.loads(raw) == {"label": "应付账款"}


def test_calls_json_keeps_unicode_readable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/parsing", task_logger=task)
        run.add_step(kind="parse-d0", messages=[{"role": "user", "content": "文档"}],
                     response=_msgs_assistant("应付账款"))
        run.flush(write_transcript=False)
        raw = (run.dir / "calls.json").read_text(encoding="utf-8")
        assert "应付账款" in raw and "文档" in raw and "\\u" not in raw


def test_write_creates_subdirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        p = log.write_json("reasoning/result", {"answer": "x"})
        assert p.exists() and p.name == "result.json" and p.parent.name == "reasoning"
        y = log.write_yaml("structuring/schema/transcript", {"messages": [{"role": "user", "content": "hi"}]})
        assert y.exists() and y.name == "transcript.yaml"


# ----- StageRun -----


def test_stage_run_makes_subdir_and_flushes_transcript() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/schema", task_logger=task, model="m", seed=0)
        assert run.dir == task.dir / "structuring/schema" and run.dir.is_dir()
        run.add_step(
            kind="llm_call",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            response=_msgs_assistant("a"),
            tokens={"prompt": 5, "completion": 1, "total": 6},
        )
        ty = run.flush()
        assert ty.name == "transcript.yaml" and ty.exists()
        assert (run.dir / "calls.json").is_file()


def test_calls_json_parses_doc_from_new_kinds() -> None:
    """calls.json carries per-call provenance — ``doc`` parsed from the new per-doc
    kinds (``parse-d0``, ``summary-r1-d2``); ``chunk`` is None (no chunking)."""
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/parsing", task_logger=task, model="m", seed=0)
        run.add_step(kind="summary-r1-d2", messages=[{"role": "user", "content": "u"}],
                     response=_msgs_assistant("a"), tokens={"prompt": 5, "completion": 1, "total": 6})
        run.add_step(kind="parse-d0", messages=[{"role": "user", "content": "u2"}],
                     response=_msgs_assistant("b", finish="length"), tokens={"prompt": 9, "completion": 3, "total": 12})
        run.flush(write_transcript=False)
        calls = json.loads((run.dir / "calls.json").read_text())
        assert (calls[0]["kind"], calls[0]["doc"], calls[0]["chunk"]) == ("summary-r1-d2", 2, None)
        assert (calls[1]["kind"], calls[1]["doc"], calls[1]["chunk"]) == ("parse-d0", 0, None)
        assert calls[0]["output"] == "a" and calls[1]["finish_reason"] == "length"


def test_flush_skip_transcript() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="relevance", task_logger=task)
        run.add_step(kind="summary-r1-d0", messages=[{"role": "user", "content": "u"}], response=_msgs_assistant("a"))
        p = run.flush(write_transcript=False)
        assert (run.dir / "calls.json").is_file()
        assert not (run.dir / "transcript.yaml").exists()
        assert p.name == "calls.json"


def test_compute_totals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="reasoning", task_logger=task)
        for pt, ct in [(10, 2), (20, 5), (5, 1)]:
            run.add_step(messages=[{"role": "user", "content": "u"}], response=_msgs_assistant("r"),
                         tokens={"prompt": pt, "completion": ct, "total": pt + ct})
        totals = run.compute_totals()
        assert totals["n_steps"] == 3
        assert totals["tokens"] == {"prompt": 35, "completion": 8, "total": 43}


def test_compute_totals_none_when_no_tokens() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/schema", task_logger=task)
        run.add_step(messages=[{"role": "user", "content": "u"}], response=_msgs_assistant("r"))
        assert run.compute_totals() == {"n_steps": 1, "tokens": None}


def test_transcript_multi_turn_thread() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="reasoning", task_logger=task)
        msgs1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
        run.add_step(kind="turn-1", messages=msgs1, response=_msgs_assistant("A1"))
        msgs2 = msgs1 + [{"role": "assistant", "content": "A1"}, {"role": "user", "content": "<observation>obs</observation>"}]
        run.add_step(kind="turn-2", messages=msgs2, response=_msgs_assistant("A2 final"))
        transcript = yaml.safe_load(run.flush().read_text())
        contents = [m["content"] for m in transcript["messages"]]
        assert contents == ["S", "U1", "A1", "<observation>obs</observation>", "A2 final"]


def test_transcript_is_final_attempt_for_retry_loop() -> None:
    """For a retry loop (the schema proposal) the transcript is just the final (successful)
    attempt; the earlier failed attempt lives in calls.json, not the transcript."""
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/schema", task_logger=task)
        run.add_step(kind="llm_call", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "first"}], response=_msgs_assistant("a1"))
        run.add_step(kind="retry", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "retry-prompt"}], response=_msgs_assistant("a2"))
        contents = [m["content"] for m in yaml.safe_load(run.flush().read_text())["messages"]]
        assert contents == ["S", "retry-prompt", "a2"]  # final attempt only
        # The failed first attempt is preserved in calls.json.
        calls = json.loads((run.dir / "calls.json").read_text())
        assert calls[0]["output"] == "a1" and calls[1]["output"] == "a2"


def test_transcript_single_step() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/schema", task_logger=task)
        run.add_step(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], response=_msgs_assistant("A"))
        transcript = yaml.safe_load(run.flush().read_text())
        assert [m["content"] for m in transcript["messages"]] == ["S", "U", "A"]


def test_no_steps_is_safe_to_flush() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/schema", task_logger=task)
        transcript = yaml.safe_load(run.flush().read_text())
        assert transcript["messages"] == []
        assert run.compute_totals()["n_steps"] == 0


def test_add_step_is_thread_safe() -> None:
    """Concurrent add_step (the parallel doc fan-out shares one run) assigns unique,
    contiguous step numbers and loses no steps."""
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="relevance", task_logger=task)

        def worker(i: int) -> None:
            run.add_step(kind=f"summary-r1-d{i}", messages=[{"role": "user", "content": str(i)}], response=_msgs_assistant("x"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(run.steps) == 50
        assert sorted(s.step for s in run.steps) == list(range(1, 51))


# ----- helpers -----


def test_normalize_model_name() -> None:
    assert normalize_model_name("openai/gpt-5.6-luna") == "gpt-5.6-luna"
    assert normalize_model_name("hosted_vllm/Qwen/Qwen3-8B") == "Qwen3-8B"
    assert normalize_model_name("gpt-5.6-luna") == "gpt-5.6-luna"
    assert normalize_model_name(None) is None


def test_transcript_normalizes_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="structuring/schema", task_logger=task, model="hosted_vllm/Qwen/Qwen3-8B")
        run.add_step(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], response=_msgs_assistant("a"))
        assert yaml.safe_load(run.flush().read_text())["model"] == "Qwen3-8B"


def test_transcript_block_style_despite_trailing_spaces() -> None:
    from r3con.runs import _dump_transcript_yaml

    out = _dump_transcript_yaml({"content": "1. Thorir  \n2. Gudrun  \n3. Amleth"})
    assert "content: |" in out
    assert "\\n" not in out
    assert yaml.safe_load(out)["content"] == "1. Thorir\n2. Gudrun\n3. Amleth"



# ---------- manifest: the run folder is opaque, so this is the only identity ----------


def test_manifest_records_the_identity_and_the_package_version() -> None:
    from r3con.config import RunConfig
    from r3con.runs import write_manifest

    cfg = RunConfig(
        model="hosted_vllm/Some/Model-X", seed=7, relevance_rounds=3,
        prompts={"relevance": "v1", "structuring/schema": "v1", "structuring/parsing": "v1",
                 "reasoning": "v1"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("run-x", root=Path(tmp))
        write_manifest(log, task="what happened?", config=cfg, n_docs=3, context_chars=1234)
        data = json.loads((Path(tmp) / "run-x" / "manifest.json").read_text())

    assert data["task"] == "what happened?"
    assert data["n_docs"] == 3 and data["context_chars"] == 1234
    # the provider prefix is transport, not identity — the manifest records the bare name
    assert data["config"]["model"] == "Model-X"
    assert data["config"]["seed"] == 7 and data["config"]["relevance_rounds"] == 3
    assert data["config"]["prompts"]["reasoning"] == "v1"
    # prompts ship INSIDE the package, so "v1" only means something against a version
    assert data["r3con_version"]
    assert data["settings"]["doc_workers"] >= 1
    assert data["created"].startswith("20")


def test_manifest_is_written_before_the_first_stage() -> None:
    """A run that dies in stage 1 must still be identifiable, so the manifest cannot
    wait until the end."""
    import r3con.pipeline as pipeline_mod
    from r3con.config import RunConfig

    cfg = RunConfig(
        model="m", prompts={"relevance": "v1", "structuring/schema": "v1",
                            "structuring/parsing": "v1",
                            "reasoning": "v1"},
    )
    original = pipeline_mod.surface_relevance

    def boom(**_):
        raise RuntimeError("stage 1 exploded")

    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("crashed", root=Path(tmp))
        pipeline_mod.surface_relevance = boom
        try:
            pipeline_mod.run_pipeline(task="q", documents=["d"], config=cfg, task_logger=log)
        except RuntimeError:
            pass
        finally:
            pipeline_mod.surface_relevance = original
        manifest = Path(tmp) / "crashed" / "manifest.json"
        assert manifest.is_file(), "a crashed run left no identity behind"
        assert json.loads(manifest.read_text())["task"] == "q"


def test_manifest_records_where_the_config_and_prompts_came_from() -> None:
    """An overlay file and the packaged file of the same version are indistinguishable in
    the run label, so the manifest has to say which one actually won."""
    import os

    from r3con.config import RunConfig
    from r3con.runs import write_manifest
    from r3con.settings import settings

    stages = ("relevance", "structuring/schema", "structuring/parsing", "reasoning")
    cfg = RunConfig(model="m", prompts={k: "v1" for k in stages})
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("r", root=Path(tmp))
        write_manifest(log, task="q", config=cfg, n_docs=1)
        src = json.loads((Path(tmp) / "r" / "manifest.json").read_text())["sources"]
    # with no overlay present, every prompt resolves inside the installed package
    assert set(src["prompts"]) == set(stages)
    for stage, path in src["prompts"].items():
        assert path and str(settings.PROMPTS_DIR) in path, (stage, path)
    assert src["config"] and str(settings.CONFIGS_DIR) in src["config"]

    # now shadow one stage from an overlay and confirm the manifest points at the overlay
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as overlay:
        d = Path(overlay) / "relevance"
        d.mkdir(parents=True)
        (d / "v1.yaml").write_text("instructions: |-\n  OVERLAID\n", encoding="utf-8")
        os.environ["R3CON_PROMPTS_DIR"] = overlay
        try:
            log = TaskLogger("r2", root=Path(tmp))
            write_manifest(log, task="q", config=cfg, n_docs=1)
            src = json.loads((Path(tmp) / "r2" / "manifest.json").read_text())["sources"]
        finally:
            del os.environ["R3CON_PROMPTS_DIR"]
    assert src["prompts"]["relevance"].startswith(overlay), src["prompts"]["relevance"]
    assert str(settings.PROMPTS_DIR) in src["prompts"]["reasoning"]  # unshadowed one unchanged


def test_params_must_be_a_mapping() -> None:
    """A malformed `params:` should say so, not surface as a TypeError from inside litellm."""
    import yaml as _yaml

    from r3con.config import load_config
    from r3con.settings import settings

    stages = ("relevance", "structuring/schema", "structuring/parsing", "reasoning")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "bad.yaml").write_text(_yaml.safe_dump(
            {"model": "m", "prompts": {k: "v1" for k in stages}, "params": "temperature=0.7"}))
        original = settings.CONFIGS_DIR
        settings.CONFIGS_DIR = root
        try:
            raised = None
            try:
                load_config("bad")
            except ValueError as e:
                raised = e
            assert raised is not None and "params" in str(raised) and "mapping" in str(raised), raised
        finally:
            settings.CONFIGS_DIR = original

if __name__ == "__main__":
    tests = [
        test_manifest_records_the_identity_and_the_package_version,
        test_manifest_records_where_the_config_and_prompts_came_from,
        test_params_must_be_a_mapping,
        test_manifest_is_written_before_the_first_stage,
        test_creates_per_task_dir,
        test_new_run_folder_is_unique_timestamped_and_opaque,
        test_task_logger_flat_folder,
        test_write_text_and_json,
        test_write_json_pydantic_instance_and_class,
        test_write_json_nested_pydantic_and_overwrite,
        test_unserializable_fallback_to_repr,
        test_write_json_keeps_unicode_readable,
        test_calls_json_keeps_unicode_readable,
        test_write_creates_subdirs,
        test_stage_run_makes_subdir_and_flushes_transcript,
        test_calls_json_parses_doc_from_new_kinds,
        test_flush_skip_transcript,
        test_compute_totals,
        test_compute_totals_none_when_no_tokens,
        test_transcript_multi_turn_thread,
        test_transcript_is_final_attempt_for_retry_loop,
        test_transcript_single_step,
        test_no_steps_is_safe_to_flush,
        test_add_step_is_thread_safe,
        test_normalize_model_name,
        test_transcript_normalizes_model,
        test_transcript_block_style_despite_trailing_spaces,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")

import re
import time
from pathlib import Path

import litellm
import pytest

from r3con import settings
from r3con.cli import main
from r3con.config import PROMPT_STAGES
from r3con.r3con import read_documents

MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
ANSWER = "; ".join(memo.splitlines()[0] for memo in read_documents(MEMOS))
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"


def r3con(*argv: str) -> int:
    """The exit code ``r3con <argv>`` would leave, argparse's included."""
    try:
        return main(list(argv))
    except SystemExit as exit:
        return exit.code


def path_after(prefix: str, stderr: str) -> Path:
    """The path on the stderr line that starts with ``prefix``."""
    line = re.search(rf"^{re.escape(prefix)}(.+)$", stderr, re.MULTILINE)
    return Path(line.group(1))


@pytest.fixture
def answering(answering_llm, monkeypatch):
    monkeypatch.setattr(litellm, "completion", answering_llm)
    return answering_llm


@pytest.fixture
def scripted(llm, monkeypatch):
    monkeypatch.setattr(litellm, "completion", llm)
    return llm


def test_a_run_prints_the_answer_alone_on_stdout(answering, capsys):
    assert r3con("run", "Who?", str(MEMOS), "--relevance-rounds", "1") == 0
    out, err = capsys.readouterr()
    assert out == f"{ANSWER}\n"
    assert "5 document(s) · default[model=gpt-6-luna,rounds=1," in err
    assert (path_after("artifacts: ", err) / "manifest.json").is_file()


def test_the_flags_override_the_config_and_show_in_the_label(answering, capsys):
    argv = ["--model", QWEN, "--relevance-rounds", "1"]
    assert r3con("run", "Who?", str(MEMOS), *argv) == 0
    assert "default[model=Qwen3.5-35B-A3B,rounds=1," in capsys.readouterr().err
    assert {r["model"] for r in answering.requests} == {QWEN}


def test_seed_is_not_an_option(scripted, capsys):
    assert r3con("run", "Who?", str(MEMOS), "--seed", "7") == 2
    assert "unrecognized arguments: --seed" in capsys.readouterr().err
    assert scripted.requests == []


def test_a_source_that_matches_nothing_exits_2_and_says_so(scripted, capsys):
    assert r3con("run", "Who?", "./reports") == 2
    assert capsys.readouterr().err == "r3con: No documents found at './reports'.\n"
    assert scripted.requests == []


def test_sources_that_are_all_empty_exit_2(scripted, tmp_path, capsys):
    (tmp_path / "empty.txt").write_text("  \n")
    assert r3con("run", "Who?", str(tmp_path / "empty.txt")) == 2
    assert "r3con: no documents to read" in capsys.readouterr().err
    assert scripted.requests == []


def test_a_usage_error_exits_2(scripted, capsys):
    assert r3con("run", "Who?", str(MEMOS), "--config", "nope") == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err
    assert scripted.requests == []


def test_a_provider_failure_exits_1_and_points_at_the_partial_artifacts(
    scripted, capsys
):
    down = litellm.APIConnectionError("provider down", llm_provider="openai", model="m")
    scripted.answers(relevance=down)
    assert r3con("run", "Who?", str(MEMOS)) == 1
    err = capsys.readouterr().err
    assert "r3con: APIConnectionError: litellm.APIConnectionError: provider down" in err
    partial = path_after("r3con: partial artifacts in ", err)
    assert partial == path_after("artifacts: ", err)
    assert (partial / "manifest.json").is_file()


def test_a_config_pinning_a_missing_prompt_exits_2_before_any_request(
    scripted, configs, tmp_path, capsys
):
    prompts = {**dict.fromkeys(PROMPT_STAGES, "v1"), "reasoning": "v9"}
    configs(exp={"model": "openai/m", "prompts": prompts})
    assert r3con("run", "Who?", str(MEMOS), "--config", "exp") == 2
    err = capsys.readouterr().err
    assert err.startswith("r3con: No prompt for stage 'reasoning' version 'v9'")
    assert err.count("\n") == 1
    assert scripted.requests == []
    assert not (tmp_path / "logs").exists()


def test_an_interrupt_exits_130(scripted, capsys):
    scripted.replies(KeyboardInterrupt())
    assert r3con("run", "Who?", str(MEMOS / "01_northgate.txt")) == 130
    assert capsys.readouterr().err.endswith("r3con: interrupted.\n")


def test_no_artifacts_writes_no_run_folder(answering, tmp_path, capsys):
    argv = ["--relevance-rounds", "1", "--no-artifacts"]
    assert r3con("run", "Who?", str(MEMOS), *argv) == 0
    assert "artifacts:" not in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_logs_dir_chooses_where_the_run_folder_goes(answering, tmp_path, capsys):
    out = tmp_path / "out"
    argv = ["--relevance-rounds", "1", "--logs-dir", str(out)]
    assert r3con("run", "Who?", str(MEMOS), *argv) == 0
    run_dir = path_after("artifacts: ", capsys.readouterr().err)
    assert run_dir.parent == out
    assert (run_dir / "manifest.json").is_file()


@pytest.mark.parametrize("workers", [1, 2])
def test_doc_workers_bounds_how_many_documents_are_read_at_once(answering, workers):
    def slow_note(request) -> str:
        time.sleep(0.2)
        return "a note"

    answering.answers(relevance=slow_note)
    argv = ["--relevance-rounds", "1", "--doc-workers", str(workers)]
    assert r3con("run", "Who?", str(MEMOS), *argv) == 0
    assert answering.peak_in_flight == workers


def test_doc_workers_below_one_exits_2_before_a_run_folder_exists(
    scripted, tmp_path, capsys
):
    assert r3con("run", "Who?", str(MEMOS), "--doc-workers", "0") == 2
    assert capsys.readouterr().err == "r3con: DOC_WORKERS must be >= 1, got 0.\n"
    assert scripted.requests == []
    assert list(tmp_path.iterdir()) == []


def test_a_cap_that_is_not_an_integer_exits_2_without_a_traceback(
    scripted, tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(settings, "SCHEMA_MAX_ATTEMPTS", "3")
    assert r3con("run", "Who?", str(MEMOS)) == 2
    err = capsys.readouterr().err
    assert err == "r3con: SCHEMA_MAX_ATTEMPTS must be an integer >= 1, got '3'.\n"
    assert scripted.requests == []
    assert list(tmp_path.iterdir()) == []


def test_verbose_streams_stage_progress_to_stderr(answering, capsys):
    assert r3con("run", "Who?", str(MEMOS), "--relevance-rounds", "1", "-v") == 0
    out, err = capsys.readouterr()
    assert out == f"{ANSWER}\n"
    assert "stage 1/3" in err
    assert "stage 3/3" in err

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
    argv = ["--model", QWEN, "--relevance-rounds", "1", "--reasoning", "coding"]
    assert r3con("run", "Who?", str(MEMOS), *argv) == 0
    out, err = capsys.readouterr()
    assert out == f"{ANSWER}\n"
    assert "default[model=Qwen3.5-35B-A3B,rounds=1,reasoning=coding," in err
    assert {r["model"] for r in answering.requests} == {QWEN}
    [turn] = answering.requests_for("reasoning")
    assert "final_answer(" in turn["messages"][0]["content"]


def test_ask_prints_a_new_answer_over_a_finished_run(answering, capsys):
    assert r3con("run", "Who?", str(MEMOS), "--relevance-rounds", "1") == 0
    run_dir = path_after("artifacts: ", capsys.readouterr().err)
    assert r3con("ask", str(run_dir), "Who else?", "--model", QWEN) == 0
    assert capsys.readouterr().out == f"{ANSWER}\n"
    asked = answering.requests[-1]
    assert asked["model"] == QWEN
    assert "Who else?" in asked["messages"][1]["content"]
    assert (run_dir / "asks" / "001-Qwen3.5-35B-A3B" / "ask.json").is_file()


def test_ask_over_a_folder_that_is_not_a_run_exits_2_in_one_line(
    scripted, tmp_path, capsys
):
    assert r3con("ask", str(tmp_path), "Who?") == 2
    assert capsys.readouterr().err == (
        f"r3con: {tmp_path} is not a run folder: it has no manifest.json.\n"
    )
    assert scripted.requests == []


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


@pytest.mark.parametrize("option", ["--config", "--reasoning"])
def test_a_usage_error_exits_2(scripted, capsys, option):
    assert r3con("run", "Who?", str(MEMOS), option, "nope") == 2
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
    assert err.count("partial artifacts") == 1
    assert (partial / "manifest.json").is_file()
    assert "provider down" in (partial / "relevance" / "error.txt").read_text()


def test_a_config_pinning_a_missing_prompt_exits_2_before_any_request(
    scripted, configs, tmp_path, capsys
):
    prompts = {**dict.fromkeys(PROMPT_STAGES, "v1"), "reasoning/coding": "v9"}
    configs(exp={"model": "openai/m", "prompts": prompts})
    assert r3con("run", "Who?", str(MEMOS), "--config", "exp") == 2
    err = capsys.readouterr().err
    assert err.startswith("r3con: No prompt for stage 'reasoning/coding' version 'v9'")
    assert err.count("\n") == 1
    assert scripted.requests == []
    assert not (tmp_path / "logs").exists()


def test_a_config_pinning_a_stage_r3con_does_not_have_exits_2_in_one_line(
    scripted, configs, capsys
):
    prompts = {**dict.fromkeys(PROMPT_STAGES, "v1"), "relevance": "v1"}
    configs(old={"model": "openai/m", "prompts": prompts})
    assert r3con("run", "Who?", str(MEMOS), "--config", "old") == 2
    assert capsys.readouterr().err == (
        "r3con: config 'old': `prompts.relevance`: Input should be "
        "'relevance/first-round', 'relevance/later-rounds', 'structuring/schema', "
        "'structuring/parsing', 'reasoning/direct' or 'reasoning/coding'.\n"
    )
    assert scripted.requests == []


def test_a_config_naming_a_reasoning_path_r3con_does_not_have_exits_2_in_one_line(
    scripted, configs, capsys
):
    prompts = dict.fromkeys(PROMPT_STAGES, "v1")
    configs(bad={"model": "openai/m", "reasoning": "agent", "prompts": prompts})
    assert r3con("run", "Who?", str(MEMOS), "--config", "bad") == 2
    assert capsys.readouterr().err == (
        "r3con: config 'bad': `reasoning='agent'`: Input should be 'direct' or "
        "'coding'.\n"
    )
    assert scripted.requests == []


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


@pytest.mark.parametrize(
    ("argv", "env", "stderr"),
    [
        (
            ["--doc-workers", "0"],
            {},
            "r3con: --doc-workers must be >= 1, got 0.\n",
        ),
        (
            [],
            {"R3CON_DOC_WORKERS": "0"},
            "r3con: R3CON_DOC_WORKERS must be >= 1, got 0.\n",
        ),
        (
            [],
            {"R3CON_DOC_WORKERS": "abc"},
            "r3con: R3CON_DOC_WORKERS must be an integer >= 1, got 'abc'.\n",
        ),
        (
            [],
            {"R3CON_LOG_LEVEL": "loud"},
            "r3con: R3CON_LOG_LEVEL must be a level name such as INFO, got 'loud'.\n",
        ),
    ],
    ids=["flag-below-one", "variable-below-one", "variable-not-int", "bad-level"],
)
def test_a_bad_setting_exits_2_naming_it_before_any_document_is_read(
    scripted, tmp_path, capsys, monkeypatch, argv, env, stderr
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert r3con("run", "Who?", "./missing", *argv) == 2
    assert capsys.readouterr().err == stderr
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

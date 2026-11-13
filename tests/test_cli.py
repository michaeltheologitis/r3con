import contextlib
import json
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx2
import litellm
import platformdirs
import pytest

from r3con import settings
from r3con.app.server import (
    APP_KEY_SERVICE,
    FORCED_EXIT_CODE,
    MAC_APP_KEY_SERVICE,
    QUIT_SECONDS,
)
from r3con.cli import build_parser, main
from r3con.config import PROMPT_STAGES, load_config
from r3con.documents import read_documents
from r3con.runs import run_lock
from tests.fakes import (
    ANSWERING,
    KEYRING_FILE,
    a_parse_citation_short,
    bad_key,
    failing_once,
    in_json,
    predating_resume,
    reading,
    reading_moved_prompts,
)

REPO = Path(__file__).resolve().parents[1]
MEMOS = REPO / "examples" / "memos"
ANSWER = "; ".join(memo.splitlines()[0] for memo in read_documents(MEMOS).values())
RIVERSIDE = "MEMO — Riverside site, Q3 operations review"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
GATEWAY = "https://gateway.example"
MASKED_HEADER = {"extra_headers": {"Authorization": "REDACTED"}}
APP_MODULES = ("starlette", "uvicorn", "platformdirs", "keyring")
# Far longer than the app takes to print its two lines, which it does before serving.
LAUNCH_SECONDS = 30
# r3con app whose model calls say so on stdout, in one write, then never return.
NEVER_ANSWERING_APP = """
import sys, threading
import litellm
from r3con.cli import main

def never_answers(**request):
    sys.stdout.write("a model call\\n")
    sys.stdout.flush()
    threading.Event().wait()

litellm.completion = never_answers
sys.exit(main(sys.argv[1:]))
"""
MODEL_CALLED = "a model call\n"
# Two documents, so their calls run on the document threads Python's exit joins.
NEVER_ANSWERED_RUN = {
    "question": "Which memo?",
    "model": "openai/never-answers",
    "documents": ["A memo.", "Another memo."],
}
CUT_OFF = "r3con app: quitting, cutting off 1 run(s) or ask(s) still going"
# What uvicorn prints as its grace runs out on an open stream, which reads as a crash.
CRASH_SIGNS = ("Traceback", "CancelledError", "timeout graceful shutdown exceeded")
# Room past QUIT_SECONDS for a slow runner, and far short of the old 60 s wait.
QUIT_BOUND_SECONDS = QUIT_SECONDS + 3
NO_APP_EXTRA = (
    'r3con: the app needs its extra: uv tool install "r3context[app]" '
    '(or pip install "r3context[app]")\n'
)


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


def finished_run(capsys) -> Path:
    """The folder of a finished ``r3con run`` over the memos."""
    assert r3con("run", "Who?", str(MEMOS), "--relevance-rounds", "1") == 0
    return path_after("artifacts: ", capsys.readouterr().err)


def test_a_run_records_each_document_by_its_file_name(answering, capsys):
    run_dir = finished_run(capsys)
    names = json.loads((run_dir / "names.json").read_text())
    assert list(names.values()) == sorted(path.name for path in MEMOS.iterdir())


def as_written_before_ids(run_dir: Path) -> None:
    (run_dir / "ids.json").unlink()


def pinning_a_missing_prompt(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"]["prompts"]["reasoning/coding"] = "v9"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def test_ask_prints_a_new_answer_and_where_its_artifacts_are(answering, capsys):
    run_dir = finished_run(capsys)
    assert r3con("ask", str(run_dir), "Who else?", "--model", QWEN) == 0
    out, err = capsys.readouterr()
    assert out == f"{ANSWER}\n"
    folder = path_after("artifacts: ", err)
    assert folder.parent == run_dir / "asks"
    assert (folder / "manifest.json").is_file()
    asked = answering.requests[-1]
    assert asked["model"] == QWEN
    assert "Who else?" in asked["messages"][1]["content"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            as_written_before_ids,
            "as a run folder: FileNotFoundError",
        ),
        (
            pinning_a_missing_prompt,
            "No prompt for stage 'reasoning/coding' version 'v9'",
        ),
        (
            in_json("structuring/parsing/result.json", a_parse_citation_short),
            "as a run folder: ValueError: zip() argument 2 is shorter than argument 1",
        ),
    ],
    ids=["older-r3con", "missing-prompt", "a-parse-citation-short"],
)
def test_ask_over_a_run_folder_it_cannot_use_exits_2_in_one_line(
    answering, capsys, change, message
):
    run_dir = finished_run(capsys)
    change(run_dir)
    sent = len(answering.requests)
    assert r3con("ask", str(run_dir), "Who else?") == 2
    err = capsys.readouterr().err
    assert err.startswith("r3con: ")
    assert message in err
    assert err.count("\n") == 1
    assert len(answering.requests) == sent
    assert not (run_dir / "asks").exists()


def test_an_ask_with_a_refused_override_exits_2_in_one_line_and_makes_no_folder(
    answering, capsys
):
    run_dir = finished_run(capsys)
    sent = len(answering.requests)
    assert r3con("ask", str(run_dir), "Who else?", "--model", "") == 2
    assert capsys.readouterr().err == (
        "r3con: override: `model=''`: String should have at least 1 character.\n"
    )
    assert len(answering.requests) == sent
    assert not (run_dir / "asks").exists()


def saving_params(run_dir: Path, params: dict) -> None:
    """Rewrite ``run_dir``'s manifest as it would be with ``params`` saved."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"]["params"] = params
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def test_an_ask_that_would_send_a_masked_param_exits_2_in_one_line_and_makes_no_folder(
    answering, capsys
):
    run_dir = finished_run(capsys)
    saving_params(run_dir, MASKED_HEADER)
    sent = len(answering.requests)
    assert r3con("ask", str(run_dir), "Who else?") == 2
    assert capsys.readouterr().err == (
        "r3con: REDACTED, the mask a run's manifest saves in place of a credential, "
        "would be sent: pass --api-key or --base-url in place of a masked `api_key` or "
        "`api_base`, or ask with another --model, which sends none of the run's "
        "params. Masked params: `extra_headers.Authorization`.\n"
    )
    assert len(answering.requests) == sent
    assert not (run_dir / "asks").exists()


@pytest.mark.parametrize(
    ("saved", "option", "sent"),
    [
        ({"api_key": "REDACTED"}, ["--api-key", KEY], {"api_key": KEY}),
        (
            {"api_base": "https://REDACTED@gateway.example"},
            ["--base-url", GATEWAY],
            {"api_base": GATEWAY},
        ),
        (MASKED_HEADER, ["--model", QWEN], {"extra_headers": None}),
    ],
    ids=["api-key", "base-url", "another-model"],
)
def test_an_ask_over_a_masked_param_sends_what_the_option_its_refusal_names_gives(
    answering, capsys, saved, option, sent
):
    run_dir = finished_run(capsys)
    saving_params(run_dir, saved)
    assert r3con("ask", str(run_dir), "Who else?", *option) == 0
    asked = answering.requests[-1]
    assert {name: asked.get(name) for name in sent} == sent


def test_a_failed_ask_exits_1_and_points_at_its_partial_artifacts(answering, capsys):
    run_dir = finished_run(capsys)
    down = litellm.APIConnectionError("provider down", llm_provider="openai", model="m")
    answering.answers(reasoning=down)
    assert r3con("ask", str(run_dir), "Who else?") == 1
    err = capsys.readouterr().err
    folder = path_after("artifacts: ", err)
    assert path_after("r3con: partial artifacts in ", err) == folder
    assert "provider down" in (folder / "reasoning" / "error.txt").read_text()


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


def test_an_interrupt_exits_130_and_says_how_to_carry_on(scripted, capsys):
    scripted.replies(KeyboardInterrupt())
    source = str(MEMOS / "01_northgate.txt")
    assert r3con("run", "Who?", source) == 130
    err = capsys.readouterr().err
    run_dir = path_after("artifacts: ", err)
    assert err.endswith(f"r3con: interrupted.\n{carrying_on(run_dir, source)}\n")


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
@pytest.mark.parametrize(
    "command", [["run", "Who?"], ["resume", "./no-run"]], ids=["run", "resume"]
)
def test_a_bad_setting_exits_2_naming_it_before_any_document_is_read(
    scripted, tmp_path, capsys, monkeypatch, command, argv, env, stderr
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert r3con(*command, "./missing", *argv) == 2
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


def test_a_failed_run_shows_no_key_on_stderr(answering, capsys):
    bad_key = litellm.AuthenticationError(
        f"Incorrect API key provided: {KEY}", llm_provider="openai", model="m"
    )
    answering.answers(relevance=bad_key)
    assert r3con("run", "Who?", str(MEMOS), "--api-key", KEY, "-v") == 1
    err = capsys.readouterr().err
    assert "r3con: AuthenticationError: " in err
    assert "Incorrect API key provided: REDACTED" in err
    assert KEY not in err


@pytest.mark.parametrize(
    ("params", "shown"),
    [
        (
            {"extra_headers": {"Authorization": f"Bearer {KEY}"}},
            "params={extra_headers={'Authorization': 'REDACTED'}}",
        ),
        (
            {"vertex_credentials": {"private_key": f"-----BEGIN KEY-----\n{KEY}\n"}},
            "params={vertex_credentials={'private_key': 'REDACTED'}}",
        ),
        (
            {"metadata": {"langfuse_secret_key": KEY}},
            "params={metadata={'langfuse_secret_key': 'REDACTED'}}",
        ),
    ],
    ids=["header", "escaped-in-the-label", "callback-key-in-metadata"],
)
def test_a_run_shows_its_label_with_the_credentials_in_its_params_masked(
    answering, configs, capsys, params, shown
):
    default = load_config().model_dump(exclude={"name", "overrides"})
    configs(keyed={**default, "params": params})
    assert r3con("run", "Who?", str(MEMOS), "--config", "keyed") == 0
    err = capsys.readouterr().err
    assert shown in err
    assert KEY not in err


@pytest.mark.parametrize(
    "argv",
    [["run", "Who?", "./missing"], ["ask", "./missing", "Who?"]],
    ids=["run", "ask"],
)
def test_a_refused_setting_that_quotes_a_key_shows_it_masked(
    scripted, capsys, monkeypatch, argv
):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("R3CON_DOC_WORKERS", KEY)
    assert r3con(*argv) == 2
    assert capsys.readouterr().err == (
        "r3con: R3CON_DOC_WORKERS must be an integer >= 1, got 'REDACTED'.\n"
    )


def test_r3con_and_its_cli_import_none_of_the_apps_dependencies():
    imports = (
        "import sys, r3con, r3con.cli; r3con.cli.build_parser(); "
        f"print(sorted(set({APP_MODULES!r}) & set(sys.modules)))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", imports], capture_output=True, text=True, check=True
    )
    assert loaded.stdout == "[]\n"


def unimportable(monkeypatch: pytest.MonkeyPatch, *packages: str) -> None:
    """As if ``packages`` were not installed and ``r3con.app`` not yet imported: no
    module of theirs imports, cached or not."""
    for name in [*packages, *sys.modules]:
        if any(name == p or name.startswith(f"{p}.") for p in packages):
            monkeypatch.setitem(sys.modules, name, None)
    for name in ("r3con.app", "r3con.app.server"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_the_app_without_its_extra_says_how_to_get_it_and_exits_1(monkeypatch, capsys):
    unimportable(monkeypatch, *APP_MODULES)
    assert r3con("app", "--no-browser") == 1
    assert capsys.readouterr().err == NO_APP_EXTRA


def test_a_missing_module_of_r3cons_own_is_raised_not_blamed_on_the_extra(
    monkeypatch, capsys
):
    unimportable(monkeypatch, "r3con.runs")
    with pytest.raises(ModuleNotFoundError, match="r3con.runs"):
        r3con("app", "--no-browser")
    assert capsys.readouterr().err == ""


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_the_app_on_a_busy_port_says_so_in_one_line_and_exits_2(capsys):
    with socket.create_server(("127.0.0.1", 0)) as taken:
        port = taken.getsockname()[1]
        assert r3con("app", "--no-browser", "--port", str(port)) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"r3con: can't listen on port {port}: ")
    assert "Address already in use" in err
    assert err.count("\n") == 1


def test_python_m_r3con_is_the_command_line(monkeypatch):
    # The same width for both, which argparse wraps the help to.
    monkeypatch.setenv("COLUMNS", "100")
    shown = subprocess.run(
        [sys.executable, "-m", "r3con", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert shown.stdout == build_parser().format_help()


@pytest.mark.parametrize(
    ("variable", "flag", "runs"),
    [
        (None, None, platformdirs.user_data_path("r3con") / "runs"),
        ("from-variable", None, "from-variable"),
        ("from-variable", "from-flag", "from-flag"),
    ],
    ids=["user-data", "r3con-logs-dir", "logs-dir-flag"],
)
def test_the_app_says_where_its_runs_go(monkeypatch, variable, flag, runs):
    if variable is not None:
        monkeypatch.setenv("R3CON_LOGS_DIR", variable)
    flags = [] if flag is None else ["--logs-dir", flag]
    app = subprocess.Popen(
        [sys.executable, "-m", "r3con.cli", "app", "--no-browser", *flags],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    # A line it never prints then reads as the end of its output, not as a hang.
    watchdog = threading.Timer(LAUNCH_SECONDS, app.kill)
    watchdog.start()
    try:
        url, where = app.stdout.readline(), app.stdout.readline()
    finally:
        watchdog.cancel()
        app.kill()
        app.wait()
    assert url.startswith("r3con app: http://127.0.0.1:")
    assert where == f"r3con app: runs go to {runs}\n"


@contextlib.contextmanager
def started_app(*program: str, logs_dir: Path) -> Iterator[subprocess.Popen[str]]:
    """``r3con app``, as ``program`` runs it, serving without a browser: killed on
    leaving, or after ``LAUNCH_SECONDS``, so that a hang fails the test."""
    with subprocess.Popen(
        [*program, "app", "--no-browser", "--logs-dir", str(logs_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as app:
        watchdog = threading.Timer(LAUNCH_SECONDS, app.kill)
        watchdog.start()
        try:
            yield app
        finally:
            watchdog.cancel()
            app.kill()


def client_of(app: subprocess.Popen[str]) -> httpx2.Client:
    """A client of the server ``app`` runs, with its token, from the URL it prints."""
    base_url, token = app.stdout.readline().split(": ", 1)[1].split("/#token=")
    headers = {"Authorization": f"Bearer {token.strip()}"}
    return httpx2.Client(
        base_url=base_url, headers=headers, timeout=LAUNCH_SECONDS, trust_env=False
    )


def ctrl_c(app: subprocess.Popen[str]) -> int:
    """Press Ctrl-C in ``app``'s terminal; returns its exit code once it has ended,
    which must be within ``QUIT_BOUND_SECONDS`` (``subprocess.TimeoutExpired``)."""
    app.send_signal(signal.SIGINT)
    return app.wait(QUIT_BOUND_SECONDS)


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_r3con_app_keeps_its_keys_under_r3con_never_the_mac_apps(tmp_path, monkeypatch):
    store = tmp_path / "keyring.json"
    stored = {APP_KEY_SERVICE: {"openai": KEY}, MAC_APP_KEY_SERVICE: {"gemini": KEY}}
    store.write_text(json.dumps(stored))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "tests.fakes.FileKeyring")
    monkeypatch.setenv(KEYRING_FILE, str(store))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    with (
        started_app(sys.executable, "-m", "r3con.cli", logs_dir=tmp_path) as app,
        client_of(app) as http,
    ):
        loaded = http.get("/api/keys").json()
        http.post("/api/keys", json={"provider": "anthropic", "key": KEY})
    assert (loaded["openai"]["hint"], loaded["gemini"]) == (KEY[-4:], None)
    saved = json.loads(store.read_text())
    assert saved == {**stored, APP_KEY_SERVICE: {"openai": KEY, "anthropic": KEY}}


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_ctrl_c_with_nothing_in_flight_quits_the_app_cleanly_within_its_bound(
    tmp_path,
):
    with started_app(sys.executable, "-m", "r3con.cli", logs_dir=tmp_path) as app:
        with client_of(app) as http:
            assert http.get("/api/keys").status_code == 200
        exit_code = ctrl_c(app)
        out, err = app.communicate()
    assert exit_code == 0
    assert "r3con app: quitting" not in out
    assert "Traceback" not in err


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_ctrl_c_quits_the_app_within_its_bound_though_a_model_call_never_returns(
    tmp_path,
):
    with (
        started_app(
            sys.executable, "-c", NEVER_ANSWERING_APP, logs_dir=tmp_path
        ) as app,
        client_of(app) as http,
    ):
        run_id = http.post("/api/runs", json=NEVER_ANSWERED_RUN).json()["run_id"]
        # Held open, as the page holds it: dropping its lines would close it.
        with http.stream("GET", f"/api/runs/{run_id}/events") as events:
            lines = events.iter_lines()
            assert next(lines).startswith("data: ")
            assert MODEL_CALLED in iter(app.stdout.readline, "")
            exit_code = ctrl_c(app)
        out, err = app.communicate()
    assert exit_code == FORCED_EXIT_CODE
    assert out.endswith(f"{CUT_OFF}\n")
    assert [shown for shown in CRASH_SIGNS if shown in err] == []


# --- resume ---


def stopped_run(llm, capsys, *argv: str) -> tuple[Path, str]:
    """The folder of an ``r3con run`` over the memos, with ``argv``, that ``llm``
    fails in Riverside's parse, and what it said on stderr."""
    llm.answers(parsing=failing_once(reading(RIVERSIDE), ANSWERING["parsing"]))
    assert r3con("run", "Who?", str(MEMOS), *argv) == 1
    err = capsys.readouterr().err
    return path_after("artifacts: ", err), err


def carrying_on(run_dir: Path, *sources: str, options: tuple = ()) -> str:
    """The line that says how to carry on the run in ``run_dir``."""
    command = shlex.join(["r3con", "resume", str(run_dir), *sources, *options])
    return f"r3con: to carry on: {command}"


def test_a_stopped_run_says_how_to_carry_on_and_the_command_pasted_finishes_it(
    answering, tmp_path, capsys
):
    sources = tmp_path / "q3 memos"
    shutil.copytree(MEMOS, sources)
    logs = tmp_path / "run logs"
    answering.answers(parsing=failing_once(reading(RIVERSIDE), ANSWERING["parsing"]))
    assert r3con("run", "Who?", str(sources), "--logs-dir", str(logs)) == 1
    err = capsys.readouterr().err
    run_dir = path_after("artifacts: ", err)
    assert err.splitlines()[-1] == carrying_on(run_dir, str(sources))
    pasted = shlex.split(err.splitlines()[-1].removeprefix("r3con: to carry on: "))
    sent = len(answering.requests)
    assert r3con(*pasted[1:]) == 0
    out, err = capsys.readouterr()
    assert out == f"{ANSWER}\n"
    assert path_after("artifacts: ", err) == run_dir.resolve()
    # Riverside's parse and reasoning.
    assert len(answering.requests) - sent == 2


def test_the_command_to_carry_on_keeps_its_options_and_names_but_hides_the_key(
    answering, capsys
):
    options = ("--base-url", GATEWAY, "--doc-workers", "2")
    run_dir, err = stopped_run(answering, capsys, *options, "--api-key", KEY)
    assert err.splitlines()[-1] == (
        f"{carrying_on(run_dir, str(MEMOS), options=options)} (and your --api-key)"
    )
    assert KEY not in err


def test_no_artifacts_has_no_run_to_carry_on(answering, capsys):
    answering.answers(parsing=failing_once(reading(RIVERSIDE), ANSWERING["parsing"]))
    assert r3con("run", "Who?", str(MEMOS), "--no-artifacts") == 1
    assert "to carry on" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("stopping", "code", "said"),
    [
        (bad_key(), 1, "r3con: AuthenticationError: "),
        (KeyboardInterrupt(), 130, "r3con: interrupted.\n"),
    ],
    ids=["failed", "interrupted"],
)
def test_a_stopped_resume_says_how_to_carry_on_again(
    answering, capsys, stopping, code, said
):
    run_dir, _ = stopped_run(answering, capsys)
    answering.answers(reasoning=stopping)
    assert r3con("resume", str(run_dir), str(MEMOS)) == code
    err = capsys.readouterr().err
    assert said in err
    assert err.endswith(f"{carrying_on(run_dir.resolve(), str(MEMOS))}\n")


def test_a_value_error_on_the_way_is_a_failure_not_a_refusal(answering, capsys):
    answering.answers(
        relevance=failing_once(reading(RIVERSIDE), ANSWERING["relevance"])
    )
    assert r3con("run", "Who?", str(MEMOS)) == 1
    run_dir = path_after("artifacts: ", capsys.readouterr().err)
    answering.answers(schema="Thought: none.\n<schema>\nnot a schema\n</schema>")
    assert r3con("resume", str(run_dir), str(MEMOS)) == 1
    err = capsys.readouterr().err
    assert "r3con: SchemaError: propose_schema exhausted" in err
    assert err.endswith(f"{carrying_on(run_dir.resolve(), str(MEMOS))}\n")


def resumed_to_the_end(run_dir: Path) -> None:
    assert r3con("resume", str(run_dir), str(MEMOS)) == 0


@pytest.mark.parametrize(
    ("change", "sources", "said"),
    [
        (
            resumed_to_the_end,
            [str(MEMOS)],
            "this run finished; ask it a new question instead (r3con ask, or r3con.ask).",
        ),
        (predating_resume, [str(MEMOS)], "this run predates resume"),
        (
            lambda run_dir: None,
            [str(MEMOS / "01_northgate.txt")],
            "these are not the run's documents: missing 02_eastfield.txt, ",
        ),
        (
            lambda run_dir: saving_params(run_dir, MASKED_HEADER),
            [str(MEMOS)],
            (
                "pass --api-key or --base-url in place of a masked `api_key` or "
                "`api_base`; any other masked param can be passed again only through "
                "r3con.resume's `params`. Masked params: `extra_headers.Authorization`."
            ),
        ),
        (
            reading_moved_prompts,
            [str(MEMOS)],
            "the run's prompts resolve to other files than it read",
        ),
    ],
    ids=["finished", "predates-resume", "other-documents", "masked", "prompts-moved"],
)
def test_a_resume_it_cannot_carry_on_exits_2_in_one_line(
    answering, capsys, change, sources, said
):
    run_dir, _ = stopped_run(answering, capsys)
    change(run_dir)
    capsys.readouterr()
    sent = len(answering.requests)
    assert r3con("resume", str(run_dir), *sources) == 2
    out, err = capsys.readouterr()
    assert (out, err.count("\n")) == ("", 1)
    assert err.startswith("r3con: ")
    assert said in err
    assert len(answering.requests) == sent


def test_a_resume_of_a_run_still_going_exits_2_in_one_line(answering, capsys):
    run_dir, _ = stopped_run(answering, capsys)
    with run_lock(run_dir):
        assert r3con("resume", str(run_dir), str(MEMOS)) == 2
    err = capsys.readouterr().err
    assert err.endswith(
        f"r3con: {run_dir.resolve()}: a run or resume is going in it; carry it on "
        "once that one has stopped.\n"
    )


def test_a_mistyped_run_folder_is_refused_before_any_document_is_read(
    scripted, tmp_path, capsys
):
    typo = tmp_path / "logs" / "2026111"
    assert r3con("resume", str(typo), str(tmp_path / "no documents here")) == 2
    assert capsys.readouterr().err == (
        f"r3con: {typo} is not a run folder: it has no manifest.json.\n"
    )
    assert scripted.requests == []


def test_a_resume_given_an_asks_folder_carries_on_the_run_it_asked_over(
    answering, capsys
):
    answering.answers(
        reasoning=failing_once(lambda request: True, ANSWERING["reasoning"])
    )
    assert r3con("run", "Who?", str(MEMOS)) == 1
    run_dir = path_after("artifacts: ", capsys.readouterr().err).resolve()
    assert r3con("ask", str(run_dir), "Who else?") == 0
    asked = path_after("artifacts: ", capsys.readouterr().err)
    answering.answers(reasoning=bad_key())
    assert r3con("resume", str(asked), str(MEMOS)) == 1
    err = capsys.readouterr().err
    assert path_after("artifacts: ", err) == run_dir
    assert err.endswith(f"{carrying_on(run_dir, str(MEMOS))}\n")


def test_a_resume_over_a_masked_api_key_sends_the_one_given(answering, capsys):
    run_dir, _ = stopped_run(answering, capsys)
    saving_params(run_dir, {"api_key": "REDACTED"})
    assert r3con("resume", str(run_dir), str(MEMOS), "--api-key", KEY) == 0
    assert answering.requests[-1]["api_key"] == KEY

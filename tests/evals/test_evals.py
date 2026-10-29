import hashlib
import json
import re
import shutil
import subprocess

import httpx
import litellm
import pytest
from litellm import exceptions

import r3con
from evals import JUDGE_MODEL, download
from evals.__main__ import SEED, compare, evaluate, main, sample, spend
from r3con.config import load_config
from r3con.stages.structuring.parsing import SchemaError

PINNED = {
    "seed": SEED,
    "loong": ["loong-financial", "loong-paper"],
    "corpusqa": ["sales_en_1", "sales_en_2"],
}
# A dollar per input token and nothing for output: each fake call, 10 tokens in, is $10.
LUNA_PRICE = {
    "input_cost_per_token": 1.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "openai",
    "mode": "chat",
}
USD_PER_CALL = 10
ANSWER_MODEL = "anthropic/claude-sonnet-5-5"
ASKED = "asks/anthropic-claude-sonnet-5-5-d1236105"
# What ANSWER_MODEL answers each task: the fake judge rates a Loong answer naming ACME 100.
ANSWER = "ACME, says the other model."
GIT_IDENTITY = ("-c", "user.name=evals", "-c", "user.email=evals@example.invalid")
KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
LITELLM_ERROR = {"message": "down", "llm_provider": "openai", "model": "gpt-6-luna"}
FORBIDDEN = httpx.Response(403, request=httpx.Request("POST", "https://api.openai.com"))
INFRASTRUCTURE_FAILURES = [
    exceptions.AuthenticationError(**LITELLM_ERROR),
    exceptions.PermissionDeniedError(**LITELLM_ERROR, response=FORBIDDEN),
    exceptions.APIConnectionError(**LITELLM_ERROR),
    exceptions.Timeout(**LITELLM_ERROR),
    exceptions.RateLimitError(**LITELLM_ERROR),
    exceptions.NotFoundError(**LITELLM_ERROR),
    exceptions.ServiceUnavailableError(**LITELLM_ERROR),
    exceptions.InternalServerError(**LITELLM_ERROR),
    exceptions.BadGatewayError(**LITELLM_ERROR),
    exceptions.APIError(status_code=524, **LITELLM_ERROR),
]
TASK_FAILURES = [
    exceptions.ContextWindowExceededError(**LITELLM_ERROR),
    exceptions.APIError(status_code=400, **LITELLM_ERROR),
    SchemaError("no schema fits the question"),
    RuntimeError("the model is down"),
]


@pytest.fixture(autouse=True)
def repo(tmp_path, monkeypatch):
    """The git checkout a run reads its commit and dirtiness from: a clean one."""
    root = tmp_path / "repo"
    for name in ("src/r3con.py", "evals/results.jsonl"):
        (root / name).parent.mkdir(parents=True)
        (root / name).write_text("")
    git(root, "init")
    git(root, "add", ".")
    git(root, *GIT_IDENTITY, "-c", "commit.gpgsign=false", "commit", "-m", "start")
    monkeypatch.setattr("evals.__main__.REPO_DIR", root)
    return root


def git(root, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def prices():
    """Luna and ANSWER_MODEL at LUNA_PRICE."""
    answer_price = {**LUNA_PRICE, "litellm_provider": "anthropic"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(litellm.model_cost, "gpt-6-luna", LUNA_PRICE)
        patch.setitem(litellm.model_cost, "claude-sonnet-5-5", answer_price)
        # litellm caches each model's lookup by name, so a changed price needs it cleared
        litellm.get_model_info.cache_clear()
        yield
    litellm.get_model_info.cache_clear()


@pytest.fixture
def cli_paths(tmp_path, monkeypatch):
    """Where ``main`` reads the sample and writes runs and results: under ``tmp_path``."""
    for name in ("TASKS_PATH", "RESULTS_PATH", "RUNS_DIR"):
        monkeypatch.setattr(f"evals.__main__.{name}", tmp_path / name.lower())
    return tmp_path


def judge_requests(llm) -> list[dict]:
    return [r for r in llm.requests if schema_name(r) in {"Rating", "Equivalence"}]


def schema_name(request: dict) -> str | None:
    return request.get("response_format", {}).get("json_schema", {}).get("name")


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_draw_is_reproducible_from_the_seed():
    ids = [f"task-{i}" for i in range(100)]

    assert sample(ids, 5) == sample(reversed(ids), 5)
    assert sample(ids, 5) == ["task-80", "task-73", "task-89", "task-92", "task-76"]


def test_a_full_run_scores_every_task_and_writes_its_line(
    loong_cache, corpusqa_cache, judging_llm, prices, tmp_path
):
    results = tmp_path / "results.jsonl"

    line = evaluate(PINNED, tmp_path / "run", results)

    assert line is not None
    assert line["benchmarks"] == {
        "loong": {"n": 2, "errors": 0, "avg_score": 70.0, "perfect_rate": 0.5},
        "corpusqa": {"n": 2, "errors": 0, "accuracy": 0.5},
    }
    assert line["usd"] == USD_PER_CALL * len(judging_llm.requests)
    assert line["label"] == load_config().label()
    assert line["judge_model"] == JUDGE_MODEL
    assert line["r3con_version"] == r3con.__version__
    assert len(line["commit"]) == 40
    assert [json.loads(text) for text in results.read_text().splitlines()] == [line]
    task = tmp_path / "run" / "loong" / "loong-paper"
    record = read_json(task / "record.json")
    assert (record["score"], record["error"]) == (40, None)
    assert record["answer"] == read_json(task / "answer.json")["answer"]


def test_a_partial_run_writes_no_line(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    results = tmp_path / "results.jsonl"

    assert evaluate(PINNED, tmp_path / "run", results, limit=1) is not None
    assert not results.exists()


def test_resuming_reruns_only_unfinished_tasks_and_keeps_one_line_per_run(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    folder, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, folder, results, limit=1)
    finished = {path: path.read_text() for path in folder.rglob("record.json")}

    evaluate(PINNED, folder, results)
    assert len(finished) == 2
    assert all(path.read_text() == text for path, text in finished.items())
    assert len(judge_requests(judging_llm)) == len(PINNED["loong"] + PINNED["corpusqa"])

    shutil.rmtree(folder / "corpusqa" / "sales_en_2")
    line = evaluate(PINNED, folder, results)
    requests = len(judging_llm.requests)
    evaluate(PINNED, folder, results)
    assert len(judging_llm.requests) == requests
    assert [json.loads(text) for text in results.read_text().splitlines()] == [line]


def test_a_task_the_judge_fails_has_no_record_and_is_only_rejudged_on_resume(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, monkeypatch
):
    def judge_down(**request):
        if schema_name(request) == "Rating":
            raise RuntimeError("the judge is down")
        return judging_llm(**request)

    folder, results = tmp_path / "run", tmp_path / "results.jsonl"
    monkeypatch.setattr(litellm, "completion", judge_down)
    assert evaluate(PINNED, folder, results) is None
    assert not results.exists()
    answers = {path: read_json(path) for path in folder.glob("loong/*/answer.json")}
    assert not list(folder.glob("loong/*/record.json"))

    monkeypatch.setattr(litellm, "completion", judging_llm)
    requests = len(judging_llm.requests)
    assert evaluate(PINNED, folder, results) is not None

    resumed = judging_llm.requests[requests:]
    assert [schema_name(request) for request in resumed] == ["Rating"] * 2
    for path, answered in answers.items():
        assert read_json(path.with_name("record.json"))["answer"] == answered["answer"]


@pytest.mark.parametrize("failure", TASK_FAILURES, ids=lambda e: type(e).__name__)
def test_a_task_whose_r3con_run_raises_is_recorded_with_its_error_and_scored_worst(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, failure
):
    judging_llm.answers(schema=failure)

    line = evaluate(PINNED, tmp_path / "run", tmp_path / "results.jsonl")

    assert line is not None
    assert line["benchmarks"] == {
        "loong": {"n": 2, "errors": 2, "avg_score": 1.0, "perfect_rate": 0.0},
        "corpusqa": {"n": 2, "errors": 2, "accuracy": 0.0},
    }
    record = read_json(tmp_path / "run" / "corpusqa" / "sales_en_1" / "record.json")
    assert record["error"].startswith(f"{type(failure).__name__}: ")
    assert record["answer"] == ""
    assert judge_requests(judging_llm) == []


@pytest.mark.parametrize(
    "failure", INFRASTRUCTURE_FAILURES, ids=lambda e: type(e).__name__
)
def test_a_task_the_connection_fails_saves_no_answer_and_fails_the_run(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, failure
):
    judging_llm.answers(schema=failure)
    folder, results = tmp_path / "run", tmp_path / "results.jsonl"

    assert evaluate(PINNED, folder, results) is None
    assert not list(folder.rglob("answer.json"))
    assert not results.exists()


def test_a_run_with_a_call_litellm_could_not_price_has_no_cost(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    task = tmp_path / "run" / "loong" / "loong-financial"
    task.mkdir(parents=True)
    finished = {"score": 100, "error": None, "usd": None}
    (task / "record.json").write_text(json.dumps(finished))

    line = evaluate(PINNED, tmp_path / "run", tmp_path / "results.jsonl")

    assert line is not None
    assert line["usd"] is None


def test_a_run_resumes_only_from_the_commit_and_label_it_started_with(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    folder = tmp_path / "run"
    evaluate(PINNED, folder, tmp_path / "results.jsonl", limit=1)
    started = read_json(folder / "run.json")
    (folder / "run.json").write_text(json.dumps({**started, "commit": "0" * 40}))

    with pytest.raises(SystemExit, match="was started at commit 0{40} as default"):
        evaluate(PINNED, folder, tmp_path / "results.jsonl")


def test_a_run_started_from_a_dirty_tree_never_resumes(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, repo
):
    folder, results = tmp_path / "run", tmp_path / "results.jsonl"
    (repo / "src" / "r3con.py").write_text("changed")
    assert evaluate(PINNED, folder, results, limit=1)["dirty"] is True
    git(repo, "checkout", "--", "src")

    with pytest.raises(SystemExit, match="started with uncommitted changes.*Commit"):
        evaluate(PINNED, folder, results)


def test_a_run_does_not_resume_while_the_tree_is_dirty(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, repo
):
    folder, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, folder, results, limit=1)
    (repo / "evals" / "loong.py").write_text("changed")
    git(repo, "add", "evals/loong.py")

    with pytest.raises(SystemExit, match="now has uncommitted changes.*Commit"):
        evaluate(PINNED, folder, results)


@pytest.mark.parametrize("change", ["src/scratch.py", "evals/results.jsonl"])
def test_untracked_files_and_the_results_file_leave_the_tree_clean(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, repo, change
):
    (repo / change).write_text("changed")

    line = evaluate(PINNED, tmp_path / "run", tmp_path / "results.jsonl", limit=1)

    assert line is not None
    assert line["dirty"] is False


def test_a_full_run_reads_the_results_file_first_and_stops_at_a_broken_line(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    results = tmp_path / "results.jsonl"
    results.write_text('{"run": "earlier"}\n\n{"run": "cut off\n')

    with pytest.raises(SystemExit, match="line 3 of .*results.jsonl is not JSON"):
        evaluate(PINNED, tmp_path / "run", results)
    assert judging_llm.requests == []
    assert not (tmp_path / "run").exists()


def test_blank_lines_in_the_results_file_are_skipped(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    results = tmp_path / "results.jsonl"
    results.write_text('\n{"run": "earlier"}\n\n')

    line = evaluate(PINNED, tmp_path / "run", results)

    assert results.read_text().splitlines() == ['{"run": "earlier"}', json.dumps(line)]


@pytest.mark.parametrize(
    ("costs", "total"),
    [([0.25, 0.5], 0.75), ([0.25, None], None)],
    ids=["priced", "not"],
)
def test_spend_sums_every_recorded_calls_tokens_and_cost(tmp_path, costs, total):
    tokens = {"prompt": 3, "completion": 1}
    for stage, usd in zip(("relevance", "judge"), costs, strict=True):
        (tmp_path / stage).mkdir()
        call = {"tokens": tokens, "usd": usd}
        (tmp_path / stage / "calls.json").write_text(json.dumps([call]))

    assert spend(tmp_path) == {"prompt_tokens": 6, "completion_tokens": 2, "usd": total}


def test_draw_samples_each_benchmark_without_fetching_the_document_pool(
    loong_cache, corpusqa_cache, cli_paths, monkeypatch
):
    shutil.rmtree(loong_cache / "doc")
    monkeypatch.setattr("evals.__main__.SAMPLE_SIZES", {"loong": 3, "corpusqa": 1})

    assert main(["--draw"]) == 0

    assert read_json(cli_paths / "tasks_path") == {
        "seed": SEED,
        "loong": sample([row["id"] for row in loong_rows(loong_cache)], 3),
        "corpusqa": sample(["sales_en_1", "sales_en_2"], 1),
    }


def loong_rows(cache) -> list[dict]:
    return [
        json.loads(line) for line in (cache / "loong.jsonl").read_text().splitlines()
    ]


def test_main_prints_a_partial_runs_line_and_writes_none(
    loong_cache, corpusqa_cache, judging_llm, cli_paths, capsys
):
    (cli_paths / "tasks_path").write_text(json.dumps(PINNED))

    assert main(["--limit", "1", "--workers", "2"]) == 0

    line = json.loads(capsys.readouterr().out)
    assert [group["n"] for group in line["benchmarks"].values()] == [1, 1]
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", line["run"])
    assert (cli_paths / "runs_dir" / line["run"] / "run.json").is_file()
    assert not (cli_paths / "results_path").exists()


def test_main_fails_while_a_task_has_no_record_and_resume_runs_it_again(
    loong_cache, corpusqa_cache, judging_llm, cli_paths
):
    (cli_paths / "tasks_path").write_text(json.dumps(PINNED))
    judging_llm.answers(schema=exceptions.BadGatewayError(**LITELLM_ERROR))

    assert main([]) == 1
    [folder] = (cli_paths / "runs_dir").iterdir()
    requests = len(judging_llm.requests)
    assert main(["--resume", str(folder)]) == 1
    assert len(judging_llm.requests) == 2 * requests


@pytest.mark.parametrize(
    "argv", [["--limit", "0"], ["--workers", "-2"], ["--workers", "many"]]
)
def test_limit_and_workers_must_be_positive_integers(argv):
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == 2


def test_a_download_is_kept_only_when_its_checksum_matches(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    target = tmp_path / "cache" / "tasks.jsonl"

    with pytest.raises(ValueError, match="SHA-256"):
        download(source.as_uri(), target, "0" * 64)
    assert list(target.parent.iterdir()) == []

    digest = hashlib.sha256(b"{}\n").hexdigest()
    assert download(source.as_uri(), target, digest).read_text() == "{}\n"


def test_a_recorded_error_never_holds_the_key(
    loong_cache, corpusqa_cache, judging_llm, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    echoing = {**LITELLM_ERROR, "message": f"refused for {KEY}"}
    judging_llm.answers(schema=exceptions.BadRequestError(**echoing))

    evaluate(PINNED, tmp_path / "run", tmp_path / "results.jsonl")

    record = read_json(tmp_path / "run" / "corpusqa" / "sales_en_1" / "record.json")
    assert record["error"].endswith("refused for REDACTED")
    files = [path for path in (tmp_path / "run").rglob("*") if path.is_file()]
    assert [path for path in files if KEY in path.read_text()] == []


def test_a_comparison_judges_the_models_answers_over_the_runs_and_writes_its_line(
    loong_cache, corpusqa_cache, judging_llm, prices, tmp_path, repo
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    ran = evaluate(PINNED, run, results)
    git(repo, *GIT_IDENTITY, "commit", "--allow-empty", "-m", "later")
    judging_llm.answers(reasoning=reply(ANSWER))
    requests = len(judging_llm.requests)

    line = compare(PINNED, run, ANSWER_MODEL, results)

    assert line is not None
    asked = judging_llm.requests[requests:]
    reasoning = [r for r in asked if schema_name(r) == "ReasoningAnswer"]
    assert [r["model"] for r in reasoning] == [ANSWER_MODEL] * 4
    assert line["answer_model"] == ANSWER_MODEL
    assert line["ask_of"] == {
        "run": "run",
        "commit": ran["commit"],
        "dirty": False,
        "label": ran["label"],
    }
    assert line["commit"] != ran["commit"]
    assert line["benchmarks"] == {
        "loong": {"n": 2, "errors": 0, "avg_score": 100.0, "perfect_rate": 1.0},
        "corpusqa": {"n": 2, "errors": 0, "accuracy": 0.5},
    }
    assert line["usd"] == USD_PER_CALL * len(asked)
    assert line["run"] == f"run/{ASKED}"
    assert [json.loads(text) for text in results.read_text().splitlines()] == [
        ran,
        line,
    ]
    record = read_json(run / ASKED / "loong" / "loong-paper" / "record.json")
    assert (record["answer"], record["score"], record["error"]) == (ANSWER, 100, None)
    assert (run / record["run_dir"]).parent == run / "loong" / "loong-paper"
    assert (run / record["ask_dir"]).parent == run / record["run_dir"] / "asks"


def reply(answer: str) -> dict:
    """A direct reasoning reply that answers ``answer``, citing nothing."""
    return {"answer": answer, "provenance_ids": {"summary_ids": [], "parse_ids": []}}


def test_running_a_comparison_again_asks_only_the_tasks_without_a_record(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)
    compare(PINNED, run, ANSWER_MODEL, results, limit=1)
    finished = {path: path.read_text() for path in run.glob("asks/*/*/*/record.json")}
    requests = len(judging_llm.requests)

    line = compare(PINNED, run, ANSWER_MODEL, results)

    assert len(finished) == 2
    assert all(path.read_text() == text for path, text in finished.items())
    resumed = sorted(schema_name(r) for r in judging_llm.requests[requests:])
    assert resumed == ["Equivalence", "Rating", "ReasoningAnswer", "ReasoningAnswer"]
    requests = len(judging_llm.requests)
    assert compare(PINNED, run, ANSWER_MODEL, results) == line
    assert len(judging_llm.requests) == requests
    lines = [json.loads(text)["run"] for text in results.read_text().splitlines()]
    assert lines == ["run", f"run/{ASKED}"]


def test_asking_over_a_run_that_never_finished_parsing_records_the_error(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    judging_llm.answers(schema=SchemaError("no schema fits the question"))
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)

    line = compare(PINNED, run, ANSWER_MODEL, results)

    assert line is not None
    assert line["benchmarks"] == {
        "loong": {"n": 2, "errors": 2, "avg_score": 1.0, "perfect_rate": 0.0},
        "corpusqa": {"n": 2, "errors": 2, "accuracy": 0.0},
    }
    record = read_json(run / ASKED / "corpusqa" / "sales_en_1" / "record.json")
    assert record["error"].startswith("FileNotFoundError: ")
    assert "the run never finished parsing" in record["error"]
    assert (record["answer"], record["ask_dir"]) == ("", None)
    assert [r for r in judging_llm.requests if r["model"] == ANSWER_MODEL] == []


def test_a_comparison_asks_only_over_a_finished_run(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results, limit=1)

    with pytest.raises(SystemExit, match=r"2 task\(s\) of .* have no record"):
        compare(PINNED, run, ANSWER_MODEL, results)
    assert not (run / "asks").exists()


def test_a_comparison_resumes_only_from_the_commit_it_started_with(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)
    compare(PINNED, run, ANSWER_MODEL, results, limit=1)
    started = read_json(run / ASKED / "run.json")
    (run / ASKED / "run.json").write_text(json.dumps({**started, "commit": "0" * 40}))

    with pytest.raises(
        SystemExit, match=f"at commit 0{{40}} as {ANSWER_MODEL}.*delete"
    ):
        compare(PINNED, run, ANSWER_MODEL, results)


def test_a_runs_cost_leaves_out_the_asks_over_its_tasks(
    loong_cache, corpusqa_cache, judging_llm, prices, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    line = evaluate(PINNED, run, results)
    for manifest in run.glob("*/*/*/manifest.json"):
        r3con.ask(manifest.parent, "Asked again?")
    for record in run.glob("*/*/record.json"):
        record.unlink()

    assert len(list(run.glob("*/*/*/asks/*/reasoning/calls.json"))) == 4
    assert evaluate(PINNED, run, results)["usd"] == line["usd"]


def test_main_asks_over_a_run_and_prints_the_comparisons_line(
    loong_cache, corpusqa_cache, judging_llm, cli_paths, capsys, monkeypatch
):
    (cli_paths / "tasks_path").write_text(json.dumps(PINNED))
    assert main([]) == 0
    run = json.loads(capsys.readouterr().out)["run"]
    monkeypatch.chdir(cli_paths / "runs_dir" / run)

    assert main(["--ask-over", ".", "--model", ANSWER_MODEL]) == 0

    assert json.loads(capsys.readouterr().out)["ask_of"]["run"] == run
    results = (cli_paths / "results_path").read_text().splitlines()
    assert [json.loads(text)["run"] for text in results] == [run, f"{run}/{ASKED}"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", ANSWER_MODEL],
        ["--ask-over", "run"],
        ["--ask-over", "run", "--model", ANSWER_MODEL, "--resume", "run"],
    ],
    ids=["model-alone", "no-model", "with-resume"],
)
def test_ask_over_takes_a_model_and_no_resume(argv):
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == 2


def test_a_model_whose_provider_litellm_does_not_know_stops_the_comparison(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)
    requests = len(judging_llm.requests)

    with pytest.raises(SystemExit, match="no provider for --model antropic/claude"):
        compare(PINNED, run, "antropic/claude-sonnet-5-5", results)
    assert len(judging_llm.requests) == requests
    assert not (run / "asks").exists()


def test_a_model_the_provider_does_not_know_saves_no_answer_and_fails_the_comparison(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    ran = evaluate(PINNED, run, results)
    judging_llm.answers(reasoning=exceptions.NotFoundError(**LITELLM_ERROR))

    assert compare(PINNED, run, ANSWER_MODEL, results) is None
    assert not list((run / ASKED).rglob("answer.json"))
    assert [json.loads(text) for text in results.read_text().splitlines()] == [ran]


def test_each_model_has_its_own_comparison_even_when_their_names_look_alike(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)

    for model in ("openai/gpt-4.1", "openai/gpt-4-1"):
        assert compare(PINNED, run, model, results, limit=1) is not None

    started = {
        path.parent.name: read_json(path) for path in run.glob("asks/*/run.json")
    }
    assert {name: info["answer_model"] for name, info in started.items()} == {
        "openai-gpt-4-1-c8893463": "openai/gpt-4.1",
        "openai-gpt-4-1-16cdc59d": "openai/gpt-4-1",
    }


def test_a_moved_run_still_prices_its_asks_when_judged_again(
    loong_cache, corpusqa_cache, judging_llm, prices, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)
    line = compare(PINNED, run, ANSWER_MODEL, results)
    moved = tmp_path / "moved"
    shutil.move(run, moved)
    for record in moved.glob(f"{ASKED}/*/*/record.json"):
        record.unlink()

    assert compare(PINNED, moved, ANSWER_MODEL, results)["usd"] == line["usd"]


def test_a_task_whose_r3con_run_is_gone_is_recorded_as_an_error_when_asked(
    loong_cache, corpusqa_cache, judging_llm, tmp_path
):
    run, results = tmp_path / "run", tmp_path / "results.jsonl"
    evaluate(PINNED, run, results)
    [manifest] = (run / "loong" / "loong-paper").glob("*/manifest.json")
    shutil.rmtree(manifest.parent)

    line = compare(PINNED, run, ANSWER_MODEL, results)

    assert line["benchmarks"]["loong"]["errors"] == 1
    record = read_json(run / ASKED / "loong" / "loong-paper" / "record.json")
    assert record["error"].startswith("FileNotFoundError: ")
    assert record["error"].endswith("loong-paper holds no r3con run")


@pytest.mark.parametrize(
    ("argv", "refusal"),
    [
        (["--resume", "{comparison}"], "is a comparison; finish it with {finish}"),
        (
            ["--ask-over", "{comparison}", "--model", ANSWER_MODEL],
            "is a comparison; finish it with {finish}",
        ),
        (
            ["--ask-over", "{elsewhere}", "--model", ANSWER_MODEL],
            "is not a run: it has no run.json with a label",
        ),
    ],
    ids=["resume-a-comparison", "ask-over-a-comparison", "ask-over-no-run"],
)
def test_a_comparison_or_a_folder_that_is_no_run_is_refused_with_what_to_do(
    cli_paths, argv, refusal
):
    (cli_paths / "tasks_path").write_text(json.dumps(PINNED))
    run = cli_paths / "runs_dir" / "run"
    comparison = run / ASKED
    comparison.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"label": "default[...]"}))
    (comparison / "run.json").write_text(json.dumps({"answer_model": ANSWER_MODEL}))
    paths = {"comparison": comparison, "elsewhere": cli_paths / "elsewhere"}
    finish = f"--ask-over {run} --model {ANSWER_MODEL}"

    with pytest.raises(SystemExit, match=re.escape(refusal.format(finish=finish))):
        main([arg.format(**paths) for arg in argv])

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
from evals.__main__ import SEED, evaluate, main, sample, spend
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
LITELLM_ERROR = {"message": "down", "llm_provider": "openai", "model": "gpt-6-luna"}
FORBIDDEN = httpx.Response(403, request=httpx.Request("POST", "https://api.openai.com"))
INFRASTRUCTURE_FAILURES = [
    exceptions.AuthenticationError(**LITELLM_ERROR),
    exceptions.PermissionDeniedError(**LITELLM_ERROR, response=FORBIDDEN),
    exceptions.APIConnectionError(**LITELLM_ERROR),
    exceptions.Timeout(**LITELLM_ERROR),
    exceptions.RateLimitError(**LITELLM_ERROR),
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
    identity = ("-c", "user.name=evals", "-c", "user.email=evals@example.invalid")
    git(root, *identity, "-c", "commit.gpgsign=false", "commit", "-m", "start")
    monkeypatch.setattr("evals.__main__.REPO_DIR", root)
    return root


def git(root, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def luna_price():
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(litellm.model_cost, "gpt-6-luna", LUNA_PRICE)
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
    loong_cache, corpusqa_cache, judging_llm, luna_price, tmp_path
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

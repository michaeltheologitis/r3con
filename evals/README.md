# evals

A small regression check for r3con, not the paper's evaluation: it runs `r3con.run` with
the default config on 75 pinned tasks, judges each answer, and writes one line of scores
to [`results.jsonl`](results.jsonl), so the numbers can be compared commit to commit.

- **Loong**: 40 tasks drawn from all 1,600 (English and Chinese; financial, paper and
  legal; every task type and length set). Each answer is rated 1–100 against the gold:
  `avg_score` is the mean rating and `perfect_rate` the share rated 100.
- **CorpusQA**, 128k tier: 35 tasks drawn from all 329. Each answer is judged equivalent to
  the gold or not: `accuracy` is the share judged equivalent.

The judge is `openai/gpt-6-luna` on default settings. The loaders, judge rubrics and task
adapters are ported from [r3con-evaluation](https://github.com/michaeltheologitis/r3con-evaluation)
(commit `d71a54a`); each module's docstring says what changed.

## Cost and time

About **$3 a run** on Luna (the first full run, on 2026-10-28, cost $2.99): the sample
holds 7.3M document tokens, r3con sends about 3.2 times that as input, and output and
judging are small. The cost is litellm's price
for each call r3con and the judge made, as recorded in their `calls.json`, cached input
included; it is `null` when litellm could not price a call.

A task takes one to two minutes, and a run does 32 at a time by default (`--workers`),
each fanning out over its own documents; the first full run took about three minutes.
Rate-limited calls are resent with backoff by r3con itself.

## Running it

From the repo root, with `OPENAI_API_KEY` set:

```bash
uv run python -m evals                       # the full run; writes its line to results.jsonl
uv run python -m evals --limit 1             # one task per benchmark, to try it out
uv run python -m evals --resume evals/runs/<folder>
```

The first run downloads the data (about 150 MB) into `evals/.cache/` and checks each file's
SHA-256. Each run writes to `evals/runs/<UTC time>_<hex>/`: `run.json` (the commit and run
label it started with), then per task `<benchmark>/<task id>/` with `answer.json` (r3con's
answer, or its error), r3con's run folder, the judge's `judge/calls.json`, and
`record.json` (the score, the judge's rationale, tokens and cost). Both folders are
git-ignored.

A task whose r3con run raises is recorded with its error and scored worst (Loong 1,
CorpusQA 0). A task gets no record instead when the connection fails (authentication,
permission, connection, timeout, rate limit or a provider's server error 500, 502 or
503, once r3con's own resends are spent; other server errors are scored), or when
loading the task or judging its answer raises: the run logs it, exits non-zero and
writes no line, and `--resume` retries it, judging a saved answer again rather than
running r3con twice. To retry a recorded task, delete its folder and resume.

A run resumes only from the commit and run label it started with, and only from a clean
tree: if `src/` or `evals/` had uncommitted changes to tracked files when it started, or
has them now, the checkout may not be the code it ran, so commit first and start a new
run. Ctrl-C cancels the tasks still queued; tasks already running finish first, since a
thread cannot be stopped mid-call.

Only a full run (no `--limit`) writes a line, and a run has one line: finishing or
retrying it replaces its earlier line. A full run reads `results.jsonl` before any task
starts and stops at a line that is not JSON, so a broken file never costs a paid run.

## Reading `results.jsonl`

One JSON object per full run:

| Field | Meaning |
| --- | --- |
| `date`, `commit`, `dirty` | When the run started (UTC), its commit, and whether tracked files under `src/` or `evals/` (other than `results.jsonl`) had uncommitted changes |
| `r3con_version`, `label` | The package version, and the run label: the config name, model, relevance rounds, reasoning path, generation params when set, and every prompt version |
| `judge_model` | The judge |
| `benchmarks.loong` | `n`, `errors`, `avg_score` (1–100), `perfect_rate` (0–1) |
| `benchmarks.corpusqa` | `n`, `errors`, `accuracy` (0–1) |
| `usd` | What the run's calls cost, or `null` if litellm could not price one |
| `run` | The run's folder under `evals/runs/`, on the machine that ran it |

With 40 and 35 tasks, a difference of a few points is noise; look for large moves.

## The sample

The tasks in [`tasks.json`](tasks.json) are random, not hand-picked:
`random.Random(20261028).sample(sorted(all_ids), n)` for each benchmark, with n = 40 and
35. Never edit the list by hand; `uv run python -m evals --draw` draws it again, and
`git diff evals/tasks.json` should then show nothing.

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
included, but not the asks over a task's r3con run (its `asks/`); it is `null` when
litellm could not price a call.

A task takes one to two minutes, and a run does 32 at a time by default (`--workers`),
each fanning out over its own documents; the first full run took about three minutes.
Rate-limited calls are resent with backoff by r3con itself.

## Running it

From the repo root, with `OPENAI_API_KEY` set:

```bash
uv run python -m evals                       # the full run; writes its line to results.jsonl
uv run python -m evals --limit 1             # one task per benchmark, to try it out
uv run python -m evals --resume evals/runs/<folder>
uv run python -m evals --ask-over evals/runs/<folder> --model <model>   # see below
```

The first run downloads the data (about 150 MB) into `evals/.cache/` and checks each file's
SHA-256. Each run writes to `evals/runs/<UTC time>_<hex>/`: `run.json` (the commit and run
label it started with), then per task `<benchmark>/<task id>/` with `answer.json` (r3con's
answer, or its error), r3con's run folder, the judge's `judge/calls.json`, and
`record.json` (the score, the judge's rationale, tokens and cost). Both folders are
git-ignored.

A task is left without a record when its r3con run raises an error r3con resends (a
failed connection, a server error, a timeout or a rate limit) once its resends are
spent, or a refused key, access or quota, and when loading the task or judging its
answer raises: the run logs it, exits non-zero and writes no line, and `--resume`
retries it, judging a saved answer again rather than running r3con twice. Every other
error is recorded with the task and scored worst (Loong 1, CorpusQA 0); to retry a
recorded task, delete its folder and resume.

A run resumes only from the commit and run label it started with, and only from a clean
tree: if `src/` or `evals/` had uncommitted changes to tracked files when it started, or
has them now, the checkout may not be the code it ran, so commit first and start a new
run. Ctrl-C cancels the tasks still queued; tasks already running finish first, since a
thread cannot be stopped mid-call.

Only a full run (no `--limit`) writes a line, and a run has one line: finishing or
retrying it replaces its earlier line. A full run reads `results.jsonl` before any task
starts and stops at a line that is not JSON, so a broken file never costs a paid run.

## Comparing answering models

`--ask-over` answers a finished run's tasks again with another model, over the same
representations: per task, `r3con.ask` reruns only reasoning over the task's r3con run
folder, with the run's question, and the answer is judged as in a run.

```bash
uv run python -m evals --ask-over evals/runs/<folder> --model anthropic/claude-sonnet-5-5
```

It writes to `<folder>/asks/<model>/` (`/` and other punctuation as `-`), laid out as a
run folder; each `answer.json` also names `ask_dir`, where r3con wrote the ask (in the
`asks/` of the task's r3con run). Errors are kept as in a run, and a task whose r3con
run never finished parsing cannot be asked, so it is recorded with that error. Every
task asked over needs a record in the run. Running the command again finishes the
comparison; it resumes only from the commit it started with and a clean tree, and to
start it over, delete its folder. `--limit` and `--workers` work as for a run.

The asks run this checkout's reasoning over what the run's commit built, so the line
names both commits. On Luna a comparison costs about 2% of a run (for the run of
2026-10-28 at `c6c8e60`, reasoning cost $0.06 and judging $0.01 of $3.06); on another
model, its price for about 340k input and 30k output tokens.

## Reading `results.jsonl`

One JSON object per full run, and one per full comparison:

| Field | Meaning |
| --- | --- |
| `date`, `commit`, `dirty` | When the run or comparison started (UTC), its commit, and whether tracked files under `src/` or `evals/` (other than `results.jsonl`) had uncommitted changes |
| `r3con_version`, `label` | The package version, and the run label: the config name, model, relevance rounds, reasoning path, generation params when set, and every prompt version. A comparison has no `label` |
| `ask_of`, `answer_model` | A comparison's only: the run it asked over (its `run` folder, `commit` and `label`), and the model that answered |
| `judge_model` | The judge |
| `benchmarks.loong` | `n`, `errors`, `avg_score` (1–100), `perfect_rate` (0–1) |
| `benchmarks.corpusqa` | `n`, `errors`, `accuracy` (0–1) |
| `usd` | What the run's calls cost, or a comparison's asks and judging; `null` if litellm could not price one |
| `run` | The folder under `evals/runs/`, on the machine that ran it: `<run>` or `<run>/asks/<model>` |

With 40 and 35 tasks, a difference of a few points is noise; look for large moves.

## The sample

The tasks in [`tasks.json`](tasks.json) are random, not hand-picked:
`random.Random(20261028).sample(sorted(all_ids), n)` for each benchmark, with n = 40 and
35. Never edit the list by hand; `uv run python -m evals --draw` draws it again, and
`git diff evals/tasks.json` should then show nothing.

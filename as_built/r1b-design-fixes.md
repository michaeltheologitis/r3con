# TASK-40 · R1b as built: r3con's design fixes (r3context 0.2.0)

Cartographer, for
[TASK-40](https://app.notion.com/p/R1b-r3con-s-design-fixes-settings-honoured-the-model-s-schema-in-the-sandbox-a-failed-stage-say-3ef62fb222378165a493f2aaba2fc13f).
This is what exists at `243f647` on `claude/tender-shannon-eq4lq7-r1b`, checked against
`design/r1b-design-fixes.md` at that commit. The base is `main` at `43e2687`, merged in
as `623d21b`; the two trees are identical.

**Evidence.**

- *[run]*: I executed it. That means the suite, an offline probe (a fake completion, or
  litellm against a server on 127.0.0.1), or a CI run whose artifact I downloaded and
  read.
- *[read]*: I read it in the code and did not execute it.

Nothing I ran reached a provider.

**Reading it.**

- §1: what a user of 0.1.1 sees differently.
- §2: the map of the code.
- §3: the divergences from the design, two of them new.
- §4: the measured results.
- §5: what I could not verify.

## 1 · What a 0.1.1 user sees in 0.2.0

**Now works**

- **Anthropic and Gemini models run.** 0.1.1 sent `seed` on every call, and litellm
  refuses it for those providers before sending anything. 0.2.0 sends a seed only if the
  caller puts one in `params` [run: through litellm's own parameter check].
- **Setting a cap takes effect.** `r3con.settings.REASONING_MAX_TURNS = 3` bounds every
  later run. The manifest records the caps the run actually used, including an explicit
  `max_reasoning_turns` [run].
- **A model that refuses `stop` no longer fails the reasoning stage.** This is a model
  newer than litellm's map. The turn is asked again without `stop`, and the rest of the
  loop goes without it.
  - **Cost, every run.** litellm sends the refused request 11 times
    (`LLM_NUM_RETRIES = 10`) and prints its "Give Feedback / Get Help" banner to stderr
    each time [run: local server].
- **`timeout_s=0.5` means 0.5 s.** 0.1.1 truncated it to 0, so every program timed out
  [run]. A timed-out program still holds the loop until it ends (§3.2 item 2).
- **The model's schema runs in the restricted interpreter.** `import os`, `open(...)`
  and `__import__` are refused, and the model retries [run].

**Refused before any request and before a run folder exists**

- **A misspelt override** raises a `TypeError` that names it [run].
- **A config file field r3con does not read** raises a `ValueError` that lists the
  allowed fields. For `seed: 7` the message adds: `` `seed` was removed in r3con 0.2.0;
  to send one, put it in `params` (params: {seed: 7}). `` [run]
- **A pinned prompt version with no file** raises `FileNotFoundError` on load, naming
  the stage, the version and both places looked [run]. A `RunConfig` passed to `run()`
  is the exception (§3.2 item 1).
- **A cap below its floor** raises `ValueError: DOC_WORKERS must be >= 1, got 0.`, and
  the CLI prints it and exits 2 [run]. `--doc-workers 0` and `R3CON_DOC_WORKERS=0`, which
  meant "one at a time", are refused; 1 now means that.
- **A bad config in the CLI** gives `r3con: <message>` and exit 2, not a traceback
  [read; the missing-prompt case run by its test].

**Fails differently mid-run**

- **A stage that raises now leaves a trace.** It writes `<stage>/calls.json` (the calls
  that completed) and `<stage>/error.txt`, and no `result.json`. In 0.1.1 only reasoning
  wrote `error.txt`. The exception carries the note
  `r3con: partial artifacts in <run folder>`, which the traceback and the CLI print
  [run].
- **A failing document stops the documents not yet started**, in relevance and in
  parsing. The lowest-indexed failure is raised after the started calls finish [run].
- **More schemas are refused and retried.** These are schemas with a decorator, a nested
  `class Config:`, `from __future__ import annotations`, or an import from outside
  pydantic, typing, datetime, enum, decimal and the interpreter's base modules. 0.1.1 ran
  them with `exec` [run].

**Breaks 0.1.1 code** (each of these now raises)

- **Seed.** `run(seed=)`, `load_config(seed=)`, `r3con run --seed`, `RunConfig(seed=)`
  and `.seed`, and `StageRun(seed=)`. Rebuilding a `RunConfig` from a 0.1 manifest's
  `config` block also raises [run].
- **Settings.** `from r3con.settings import settings` raises; `r3con.settings.X` works
  [run].
- **`load_config`** takes `model`, `relevance_rounds` and `params` by keyword only. A
  non-mapping `params` raises `TypeError`, where 0.1.1 raised `ValueError` [run].
- **Removed.** `render_relevance(doc_ids=)`, `parse_documents(doc_ids=)`,
  `ParseResult.doc_ids` and `.doc_label()`, `r3con.version` and
  `r3con.PackageNotFoundError` [read; the last two run].
- **Labels** lose `seed=42`, so a 0.1 label never matches a 0.2 label.
  `params={"seed": 42}` shows as `params={seed=42}` [run].

**Unchanged**

- No prompt file is in the diff [run].
- A successful run's folder has the same files and keys as in 0.1.1, minus `seed` in the
  manifest's `config` block and in the transcript headers [run: the two live artifacts
  compared].
- Model-facing text changes only after a refused schema attempt, and in the timeout
  observation [read].

## 2 · How it is built

**Size.**

- `src/` changes by +574/−542 lines over 17 files. r3con's own Python goes from 3,761
  to 3,794 lines; the vendored executor is untouched.
- `tests/` changes by +643/−82 lines.

**Where the complexity is.**

- The refuse-first checks are ordered across three entry points (`run()`, the CLI,
  `run_pipeline`). This is where the one behavioural gap sits (§3.2 item 1).
- `_recorded_stage` is the helper every stage now goes through.

### 2.1 Caps: `settings.settings_snapshot`

`settings.py` is a module of constants (`DOC_WORKERS`, `REASONING_MAX_TURNS`, …). One
function reads and checks a run's caps:

```python
def settings_snapshot(*, reasoning_max_turns: int | None = None) -> dict[str, int]: ...
```

It reads the seven caps when it is called. An explicit turn cap replaces the setting. It
raises naming the first cap below its floor, using `_FLOORS`: 1 for the worker, turn and
attempt caps, 0 for the other three.

- **`run()` and the CLI** call it before creating their `TaskLogger`, and discard the
  result.
- **`run_pipeline`** calls it again, after `check_prompts`. It records the result
  through `write_manifest(settings=caps)` and passes four caps to the stages. The other
  three (`LLM_NUM_RETRIES`, `LLM_EMPTY_CONTENT_RETRIES`, `REASONING_PARSE_MAX_TOKS`) are
  read again where each call runs.

So through `run()` and the CLI the caps are read twice, and the manifest holds the second
read. Stage functions called directly take `None` and read `settings.X` when they run.
No cap is bound in a default argument [run: grep, and each cap's test].

### 2.2 Config: refused on load, prompts checked twice

```python
def load_config(
    name: str = "default",
    *,
    model: str | None = None,
    relevance_rounds: int | None = None,
    params: Mapping[str, Any] | None = None,
) -> RunConfig: ...
```

`load_config` refuses, in this order:

1. an unknown name (`KeyError`);
2. a field outside `{model, relevance_rounds, prompts, params}` (`ValueError`);
3. a missing `model`;
4. a non-mapping `params` (`TypeError`), checked as the `RunConfig`
   (`extra="forbid"`) is built.

It then applies the overrides, and calls `config.check_prompts` last.

`check_prompts` refuses a stage with no pin, then calls `prompts.require_prompt_path` for
each stage. `load_prompt` uses that same function, so the message has one source.
`run_pipeline` calls `check_prompts` first. `run()` given a `RunConfig` object never calls
`load_config`, so for it the only check is the one in `run_pipeline` [read; run by
probe].

### 2.3 Stage records: `pipeline._recorded_stage`

All four stages run inside this one context manager.

- **The body** passes `record.run` to the stage and fills `record.result`.
- **On a clean exit** it flushes the calls (and `transcript.yaml` for schema and
  reasoning), then writes `result.json` with `totals` last.
- **On any `BaseException`** it flushes the calls, writes `error.txt`, adds the note and
  re-raises.
- **Without a logger** it records nothing and adds no note.

A call is recorded only after it returns, so the failing call is never in `calls.json`.
The CLI prints the error and then each note [read; run by tests and the CLI probe].

### 2.4 The model's schema: `stages/structuring/parsing.check_schema`

`check_schema(schema_code: str) -> type[BaseModel]` runs four steps:

1. `_decorators` walks the AST and refuses any decorator, listing each one (e.g.
   `@field_validator('n') on pos, @classmethod on pos`) and saying to use `Field(...)`
   or a type instead.
2. A fresh `LocalPythonExecutor` runs the code. It may also import pydantic, typing,
   datetime, enum and decimal, uses its default 30 s timeout, and has the common
   pydantic and typing names pre-bound.
3. An `InterpreterError` or `ExecutionTimeoutError` becomes
   `SchemaError("Schema code failed to execute: …")`, the text `propose_schema` feeds
   back to the model.
4. The checks on the result run on `executor.state`: `Parse` exists, is a `BaseModel`,
   rebuilds, has no untyped objects, and has only list fields.

I tried six routes from an allowed module to `os`: `pydantic.main.sys`, `typing.sys`,
`enum.sys`, `random._os`, `from pydantic import main`, and a function's `__globals__`.
All were refused, and no marker file was written [run]. That is a probe, not a proof.

### 2.5 The reasoning loop: `runtime/codeact.run_codeact`

**`stop`.** A local `stop` starts as `_STOP_SEQUENCES` when the caller passed no `stop`
and litellm's map says the model takes it.

- **When it is dropped.** A turn may raise `litellm.exceptions.BadRequestError` whose
  text matches `['"]stop['"]` while `stop` is set. The loop then logs at INFO, sets
  `stop = None`, and re-sends the same turn. No turn is spent.
- **What is never dropped or retried.** A caller's own `stop` is never dropped. Any
  other 400 is raised: `temperature`, context window, or the bare word "stop".
- **How long it lasts.** The memory lasts one `run_codeact` call.

**Timeout.** `timeout_s` reaches the executor as the float it is given, with a pyright
ignore on the vendored `int` annotation [read; run by tests and the local-server probe].

### 2.6 The smaller seams

- **`parallel_map`** is `list(ex.map(...))`. With one worker or one item, the calls run
  in the calling thread.
- **Requests** carry what `config.params` and the transport set, so they carry a seed
  only when `params` has one. The empty-reply re-roll re-sends the same request, and a
  `seed=` passed to `runtime/llm.py` reaches the provider through `**kwargs`.
- **`stages/reasoning.py`.**
  - It dumps the parse to JSON once, and that string serves both `parse_block` and
    `parse_json`.
  - A tokenizer that cannot load (`OSError` or `ValueError`) falls back to a
    4-characters-per-token estimate, with one warning.
- **`_merge_with_source_docs`** concatenates every field, since `check_schema` admits
  only list fields.
- **The manifest** reads `r3con.__version__`.
- **`noqa` markers.** Two remain in `src/` [run: grep].
- **Docstrings** say what a thing does, not who calls it [read, not checked site by
  site].
- **`examples/options.ipynb`** passes its seed in `params`, and cell 5's stored output
  shows labels without a seed [read; not executed].

## 3 · Divergences from the design

### 3.1 Recorded by the Implementer, and true in the code

The design's §3.1 items 3 and 4 (the Conductor's rulings) and §3.3 items 1 to 4 hold at
`243f647`:

- **The cap ruling.** A cap below its floor is refused before the folder exists [run].
- **The seed message** [run].
- **`_refuses_stop`** is the regex alone, and the loop catches
  `litellm.exceptions.BadRequestError` [read].
- **tiktoken** is imported at module level, so `import r3con` loads it [read].
- **The three guard tests** pass before their fix [run, §4.2].
- **The added tests** exist and pass [run].

I did not verify §3.3 item 5 (notebook cell 5 run alone).

### 3.2 Not recorded: found in the build

1. **`run()` given a `RunConfig` that pins a missing prompt leaves an empty run folder.**
   - **What the design claims.**
     - Design §4.4: through `run()` and the CLI, "a refused cap or prompt leaves no
       folder".
     - §18: only a caller with its own `TaskLogger` gets an empty folder.
     - `run()`'s docstring: the `FileNotFoundError` comes "before a run folder exists".
   - **What the build does.** `run()` checks prompts only through `load_config`, which it
     skips for a `RunConfig`. `run_pipeline` then refuses it after the folder is made.
   - **Measured** [run]. The default config with `reasoning: v9`, passed as an object,
     leaves one empty folder. The same pin passed by config name leaves none.
   - A missing stage pin takes the same path [read]. No request is sent either way.
   - I added this to the design as §3.3 item 6.
2. **A program that times out still holds the loop until it ends.**
   - **Why.** The vendored `timeout` runs the program inside
     `with ThreadPoolExecutor(...)`, whose exit waits for it.
   - **What it means.** The model is told "Program timed out after 0.5s", but
     `run_codeact` returns only when the program stops. The same holds for the schema
     check's 30 s.
   - **Measured** [run]. `time.sleep(2)` under a 0.5 s limit raises after 2.00 s.
   - **Scope.** This predates R1b and R1.14 does not change it.
   - **What it contradicts.** Design §1 and §11 say `timeout_s=0.5` means half a second.
     That is true of the message, not of wall-clock time.
   - I added this to the design as §3.3 item 7.
3. **A test the design replaces is kept, renamed.**
   `test_the_configs_params_and_seed_are_in_every_request` is now
   `test_the_configs_params_are_in_every_request`, and it sits beside
   `test_no_request_carries_a_seed_unless_params_sets_one`. Design §10.3 says the new
   test replaces it. This is a test-only difference, and I did not add it to the design.

### 3.3 Known, left as they are

- **The allowed-import list changes order between processes.**
  - **Cause.** `python_executor.py:1780` builds it from a `set`.
  - **Measured** [run]. With `PYTHONHASHSEED` 1 and 2, the same 15 modules print in
    different orders.
  - **Effect.** The model reads this list after a refused import, so two otherwise
    identical runs can send different retry prompts. That was already true in reasoning,
    and is new in the schema check.
- **A `RunConfig` rebuilt from a 0.1 manifest** is refused, with `ValidationError … seed`
  [run].
- **The caps are read twice** through `run()` and the CLI (§2.1) [read].

### 3.4 History

- **The release commit was force-pushed once, with a lease.** The old head `a6ea5bb`
  has the same parent and tree as `5e39d25`; only the message differs [run]. Its CI run,
  `37270317484`, was cancelled.
- **The R1.9 commit `8814eb9` dropped a test.** It dropped R1.12's
  `test_a_config_that_cannot_render_its_prompts_is_refused_before_any_request`, and
  `ac4f3ba` restores it unchanged. In a PR split, that test lands after R1.9.

## 4 · Measured results

### 4.1 The suite and CI at `243f647`

- **Locally** [run]. `uv run --locked pytest`: 317 passed and 1 deselected (live) in
  16.8 s, on Python 3.13 with litellm 1.104.0. The base collects 261.
- **Push CI `37270807733`** [opened]. All 7 jobs green: `checks` (ruff, ruff format,
  pyright), `test` on 3.11 to 3.14, `lowest` and `wheel`. `live` is skipped on push.
- **Dispatched run `37270911857`** [opened]. The same 7 jobs plus `live`, all green.

### 4.2 E1: each new test against the code before its fix

**How it was run** [run]. For each item's commit `C`, I took the test functions `C` adds
and ran them on `C`'s tests with the parent's `src/`:
`git checkout C^ -- src && uv run --locked pytest -k "<added names>"`, offline.

| Item | Commit | Fail at the parent | Pass at the parent, and why |
| --- | --- | --- | --- |
| R1.7 | `5efe3bf` | 18 of 18 | — |
| R1.13 | `1f5ff5f` | 5 of 7 | same-request re-roll (design §3.3 item 3); the renamed params test (§3.2 item 3) |
| R1.10 | `082ed76` | 7 of 7 | — |
| R1.12 | `49defe1` | 5 of 6 | the label test, changed to add a v9 overlay |
| R1.9 | `8814eb9` | 6 of 7 | no note without a logger (design §3.3 item 3) |
| R1.8 | `3c9c81b` | 6 of 7 | plain methods and rich types (guard) |
| R1.11 | `b1cf910` `9d3bd42` `814d9ef` `ac919d7` | 3 of 8 | lowest-indexed failure (guard); byte-identical prompt (guard); 3 render rows, whose `doc_ids` rows went |
| R1.14 | `b6f287a` | 1 of 1 | — (the parent says "Program timed out after 0.5s") |
| R1.15 | `ebe09fa` | 1 of 5 | 4 guards: temperature, context window, bare "stop", the caller's `stop` |

Two more checks:

- The changed CLI test `test_a_provider_failure_exits_1_and_points_at_the_partial_artifacts`
  fails before R1.9, on the missing `relevance/error.txt`.
- The restored R1.12 test fails before `49defe1`.

**Null not shown.** Every test that passes at its parent is a named guard, a changed
test, or the rename.

### 4.3 E8: does the schema in the interpreter keep the answer?

**Conditions.** The live tier: the default config, `gpt-6-luna`, 2 rounds, v1 prompts,
and the 5 memos in `examples/memos`. Reasoning committed on turn 1 in both runs. Both
`live-run` artifacts were downloaded and read [run].

| Run | Commit | Manifest | Schema attempts | Answer | Assertion |
| --- | --- | --- | --- | --- | --- |
| Baseline, PR #13 `37266076633` | `78b3585` | 0.1.1, `seed: 42` | 1, no error | "Halloran Services Ltd's sites logged the most equipment incidents in Q3, with 11 total." | passed |
| R1b `37270911857` | `243f647` | 0.2.0, no seed | 1, no error | "Halloran Services Ltd had the most Q3 equipment incidents, with 11 in total." | passed |

- **Null not shown.** The schema needed no more attempts than the baseline, and the
  assertion held.
- **Runs per arm.** There is one run per arm. The wording varies between runs; the
  asserted facts (Halloran, 11) do not.
- **Artifact digests.** `d3f44c9d…` for R1b and `66ebd841…` for the baseline.
- **To reproduce.** Dispatch `ci.yml` on the branch with `live` true. It bills
  `OPENAI_API_KEY` and runs `uv run --locked pytest -m live -rA`. Then compare
  `len(structuring/schema/result.json["attempts"])`.
- **The offline half is the System Designer's,** run at design time and not re-run.
  1,918 of 1,920 archived schemas give identical JSON schemas under `exec` and the
  interpreter, and the other 2 are refused by both.

### 4.4 Other probes (offline, at `243f647`)

| Probe | Conditions | Result |
| --- | --- | --- |
| `stop` refused | `run_codeact` on `openai/gpt-4o` through litellm 1.104.0. A 127.0.0.1 server answers 400 "Unsupported parameter: 'stop'…" to a body with `stop`. | 11 requests with `stop`, 1 without. Answer `ok` in 1.0 s, with 11 litellm banners on stderr. |
| Cancellation | `parallel_map`: 10 items, 2 workers, item 0 raises, the others sleep 0.2 s. Three runs. | 3, 3 and 2 calls started; 0.20 s each |
| Timeout | `LocalPythonExecutor(timeout_seconds=0.5)` running `time.sleep(2)` | raised after 2.00 s |
| Failing stage via the CLI | `--base-url http://127.0.0.1:9` (a closed port) | Exit 1. The note names the folder, which holds `manifest.json` and `relevance/{calls.json,error.txt}`. |

## 5 · What I could not verify

- **A real provider refusing `stop`.** None was sent. Live `gpt-6-luna` is never sent
  `stop`, so the live run does not exercise R1.15.
- **Real Anthropic and Gemini calls.** The test covers litellm's client-side check and a
  mock reply.
- **Python 3.11, 3.12 and 3.14, and the lowest dependency bounds.** These ran in CI only.
- **The notebook's outputs and R1.11's docstring sites.** Read, not executed.
- **That the schema interpreter contains everything.**

**Where this file lives.** Of the repo's tools, only `ruff format --check` reads
`as_built/`. It formats the Python blocks in Markdown, and this file's blocks are
formatted. The PR split leaves this directory out.

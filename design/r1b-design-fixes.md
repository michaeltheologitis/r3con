# TASK-40 · R1b design: r3con's design fixes (R1.7 to R1.15)

System Designer, for
[TASK-40](https://app.notion.com/p/R1b-r3con-s-design-fixes-settings-honoured-the-model-s-schema-in-the-sandbox-a-failed-stage-say-3ef62fb222378165a493f2aaba2fc13f).
It works from the [approved spec](https://app.notion.com/p/3ef62fb22237810b991aea8b43677c54):
items R1.7 to R1.13 as amended at and after Gate A, plus the Conductor's amendments R1.14
and R1.15, made on 2026-10-05 under Michael's waiver of his gates. Branch
`claude/tender-shannon-eq4lq7-r1b`, cut at `c2d29ab` (R1 merged, r3context 0.1.1 released),
with `main` merged in at `623d21b` (PR #13: default model `openai/gpt-6-luna`, litellm
≥ 1.101 locked at 1.104.0, pydantic ≥ 2.11). Line numbers below are at `623d21b`.

**Revisions**

- 2026-10-05 · first version.
- 2026-10-05 · as built by the Implementer: the Conductor's two rulings (§3.1 items 3
  and 4), and what the build found (§3.3). §4.4, §7, §13 and §18 follow them.
- 2026-10-05 · the Cartographer, reading the build at `243f647`: §3.3 items 6 and 7,
  two claims the build contradicts. §4.4 and §18 follow item 6.
- 2026-10-05 · the Implementer: §3.3 item 6 fixed in `dd24413`; §4.4 and §18 say so.
- 2026-10-05 · the Implementer, from the Scout's report: §3.3 items 8 and 9. §4.3, §7,
  §9 and §16 follow them and item 6.
- 2026-10-05 · the Refactorer: §3.3 items 10 and 11, the Scout's adoption and the tests
  it merged. §3.2 item 2, §3.3 items 2 and 6, §4.5, §8 b, §9, §14 and §18 follow them.

**Reading it.** §1 to §3 cover what R1b changes and where the design departs from the
spec; that is the Gate B read. §4 to §12 take one item each: why it is needed, the
change, the signatures, what a user sees, whether any model-facing text changes, and its
tests. §13 is the 0.2.0 release and what it breaks. §14 lists the tests by file, §15 the
experiments and the live tier, and §16 the changes to `src/` by module, which is the
Gate C map. §17 is what R2 inherits, and §18 the outside dependencies and the risks.

**Evidence.** *Run* means I executed it at design time on scratch copies, either at this
branch's merged head (litellm 1.104.0, pydantic 2.13.5) or at the lowest bounds (litellm
1.101.0, pydantic 2.11.0, tiktoken 0.8.0, Python 3.11). *Read* means it comes from the
code and was not executed.

**Where this file lives.** It sits in `design/` at the repo root, on the task branch only,
as R1's did. Nothing collects it: pytest reads `tests/`, the wheel carries `src/`, and
pyright checks `src/`. `ruff format --check` does format the Python blocks inside
Markdown, so every Python block here is valid Python and formatted. After editing this
file, run `uv run ruff format design/`. The PR split leaves `design/` out of the stack.

## 1 · What R1b changes, in one screen

| Item | The change | What a user sees | What breaks | Model-facing text |
| --- | --- | --- | --- | --- |
| R1.7 | `r3con.settings` becomes a module of constants. Every cap is read when the call runs. `run_pipeline` resolves, checks and records every cap once. | `r3con.settings.REASONING_MAX_TURNS = 3` is honoured and recorded. A cap below its floor is refused before any call. | `from r3con.settings import settings`. `R3CON_DOC_WORKERS=0`, which used to mean sequential, is now refused. | none |
| R1.8 | The model's schema runs in the vendored restricted interpreter, and any decorator is refused. | A schema that imports `os` is refused and retried, never run. | A schema with a decorator, a class nested in a class, or an import outside the allowlist. 0 of the 1,920 archived schemas has any of these. | Only the validator error the model reads after a refused attempt |
| R1.9 | One helper records every stage. A failure writes that stage's `calls.json` and `error.txt`, and adds a note naming the run folder. | `relevance/error.txt` exists. Every caller sees where the artifacts are, and the CLI prints the note. | nothing | none |
| R1.10 | `load_config`'s overrides become keyword-only, and a non-mapping `params` raises `TypeError`. A config field r3con does not read is refused. | A misspelt override or config field raises. | `load_config(name, unknown=…)`. A config file holding a key r3con does not read. | none |
| R1.11 | Silent excepts are narrowed or removed. `parallel_map` runs over `ex.map`. `_LazyStr`, `doc_ids`, the unreachable merge branches, `runs._version()` and the docstrings that name their callers go. | When one call of a stage fails, documents not yet started are never sent. | `render_relevance(doc_ids=)`, `parse_documents(doc_ids=)`, `ParseResult.doc_ids` and `.doc_label()`, `r3con.version`, `r3con.PackageNotFoundError` | none; the reasoning prompt stays byte-identical, and a test pins it |
| R1.12 | A config that pins a prompt version with no file is refused when loaded and again before the first call. | The error comes before anything is billed. | nothing; the same failure arrives earlier | none |
| R1.13 | `seed` is removed from r3con. | Runs on Anthropic and Gemini models work, and the run label loses `seed=42`. | `run(seed=)`, `load_config(seed=)`, `--seed`, `RunConfig.seed`, `StageRun(seed=)`, and a config file with `seed:` | none; requests lose the `seed` parameter |
| R1.14 | `run_codeact` keeps `timeout_s` as a float. | `timeout_s=0.5` means half a second, not zero. | nothing | none |
| R1.15 | A model that refuses `stop` is asked again without it, for the rest of the run. | A model newer than litellm's map no longer fails the reasoning stage. | nothing | none |

The release is 0.2.0 (§13). No README change. No versioned prompt changes: every
`prompts/<stage>/v1.yaml` stays byte-identical, and the model-facing messages written in
code (the retry and no-code messages and the no-summary label) stay in code.

Three rules shape every item:

- **Refuse before anything is billed.** Every cap, prompt pin and config field is checked
  when the config is loaded or before `run_pipeline` writes its manifest. A misspelling
  raises at once instead of being dropped (R1.7, R1.10, R1.12, R1.13).
- **A run has one value per cap, and the manifest records it.** No cap is bound at
  import (R1.7).
- **A failure leaves its trace.** The failing stage's calls and traceback are written to
  disk, and the exception says where (R1.9).

## 2 · Order of work

Each step is test-first: its new tests fail at `623d21b` (E1, §15), then the change makes
them pass. CI is green after every step.

| Step | Item | Lands |
| --- | --- | --- |
| 1 | R1.7 | `settings.py` as constants; every `from r3con.settings import settings` becomes `from r3con import settings` (src and tests); call-time cap defaults; `settings_snapshot(reasoning_max_turns=)` with floors; `write_manifest(settings=)` |
| 2 | R1.13 | `seed` out of `RunConfig`, `default.yaml`, the label, `run()`, `load_config()`, the CLI, `runtime/llm.py`, `StageRun`, `pipeline.py`; `examples/options.ipynb` |
| 3 | R1.10 | `load_config` keyword-only; `TypeError` for `params`; unknown config fields refused; `RunConfig` refuses extra fields |
| 4 | R1.12 | `prompts.require_prompt_path`; `config.check_prompts`, called by `load_config` and `run_pipeline`; the CLI's exit 2 |
| 5 | R1.9 | `pipeline._recorded_stage`; the four stages through it; the CLI prints exception notes |
| 6 | R1.8 | `check_schema` in the interpreter, with the decorator refusal |
| 7 | R1.11 | the sub-items of §8, in any order |
| 8 | R1.14 | the float timeout |
| 9 | R1.15 | the `stop` refusal |
| 10 | release | `uv version --bump minor` (0.2.0) and `uv.lock` |
| — | proof | one dispatch of `ci.yml` at the head: every job green, live included (E8) |

R1.7 goes first because it changes an import in nearly every module, and every later
diff should be written against the new one. R1.13 goes before R1.10 because it removes
one of the four overrides that R1.10 makes keyword-only.

## 3 · Divergences

### 3.1 Conductor amendments (2026-10-05, under Michael's waiver)

1. **R1.14 · `run_codeact` keeps `timeout_s` as a float.** Today `int(timeout_s)` turns
   0.5 s into 0, so every program times out at once, and 1.9 s into 1. §11.
2. **R1.15 · A refused `stop` is retried without it.** litellm before 1.101 said
   `openai/gpt-6-luna` takes `stop`, and the API answered 400 "Unsupported parameter:
   'stop' is not supported with this model." (the Conductor, asked directly). A model
   newer than the installed litellm's map gets the same answer: litellm 1.104 reports
   `stop` as supported for `openai/unknown-xyz` (*run*). §12.
3. **A cap below its floor is refused before the run folder exists.** This design let
   `run()` and the CLI leave an empty run folder for a refused cap (§4.4 as first
   written). The Conductor ruled that it is refused before the folder is created, as
   R1.12's missing prompt is. `run()` and the CLI call `settings_snapshot()` after
   loading the config and before creating their `TaskLogger`; the CLI prints
   `r3con: <message>` and exits 2. Only a caller that builds its own `TaskLogger` has
   made a folder by then.
4. **A config file that still sets `seed:` is told where the seed went.** It is refused
   by R1.10's unknown-field rule, and the message adds that `seed` was removed in
   r3con 0.2.0 and goes in `params`, with the file's own value:
   `` `seed` was removed in r3con 0.2.0; to send one, put it in `params` (params: {seed: 42}). ``

### 3.2 From the spec, in the design

1. **R1.11's `ex.map` claim is wrong, and the design follows what `ex.map` does.** The
   spec says that, over `ex.map`, "every call runs to completion before the first
   exception surfaces". That holds for today's hand-rolled loop. `ex.map` instead cancels
   every call not yet started once it reaches the lowest-indexed failure. Over 10 items
   on 2 workers with item 0 failing, today's code starts 10 calls in 1.0 s, and `ex.map`
   starts 3 in 0.2 s (*run*, Python 3.11 and 3.14). The design keeps `ex.map` and states
   the new rule: a failing stage stops sending documents. The lowest-indexed failure
   still surfaces, and every call already started still finishes first, so no thread is
   abandoned. §8 item d.
2. **R1.11's tiktoken except is narrowed and logged, not kept broad and logged.** ruff's
   BLE001 accepts a broad except that logs with `exc_info` only when the logger is named
   `logger` or `log`, not `_log`, r3con's name (*run*), so a logged `except Exception`
   would keep its `noqa`. The failure that can actually happen is an `OSError`: with a
   user's own empty `TIKTOKEN_CACHE_DIR` and no network, litellm 1.104 leaves the
   variable alone and `get_encoding` raises `requests.ConnectionError` (*run*).
   `except (OSError, ValueError)` covers that failure and tiktoken's hash check, with a
   warning. §8 item b. Superseded by §3.3 item 10.
3. **R1.11's `_supports_stop_parameter` except is removed, not logged.** litellm's
   `get_supported_openai_params` catches an unmapped model itself and returns `None`
   (*read*, 1.101 and 1.104). It returns `None` for an alias, an empty string and a bare
   name, and a list for every routable model tried (*run*). Its only `raise` is on the
   transcription path. The except guards nothing, so it goes with its `noqa`. §8 item a.
4. **R1.11's version lookup is removed, not narrowed.** `runs._version()` repeated
   `r3con.__version__`, so `write_manifest` reads `r3con.__version__`. §8 item c.
5. **R1.10 also refuses config fields r3con does not read.** It refuses them in the YAML
   (`ValueError`) and on `RunConfig` (`extra="forbid"`). Without this, R1.13 would
   silently drop the `seed: 42` of every user's 0.1 config and every
   `RunConfig(seed=…)`. That is the bug R1.10 exists to fix, moved to another door. §7.
6. **R1.12's check sits in two places.** The spec left the placement to the design.
   `load_config` refuses the config before the CLI or `run()` creates a run folder, and
   `run_pipeline` refuses a hand-built `RunConfig`. Both call one function. §9.
7. **R1.9 does three things the spec did not list.** The helper flushes the failing
   stage's `calls.json`, because R2 asks for it. It records an interrupt
   (`BaseException`) as well as a failure. And the CLI prints the exception's notes and
   drops its own "partial artifacts" line, which is the "≈3 lines" R2 had planned for
   the CLI. §6.
8. **R1.7 gives every cap a floor, `DOC_WORKERS` included.** The spec shows only the turn
   cap refused at 0. `R3CON_DOC_WORKERS=0` and `--doc-workers 0` used to run the
   documents one at a time, and are now refused (§4.4). One worker is how to ask for
   sequential.
9. **R1.8 is backed by offline evidence that the spec gave to E8 only.** All 1,920
   distinct model-written schemas in r3con-evaluation's archive (`2159d83`) were run
   through `exec` and through the interpreter on a prototype of §5: 1,918 give
   byte-identical JSON schemas, and the other 2 are refused by both (*run*). The archive
   uses no `__future__` import, decorator, nested class or `def`, and imports only from
   `pydantic` (1,920), `typing` (194) and `enum` (1) (*run*). E8's live half stays. §5.
10. **R1.13 keeps the empty-reply re-roll and drops only the perturbation.** The spec
    removes "the seed-perturbing re-roll". I read that as the re-roll's seed
    perturbation, because an empty reply from a reasoning model that spent its budget
    thinking is not about seeds. Without a seed, each re-roll is an independent sample.
    A caller's seed in `params` is re-sent unchanged. §10.
11. **R1.13 edits `examples/options.ipynb`.** Cell 3 passes `seed=42`, which 0.2.0
    refuses. The seed moves into `params`, so the stored output still answers the exact
    request the cell sends. Cell 5's output (two run labels, computed without a model
    call) is re-executed: it shows `seed=42`, and also `gpt-5.6-luna`, which has been
    stale since PR #13. §10.
12. **R1.11 takes in R1's Refactorer's leftovers.** Those are the docstrings that name
    their callers and the history comments. The spec's R1.11 is "what the Code Guide
    forbids", and the Code Guide allows a comment only for a non-obvious why. The sites
    are listed one by one in §8 item h.
13. **R1.11's `doc_ids` precondition is met.** The spec made the removal wait on
    r3con-evaluation not calling it. At `2159d83` it does not depend on r3context at
    all: it vendors its own copy of the pipeline under `evals/r3con/pipeline/` (*read*:
    its `pyproject.toml`, and no `import r3con` anywhere).

### 3.3 From the build

1. **`_refuses_stop` takes a `BadRequestError`, and the loop catches that class.**
   §12.3 shows `_refuses_stop(error: Exception)` with an `isinstance` check. The loop
   catches `litellm.exceptions.BadRequestError` instead and passes it in, so the check
   is the regex alone. pyright refuses `litellm.BadRequestError` (litellm does not
   re-export it), so the import is from `litellm.exceptions`.
2. **`reasoning.py` imports tiktoken at module level.** The import sat inside the
   guarded function because the old `except Exception` also covered a missing tiktoken.
   tiktoken is a declared dependency, so only `get_encoding` stays inside the narrowed
   `except (OSError, ValueError)`. Superseded by item 10.
3. **Three tests §6.4 and §10.3 call new pass at `623d21b`, as guards.**
   `test_an_empty_structured_reply_is_rerolled_with_the_same_request`: a direct call
   without a seed never carried one; the change is pinned by
   `test_a_seed_the_caller_sends_is_resent_unchanged_on_a_reroll` (`[5, 6, 7]` at
   `623d21b`). `test_without_a_logger_a_failure_carries_no_note`: nothing added notes
   at `623d21b`. The CLI's "partial artifacts printed once" assertion: the CLI printed
   its own line once; the test fails at `623d21b` on its other new assertion, that
   `relevance/error.txt` exists.
4. **The tests the rulings add.** `test_a_cap_below_its_floor_is_refused_before_a_run_folder_exists`
   (`test_r3con.py`), `test_doc_workers_below_one_exits_2_before_a_run_folder_exists`
   (`test_cli.py`) and the `seed` row of
   `test_a_config_field_r3con_does_not_read_is_refused`, which matches the 0.2.0
   message. Also added: `test_an_explicit_turn_cap_below_its_floor_is_refused`
   (`test_settings.py`), and a third row of
   `test_a_refusal_that_is_not_about_stop_is_raised` for a 400 that contains the word
   stop without quotes.
5. **Notebook cell 5 was run alone.** Its code reads `MODEL`, which cell 1 defines, so
   it ran with `MODEL` bound to cell 1's literal and no other cell executed.
6. **`run()` given a `RunConfig` that pins a missing prompt leaves an empty run
   folder.** `run()` checks prompts only through `load_config`, which it skips for a
   `RunConfig`, so the refusal comes from `run_pipeline` after `run()` has created the
   folder. No request is sent. The same pin given as a config name leaves no folder.
   Found by the Cartographer. Fixed in `dd24413`: `run()` calls `check_prompts` on
   whichever config it runs, a `RunConfig` included, before its caps check and the
   folder. `test_a_config_missing_a_prompt_is_refused_before_a_run_folder_exists`
   pins it for a missing file and a missing stage (item 11), and the cap test runs on a
   `RunConfig` too.
7. **A program that times out holds the loop until it ends.** The vendored `timeout`
   runs the program inside `with ThreadPoolExecutor(...)`, whose exit waits for it. The
   model is told "Program timed out after 0.5s", but `time.sleep(2)` under a 0.5 s limit
   returns after 2.00 s (*run*). So `timeout_s` (§1, §11) bounds what the model is told,
   not the wall-clock time; the same holds for the schema check's 30 s (§5.2). It
   predates R1b. Found by the Cartographer; not fixed.
8. **A config file that sets `overrides:` or `name:` is refused, and now a test says
   so.** Both are `RunConfig` fields a file must not set: a file's `overrides:` would be
   recorded as real overrides, and its `name:` would be ignored. `load_config` has
   refused them since `082ed76`, because neither is in `_CONFIG_FIELDS`, but no test
   pinned it, and a variant that let them through passed every test. Found by the
   Scout. `eb90208` adds both as rows of
   `test_a_config_field_r3con_does_not_read_is_refused`, with the same message as any
   other unread field; the rows fail when the check lets them through (*run*).
9. **A cap must be an `int`.** §4.3 checked only the floor, so
   `SCHEMA_MAX_ATTEMPTS = 2.5` or `True` was accepted and recorded as given, and a
   string raised `TypeError` from the comparison, which the CLI does not catch: the
   user got a traceback. Found by the Scout. Fixed in `4293c2c`: a value whose type is
   not exactly `int` (a `bool` included) raises
   `ValueError("SCHEMA_MAX_ATTEMPTS must be an integer >= 1, got 2.5.")`, with the
   value's `repr`, so the CLI prints `r3con: <message>` and exits 2. An `int` below its
   floor keeps the §4.3 message. Pinned by `test_a_cap_that_is_not_an_integer_is_refused`
   (2.5, `True`, `"3"`) and
   `test_a_cap_that_is_not_an_integer_exits_2_without_a_traceback`.
10. **The parse guard counts with `litellm.encode`, and tiktoken is no longer a direct
    dependency.** `len(litellm.encode(text=parse_json))` counts the same cl100k_base
    tokens, with `disallowed_special=()`, from the vocabulary litellm ships: litellm
    points `TIKTOKEN_CACHE_DIR` at it on import (1.101) or on the first `encode`
    (1.104), before loading it. `_parse_token_encoding`, `_count_tokens`, their cache
    and the warning of §8 item b go, and so does
    `test_a_tokenizer_that_cannot_load_falls_back_to_an_estimate_and_warns` with its
    `fresh_tokenizer` fixture: the Conductor ruled that it pinned the replaced code.
    `tests/conftest.py`'s `TIKTOKEN_CACHE_DIR` default goes too; only r3con's own
    `get_encoding` read it. Two offline cases now differ (*run*, litellm 1.104, the
    network refused). With a user's own empty `TIKTOKEN_CACHE_DIR` (§3.2 item 2), the
    old guard warned and estimated; the new one counts exactly. With litellm's
    `CUSTOM_TIKTOKEN_CACHE_DIR` pointing at an empty directory, the new guard raises
    and the reasoning stage fails; at 1.101 `import litellm` already fails there.
    Found by the Scout; `9e548ab`.
11. **Three pairs of tests become tables.** The `configs` fixture moves from
    `test_config.py` to `conftest.py`, and the R1.12 tests in `test_cli.py` and
    `test_r3con.py` write their overlay with it (`af2cc36`).
    `test_a_config_missing_a_prompt_is_refused_before_a_run_folder_exists`
    (`test_r3con.py`) replaces
    `test_a_config_pinning_a_missing_prompt_is_refused_before_a_run_folder_exists` and
    `test_a_run_config_missing_a_prompt_is_refused_before_a_run_folder_exists`: three
    rows, by name and as a `RunConfig` with a missing file or stage, each now matching
    its message (`19525e0`). `test_a_turn_cap_bounds_the_run_and_is_recorded`
    (`test_pipeline.py`) replaces the settings and explicit turn-cap tests of §4.5: two
    rows with a cap of 2, the explicit one over `REASONING_MAX_TURNS = 5`, so it also
    shows the argument replaces the setting (`a4cfa03`).

## 4 · R1.7 · Settings honoured

### 4.1 Why

`REASONING_MAX_TURNS`, `SCHEMA_MAX_ATTEMPTS` and `PARSING_MAX_ATTEMPTS` are copied into
default arguments at import (`pipeline.py:96`, `runtime/codeact.py:296`,
`stages/reasoning.py:180`, `stages/structuring/schema.py:143`,
`stages/structuring/parsing.py:208`). Setting `r3con.settings.X` afterwards changes
nothing, yet the manifest records the new value. An explicit `max_reasoning_turns=5` is
honoured, but the manifest records the global. `Settings` is a class used only as a
namespace.

### 4.2 `settings.py`: module constants

```python
PKG_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PKG_DIR / "prompts"
CONFIGS_DIR = PKG_DIR / "configs"

DOC_WORKERS = 16
REASONING_MAX_TURNS = 30
SCHEMA_MAX_ATTEMPTS = 5
PARSING_MAX_ATTEMPTS = 5
LLM_NUM_RETRIES = 10
LLM_EMPTY_CONTENT_RETRIES = 3
REASONING_PARSE_MAX_TOKS = 16000


def active_doc_workers() -> int: ...  # unchanged: R3CON_DOC_WORKERS, else DOC_WORKERS


def active_logs_dir() -> Path: ...  # unchanged


def settings_snapshot(*, reasoning_max_turns: int | None = None) -> dict[str, int]: ...
```

- **Names and values are unchanged**, so `r3con.settings.X` reads and writes keep
  working. `class Settings` and the instance `settings` go. `r3con/__init__.py` imports
  the module, with `from r3con import settings`, and `__all__` keeps `"settings"`. The
  one form that breaks is `from r3con.settings import settings`. It now raises
  `ImportError`, and four test files use it: `test_llm.py`, `test_pipeline.py`,
  `test_prompts.py` and `test_runs.py` (§14).
- **Every reader imports the module and reads the attribute inside a function body,
  never in a default.** `from r3con import settings`, then `settings.X`. Today's
  `from r3con.settings import settings` lines in `config.py`, `prompts.py`,
  `pipeline.py`, `runtime/codeact.py`, `runtime/llm.py`, `stages/reasoning.py`,
  `stages/structuring/parsing.py` and `stages/structuring/schema.py` change, and the
  imports of `active_doc_workers` and `active_logs_dir` stay as they are. The
  `from r3con import settings` inside a module that `__init__` is still importing is
  the pattern `__init__` already uses for `from r3con import r3con`.
- **Comments.** Each constant keeps one short line saying what it bounds. The note on the
  missing `LOGS_DIR` goes, because `active_logs_dir()`'s docstring already says why. The
  tenacity sentences go, because they have one home (§8 item h).

### 4.3 `settings_snapshot`: the caps one run executes under, checked

`settings_snapshot(*, reasoning_max_turns=None)` returns these seven keys, in this order.
These are the manifest's keys, unchanged from 0.1. Each value is read when the function
is called, and an explicit `reasoning_max_turns` replaces the setting when it is not
`None`. A value below its floor raises
`ValueError(f"{KEY.upper()} must be >= {floor}, got {value}.")`, naming the first failing
key in this order. For example: `REASONING_MAX_TURNS must be >= 1, got 0.` A value that
is not an `int`, a `bool` included, raises
`ValueError(f"{KEY.upper()} must be an integer >= {floor}, got {value!r}.")` (§3.3
item 9).

| Key | Read from | Floor | Passed to a stage by `run_pipeline` |
| --- | --- | --- | --- |
| `doc_workers` | `active_doc_workers()` | 1 | relevance and parsing, `workers=` |
| `reasoning_max_turns` | `REASONING_MAX_TURNS`, or the explicit value | 1 | reasoning, `max_turns=` |
| `schema_max_attempts` | `SCHEMA_MAX_ATTEMPTS` | 1 | schema, `max_attempts=` |
| `parsing_max_attempts` | `PARSING_MAX_ATTEMPTS` | 1 | parsing, `max_attempts=` |
| `llm_num_retries` | `LLM_NUM_RETRIES` | 0 | no: `runtime/llm.py` reads it per call |
| `llm_empty_content_retries` | `LLM_EMPTY_CONTENT_RETRIES` | 0 | no: read per call |
| `reasoning_parse_max_toks` | `REASONING_PARSE_MAX_TOKS` | 0 | no: read per call |

The floors are a private `_FLOORS: dict[str, int]` in `settings.py`. The last three keys
are checked and recorded but not threaded through the stages. They are read where they
are used, when the call runs, which R2.1 relies on for `LLM_NUM_RETRIES`. A run whose
caller changes one of them from another thread mid-run would differ from its record.
Nothing guards against that.

### 4.4 The stages: `None` means "the setting, read when the call runs"

```python
def run_codeact(*, max_turns: int | None = None) -> CodeActResult: ...
def reason(*, max_turns: int | None = None) -> CodeActResult: ...
def propose_schema(*, max_attempts: int | None = None) -> ProposalResult: ...
def parse_one_document(*, max_attempts: int | None = None) -> BaseModel: ...
def parse_documents(*, max_attempts: int | None = None) -> ParseResult: ...
def run_pipeline(*, max_reasoning_turns: int | None = None) -> Answer: ...
```

Only the changed parameter is shown, and every other parameter keeps its position and
default. `parse_documents` gains `max_attempts` and passes it to each
`parse_one_document`. `None` is resolved where the loop runs: in `run_codeact` for turns,
and in `propose_schema` and `parse_one_document` for attempts. `reason` and
`parse_documents` pass `None` through. Each function keeps its own `>= 1` check and
message, which is now accurate: "Override via settings.X".

`run_pipeline`, in this order:

1. `check_prompts(config)` (R1.12, §9).
2. `caps = settings_snapshot(reasoning_max_turns=max_reasoning_turns)`.
3. With a logger, `write_manifest(..., settings=caps)`.
4. The four stages, each given its cap from the table in §4.3.

Steps 1 and 2 raise before anything is sent or written. A caller's own `TaskLogger`
has already created its run folder by then. `run()` and the CLI create theirs only
after their own `settings_snapshot()` call, which follows `load_config` and its R1.12
check, so through them a refused cap or prompt leaves no folder (§3.1 item 3); the CLI
prints `r3con: <message>` and exits 2. `run()` also calls `check_prompts` itself, so a
`RunConfig` passed to it is refused before the folder too (§3.3 item 6).

`write_manifest` gains one keyword:

```python
def write_manifest(
    task_logger: TaskLogger,
    *,
    task: str,
    config: Any,
    n_docs: int,
    context_chars: int | None = None,
    settings: Mapping[str, int] | None = None,
) -> Path: ...
```

`settings=None` records `settings_snapshot()`. The manifest's `settings` block keeps its
seven keys, and now holds what the run used:
`{"doc_workers": 16, "reasoning_max_turns": 3, "schema_max_attempts": 5, ...}`.

### 4.5 Tests

All of these fail at `623d21b` except the snapshot's first row.

- `test_settings.py` (new; `settings.py` has no test file today):
  - `test_the_snapshot_holds_each_cap_as_set_when_it_is_taken`:
    `monkeypatch.setattr(r3con.settings, "SCHEMA_MAX_ATTEMPTS", 2)` and
    `R3CON_DOC_WORKERS=3` give 2 and 3, and an explicit `reasoning_max_turns=4` gives 4.
  - `test_a_cap_below_its_floor_is_refused`, one row per key, each one below its floor:
    `ValueError` matching `"<KEY> must be >= <floor>, got <value>."`.
- `test_pipeline.py`:
  - `test_a_turn_cap_bounds_the_run_and_is_recorded` (§3.3 item 11), with reasoning
    that never commits (`answering_llm.answers(reasoning="<code>\nprint(1)\n</code>")`).
    A cap of 2, set as `r3con.settings.REASONING_MAX_TURNS` or passed as
    `max_reasoning_turns=2` over a setting of 5, gives 3 reasoning requests (2 turns and
    the synthesis call), `n_turns == 2` in `reasoning/result.json`, and `2` in the
    manifest. Today: 31 requests, and the manifest records 30.
  - `test_an_attempt_cap_set_in_settings_bounds_its_stage`, parametrized over
    `SCHEMA_MAX_ATTEMPTS` with a schema reply that never validates, and
    `PARSING_MAX_ATTEMPTS` with `'{"rows": [{"who": 7}]}'`. Set to 2, over one document,
    each stage sends 2 requests and raises `SchemaError`. Today: 5.
  - `test_a_cap_below_its_floor_sends_and_writes_nothing`: one row,
    `REASONING_MAX_TURNS = 0`. It asserts `llm.requests == []` and no `manifest.json`.
    Today the setting is ignored and the whole run is sent, 8 requests over `DOCS`.
- `test_codeact.py`: `test_a_turn_cap_set_in_settings_bounds_a_direct_loop`. With
  `REASONING_MAX_TURNS = 2`, `run_codeact` without `max_turns` makes 3 requests. Today
  it asks for 30 and runs out of script.
- `test_runs.py`: `test_the_manifest_records_the_settings_it_is_given`. Here
  `write_manifest(settings={...})` is recorded as given, and without the keyword the
  record is the snapshot.

## 5 · R1.8 · The model's schema runs in the restricted interpreter

### 5.1 Why

`check_schema` runs model-written code with plain `exec` (`parsing.py:98`), and the model
writes it from summaries of the caller's documents. A schema that imports `os` and runs
a shell command is accepted, and the command runs (the spec, verified). Q5 (a) decided
the fix: run the schema in the interpreter the reasoning stage already uses. It also
refuses any decorated method with an error the model can retry on, and changes nothing
in the README.

### 5.2 `check_schema`

The signature is unchanged: `check_schema(schema_code: str) -> type[BaseModel]`, raising
`SchemaError`. In `stages/structuring/parsing.py`:

```python
_SCHEMA_IMPORTS = ("pydantic", "typing", "datetime", "enum", "decimal")
# Bound before a schema runs, so one that forgets its pydantic import still loads.
_SCHEMA_GLOBALS: dict[str, Any] = {
    "BaseModel": BaseModel,
    "Field": Field,
}  # and the rest


def _decorators(schema_code: str) -> list[str]: ...
```

What changes:

1. **Decorators are refused first, statically.** `_decorators` parses the code and lists
   every decorator on a `FunctionDef`, `AsyncFunctionDef` or `ClassDef` at any depth, as
   `@<ast.unparse(decorator)> on <name>`. Code that does not parse returns `[]`, and the
   interpreter reports the syntax error in step 2. A non-empty list raises
   `SchemaError(f"Schema code uses decorators, which r3con does not run: {listed}.
   Express each constraint as a field type or a Field(...) argument instead, for
   example Field(ge=0) or Literal[...].")`. The reason for refusing: the interpreter
   ignores every `decorator_list` when it builds a function or a class (*read*: the
   vendored `create_function` and `evaluate_class_def`). So `@field_validator`,
   `@model_validator`, `@computed_field` and `@property` were accepted and quietly never
   ran. Plain methods do run in the interpreter, and are allowed.
2. **The code runs in a fresh `LocalPythonExecutor`.** It is created with
   `additional_authorized_imports=list(_SCHEMA_IMPORTS)` and the executor's default
   timeout (30 s), given `send_variables(dict(_SCHEMA_GLOBALS))`, and called on the code.
   `except (InterpreterError, ExecutionTimeoutError) as e` raises
   `SchemaError(f"Schema code failed to execute: {e!r}") from e`. The interpreter wraps
   every error inside the code as an `InterpreterError`, a syntax error included, so the
   narrow except is complete. It is not a broad except, and the `noqa: S102` goes with
   the `exec`.
3. **`namespace = executor.state`**, and the rest of today's checks run on it unchanged:
   `Parse` defined, a `BaseModel` subclass,
   `model_rebuild(_types_namespace=namespace)`, no untyped objects, every top-level
   field a `list[...]`.

What the schema may use: imports from the five modules above and from the interpreter's
base list (collections, datetime, itertools, math, queue, random, re, stat, statistics,
time, unicodedata). It may not use `open`, `__import__`, dunder attribute access, a
statement other than a field, an assignment, a method, `pass` or a docstring in a class
body (so no nested `class Config:`), or more than the interpreter's operation and loop
limits. `from __future__ import annotations` is refused too. I considered authorizing
`__future__`: it grants nothing, but 0 of the 1,920 archived schemas use it, so it stays
out. A model that writes it reads which line was refused, and retries.

The rename of `_SCHEMA_EXEC_GLOBALS` to `_SCHEMA_GLOBALS` drops a word that is no longer
true. The comment shrinks to the one line above. `check_schema`'s docstring states what
it does: runs the code in the restricted interpreter, refuses decorators, and returns
`Parse`. Its history sentences and the sentence naming its caller go (§8 item h).

### 5.3 Measured on a prototype of 5.2 (*run*)

- The suite: 259 pass. 2 fail on message text only: the `"class Parse(:"` rows of
  `test_parsing.py` and `test_schema.py` match `"failed to execute: SyntaxError"`, and
  the error now reads `Schema code failed to execute: InterpreterError("Code parsing
  failed on line 1 due to: SyntaxError: …")`. Those two rows match `"SyntaxError"`.
- 1,920 archived schemas: 1,918 identical to `exec`, 2 refused by both, about 10 ms per
  check.
- Refused: `import os` ("Import of os is not allowed"), `from __future__ import
  annotations`, a nested `class Config:`, `@field_validator` with `@classmethod` (both
  listed), and an unquoted self-reference.
- Accepted with JSON identical to `exec`: Enum, Literal, `date`, `Decimal` with `ge=0`,
  an alias, `ConfigDict`, a quoted self-reference, `list[A | B]`, `Annotated[int,
  Field(ge=0)]`, a plain method (which runs), and a schema with no imports at all (the
  globals).

### 5.4 Model-facing text

The schema prompt is unchanged. Its sentence "The schema is `exec`'d as Python" is there
to ask for valid identifiers, and stays true. What changes is the validator error the
model reads after a refused attempt. That error is fed back by `_retry_prompt`, which is
model-facing text kept in code. Execution errors now arrive as the interpreter's
`InterpreterError(...)` text, and the decorator refusal is new. Both appear only on an
attempt that is refused, and an accepted schema's run sends what it sends today.

### 5.5 Tests (`test_parsing.py`)

- `test_a_schema_cannot_reach_outside_the_interpreter`, parametrized over three rows:
  `import os` followed by `os.system(f"touch {marker}")`, then `open(marker, "w")`, then
  `__import__("os")`. Each raises `SchemaError`, and `marker` (in `tmp_path`) does not
  exist afterwards. Today the first two run.
- `test_a_decorated_method_is_refused_with_what_to_do_instead`, parametrized over
  `@field_validator`, `@model_validator(mode="after")` and `@computed_field` with
  `@property`. Each raises `SchemaError` matching `"decorators, which r3con does not
  run"` and the decorator's name. Today all three are accepted.
- `test_a_schema_with_plain_methods_and_rich_types_is_accepted`: the 5.3 "accepted"
  shapes in one schema. A record validates through it, and the method runs. This passes
  today too, as a guard.
- The two syntax-error rows match `"SyntaxError"`.

## 6 · R1.9 · A failed stage says why

### 6.1 Why

Only reasoning writes `error.txt` (`pipeline.py:293–303`). When relevance, schema or
parsing fails, the run folder holds the manifest and an empty directory. The four stage
blocks repeat the same flush, `write_json` and totals (`pipeline.py:153–292`). The CLI is
the only caller told where the artifacts are.

### 6.2 The helper, in `pipeline.py`

```python
@dataclass
class _StageRecord:
    run: StageRun | None
    result: dict[str, Any] = field(default_factory=dict)


@contextmanager
def _recorded_stage(
    task_logger: TaskLogger | None, stage: str, model: str, *, transcript: bool
) -> Iterator[_StageRecord]: ...
```

The contract:

- **Without a logger**, it yields `_StageRecord(run=None)`, writes nothing, adds no note,
  and lets every exception through untouched.
- **With a logger**, it yields a record whose `run` is
  `StageRun(stage=stage, task_logger=task_logger, model=model)`. The body passes
  `record.run` to the stage and sets `record.result` to the stage's `result.json` fields.
  - **On a clean exit**, it calls `run.flush(write_transcript=transcript)`, then writes
    `<stage>/result.json` as `{**record.result, "totals": run.compute_totals()}`.
    `totals` comes last, as it does today.
  - **On any `BaseException`** from the body or from those writes, it calls
    `run.flush(write_transcript=transcript)`, which writes the failing stage's
    `calls.json` holding every call that completed. It writes `<stage>/error.txt` with
    `traceback.format_exc()` through `task_logger.write_text`, then
    `error.add_note(f"r3con: partial artifacts in {task_logger.dir}")`, and re-raises.
    There is no `result.json`. An interrupt is recorded like a failure, and the CLI
    still exits 130.

| Stage | `stage` | `transcript` | `record.result` keys (unchanged) |
| --- | --- | --- | --- |
| relevance | `"relevance"` | False | `n_rounds`, `n_docs`, `rounds` |
| schema | `"structuring/schema"` | True | `schema_code`, `thought`, `attempts` |
| parsing | `"structuring/parsing"` | False | `parsed`, `source_docs` |
| reasoning | `"reasoning"` | True | `answer`, `terminated_by`, `n_turns`, `turns` |

`run_pipeline`'s stages read like this, shown for relevance and reasoning, with the other
two the same shape:

```python
with _recorded_stage(task_logger, "relevance", model, transcript=False) as record:
    relevance = surface_relevance(
        task=task,
        documents=documents,
        model=model,
        prompt_version=config.prompts["relevance"],
        rounds=config.relevance_rounds,
        workers=caps["doc_workers"],
        run=record.run,
        **llm_kwargs,
    )
    record.result = {
        "n_rounds": config.relevance_rounds,
        "n_docs": len(documents),
        "rounds": [
            {"round": k + 1, "snippets": snippets}
            for k, snippets in enumerate(relevance.rounds)
        ],
    }

with _recorded_stage(task_logger, "reasoning", model, transcript=True) as record:
    result = reasoning.reason(
        ..., max_turns=caps["reasoning_max_turns"], run=record.run
    )
    record.result = {"answer": result.answer, "terminated_by": result.terminated_by}
```

The run folder's layout and every file's shape are unchanged, except that a failing
stage now leaves `calls.json` and `error.txt`. The `_log.info` progress lines stay.

### 6.3 Callers

- **The CLI** prints the error line as today, then each of the exception's notes on its
  own line (`for note in getattr(e, "__notes__", ())`), and no longer prints its own
  `r3con: partial artifacts in …`. The note carries that line, so stderr reads as today
  for a stage failure. A failure before the first stage (a refused cap) has no note,
  because there are no artifacts.
- **`run()`'s docstring** `Raises:` gains one sentence: an exception from a stage carries
  a note naming the run folder.
- **Python's traceback** prints the note under the error, so a notebook user sees it too.

### 6.4 Tests

All of these fail at `623d21b`.

- `test_pipeline.py`:
  - `test_a_failing_stage_leaves_its_calls_and_traceback_and_names_the_run_folder`,
    parametrized over the four stages, each answered with
    `litellm.APIConnectionError(...)`. It asserts that `<stage>/error.txt` names the
    error, that `<stage>/calls.json` exists and `<stage>/result.json` does not, that
    every earlier stage's `result.json` exists, and that `excinfo.value.__notes__ ==
    [f"r3con: partial artifacts in {logger.dir}"]`. It replaces
    `test_a_reasoning_failure_is_raised_and_its_traceback_left_on_disk`.
  - `test_a_stage_that_fails_midway_keeps_the_calls_made_before_it`: with
    `answers(relevance=["one", "two", down, down])`, round 2 fails. `relevance/calls.json`
    holds the two round-1 calls (`relevance-r1-d0` and `-d1`). Rounds are synchronous,
    so the list order is safe. This is what R2 needs.
  - `test_an_interrupted_stage_is_recorded_like_a_failure`: `KeyboardInterrupt()` from
    relevance over one document, and `relevance/error.txt` names it.
  - `test_without_a_logger_a_failure_carries_no_note`.
- `test_cli.py`: `test_a_provider_failure_exits_1_and_points_at_the_partial_artifacts`
  also asserts `err.count("partial artifacts") == 1`.

## 7 · R1.10 · `load_config` refuses typos

```python
def load_config(
    name: str = DEFAULT_CONFIG,
    *,
    model: str | None = None,
    relevance_rounds: int | None = None,
    params: Mapping[str, Any] | None = None,
) -> RunConfig: ...
```

- **Keyword-only overrides.** `seed` is gone (R1.13), so three remain, and Python refuses
  any other name: `TypeError: load_config() got an unexpected keyword argument
  'relevence_rounds'`. A `None` override is still not applied and not recorded. `run()`
  keeps passing `**given`, whose keys are those three names.
- **`params` that is not a mapping** raises `TypeError`, whether it is the override or
  the file's `params:`. The message is unchanged:
  `` override: `params` must be a mapping of litellm keyword arguments (…) — got str. ``
  `_check_params` keeps its shape and its `noqa: TRY004` goes.
- **A config file field r3con does not read** raises `ValueError(f"config {name!r} has
  field(s) r3con does not read: {unknown}. A config sets {sorted(_CONFIG_FIELDS)}.")`,
  with `_CONFIG_FIELDS = frozenset({"model", "relevance_rounds", "prompts", "params"})`.
  This catches `seed: 42` from a 0.1 config and a misspelt `relevence_rounds:` (§3.2
  item 5), and `overrides:` or `name:`, which only `load_config` sets (§3.3 item 8). When `seed` is among them, the message adds that it was removed in 0.2.0
  and goes in `params` (§3.1 item 4).
- **`RunConfig` refuses fields it does not have**, through
  `model_config = ConfigDict(extra="forbid")`. So `RunConfig(..., seed=42)` raises
  pydantic's `ValidationError`, a `ValueError`. `load_config` builds it from named
  fields only.
- Model-facing text: none.

Tests (`test_config.py`), all failing at `623d21b`:

- `test_a_misspelt_override_is_refused`.
- `test_params_that_are_not_a_mapping_are_a_type_error`, with two rows: the override and
  the file. The file row moves out of `test_a_config_missing_what_a_run_needs_is_refused`.
- `test_a_config_field_r3con_does_not_read_is_refused`, with rows `seed: 42` and
  `relevence_rounds: 3`.
- `test_a_run_config_refuses_a_field_it_does_not_have`, using
  `RunConfig(..., relevence_rounds=3)`.

## 8 · R1.11 · What the Code Guide forbids or makes unnecessary

None of R1.11 changes model-facing text.

**a. `_supports_stop_parameter` (`runtime/codeact.py:109`).** The `try`/`except Exception`
and its `noqa: BLE001` go (§3.2 item 3). The silencing of litellm's provider-list message
and the `or []` stay. The existing three-row table (`openai/gpt-4o` takes `stop`,
`openai/gpt-5` does not, an alias does not and prints nothing) guards it.

**b. `_parse_token_encoding` (`stages/reasoning.py:124`).** It becomes
`except (OSError, ValueError) as error:` with `_log.warning("tiktoken cannot load
cl100k_base (%s); estimating the parse's size at 4 characters per token", error)` and
`return None`. Its `noqa` goes. `reasoning.py` gains `_log = get_logger("reasoning")`.
`lru_cache` keeps the `None`, so the warning appears once per process. The test is
`test_a_tokenizer_that_cannot_load_falls_back_to_an_estimate_and_warns`. It
monkeypatches `tiktoken.get_encoding` to raise `OSError` (the outside world's boundary),
calls `_parse_token_encoding.cache_clear()` before and after, checks that `reason`
answers, and checks for one warning in `caplog`. It fails today, which has no warning.
Superseded by §3.3 item 10: the guard counts with `litellm.encode`, and this test goes.

**c. The version.** `__init__.py` uses `from importlib import metadata as _metadata`,
`_metadata.version("r3context")` and `except _metadata.PackageNotFoundError`, so
`r3con.version` and `r3con.PackageNotFoundError` leave the package namespace.
`runs._version()` and its `noqa: BLE001` go, and `write_manifest` reads
`r3con.__version__`, imported inside the function, as its other imports are. The test is
`test_the_version_lookup_leaves_nothing_in_the_package_namespace` in `test_r3con.py`: no
`r3con.version` and no `r3con.PackageNotFoundError`. It fails today.

**d. `parallel_map` (`parallel.py`).** The signature is unchanged:
`parallel_map(func, items, *, max_workers) -> list[R]`.

```python
def parallel_map(
    func: Callable[[int, T], R], items: Sequence[T], *, max_workers: int
) -> list[R]:
    if max_workers <= 1 or len(items) <= 1:
        return [func(i, item) for i, item in enumerate(items)]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(func, range(len(items)), items))
```

The docstring states the rule. Results come in input order. With one worker or one item,
the calls run in the calling thread, so an interrupt reaches the call directly. If calls
raise, the lowest-indexed failure propagates once every call already started has
finished, and calls not yet started are cancelled. The module docstring drops its
account of who calls it. The tests are in `test_parallel.py`:

- `test_calls_not_yet_started_when_one_fails_never_run`: 10 items, 2 workers, item 0
  raises and the others sleep 0.2 s. It asserts `len(started) < 10` (3 on the
  prototype). Today all 10 start.
- `test_the_lowest_indexed_failure_is_the_one_raised`: item 3 fails at once and item 1
  fails after 0.1 s, and item 1's error is raised. This passes today, as a guard.
- The existing three tests stay.

**e. `_LazyStr` (`stages/reasoning.py:37`).** The class goes. `reason` computes
`parse_json = json.dumps(parse_dict, indent=2, ensure_ascii=False, default=repr)` once,
passes it to `_render_parse_for_codeact(parse_dict, parse_json)` (which no longer dumps),
and passes the same string as the `parse_json` prompt variable. The prompt variables stay
`task`, `schema_code`, `relevance`, `parse_json`, `samples_block` and `parse_block`: they
are the contract a user's overlay prompt is written against. The comment above them says
that, and drops "kept for older prompt versions". The test, in `test_reasoning.py`, is
`test_a_small_parses_prompt_is_the_v1_template_with_the_parse_as_json`. For a fixed parse
and notes, the first request's system prompt `==` `load_prompt("reasoning",
version="v1", relevance=render_relevance(notes), parse_block=json.dumps(parse_dict,
indent=2, ensure_ascii=False))`, with the parse's records stamped. That pins the
byte-identical text the spec asks for. It passes before and after.

**f. `doc_ids`.** These go: `render_relevance(relevance_snippets: list[str] | None) ->
str` loses `doc_ids`, `parse_documents` loses `doc_ids=`, and `ParseResult` becomes
`parse` and `source_docs` only, without `doc_label()`. Nothing in `src/` passes them, and
r3con-evaluation does not import r3con (§3.2 item 13). In tests,
`test_doc_ids_label_the_documents` goes and `test_relevance.py`'s render table loses its
`doc_ids` rows.

**g. The unreachable merge branches (`parsing.py:390–409`).** The signature stays
`_merge_with_source_docs(per_doc: list[tuple[int, BaseModel]], parse_cls)`, because R2
merges several parts under one document index through these pairs. The empty-input
branch goes (`parse_documents` returns before the merge), and so does the non-list
branch. Every field is concatenated in pair order, with one source index per record:

```python
fields = parse_cls.model_fields
merged = {name: [r for _, p in per_doc for r in getattr(p, name)] for name in fields}
sources = {name: [i for i, p in per_doc for _ in getattr(p, name)] for name in fields}
```

The non-list branch is unreachable from the pipeline, because `check_schema` refuses
non-list fields. A direct caller of `parse_documents` who passes a non-list `Parse`
gets a `ValidationError` or a `TypeError`, never a silent merge. `ParseResult`'s
docstring drops "non-list fields take the first non-None value". The existing merge
tests cover this.

**h. Docstrings that name their callers, and history comments.** The rule: a docstring
says what the thing does and its contract, not who calls it or what it used to be. A
comment is one line, and only for a non-obvious why. The sites:

| Site | What goes |
| --- | --- |
| `runtime/codeact.py` module docstring | "The stage-3 reasoning agent (…) is the only consumer;" |
| `runtime/llm.py` module docstring | "— what the stages actually call —". Its tenacity paragraph stays, as the one copy in code. `pyproject.toml`'s comment stays beside the dependency. |
| `runtime/llm.py` `litellm_chat_completion_full` | its tenacity paragraph, "(empirically confirmed)" with it |
| `settings.py` | "The CLI and `r3con.run` load one explicitly…", and the tenacity lines under `LLM_NUM_RETRIES` |
| `logging_setup.py` | "the CLI (`r3con.cli`) calls `configure_logging()` once at startup" becomes "an application calls" |
| `stages/relevance.py` `render_relevance` | "that stages 2 and 3 embed" |
| `stages/structuring/parsing.py` | `check_schema`'s "Intended to be called inside the schema-proposal loop…" and its leniency paragraph; `ParseResult`'s "the reasoning stage threads it in and stamps…"; the `_SCHEMA_GLOBALS` comment (§5.2) |
| `stages/reasoning.py` `tag_source_documents` | "The pipeline computes this mapping at merge time … drops it at the reasoning boundary…" |
| `config.py` | "(the board's run label base)", "(recorded for the board)", and "the board's grouping key and the eval's resume key" twice, which become "a run's identity" |
| `runs.py` | the module docstring's "written by the caller, if it wants one"; `StageRun`'s "Each stage constructs one of these…"; `_version()` with its docstring (item c) |
| `pipeline.py` `run_pipeline` | the doubled blank line |
| `parallel.py` module docstring | its account of the relevance and parsing stages (item d) |

Considered and left out: the `noqa: BLE001` on `cli.py:157` and on `r3con.py:86` (the
first reports to the user and the second logs, so both are deliberate); the comment at
`runtime/llm.py:152` that credits the backoff to tenacity (wrong, per R2's spec, and R2.1
rewrites that line); and `runs.py`'s `_DOC_CHUNK_RE` comment "never chunked" (R2 changes
it).

**The `noqa` markers.** Five leave `src/`: S102 (R1.8), TRY004 (R1.10), and BLE001 on
`_supports_stop_parameter`, `_parse_token_encoding` and `runs._version()` (R1.11). RUF100
fails on any that is left behind.

## 9 · R1.12 · A config that pins a missing prompt is refused before the first call

```python
# prompts.py
def require_prompt_path(name: str, version: str) -> Path: ...


# config.py
def check_prompts(config: RunConfig) -> None: ...
```

- `require_prompt_path` returns `resolve_prompt_path(name, version)` or raises today's
  `load_prompt` error, `FileNotFoundError(f"No prompt for stage {name!r} version
  {version!r} (looked in {a} or {b}).")`. `load_prompt` calls it, so the message has one
  source.
- `check_prompts` raises `ValueError(f"config {config.name!r} is missing prompt versions
  for stage(s): {missing}")` when a stage in `PROMPT_STAGES` has no pin. That check
  moves here from `load_config`. Then it calls `require_prompt_path` for each stage.
- **Called twice, one function.** `load_config` calls it last, so `run()` and the CLI
  refuse before creating a run folder. `run_pipeline` calls it first (§4.4), so a
  hand-built `RunConfig` is refused before the manifest and the first call. `run()`
  also calls it on the config it runs, so a `RunConfig` passed to `run()` is refused
  before its run folder too (§3.3 item 6). The second
  call also catches a working directory that changed between load and run, since the
  overlay is `./prompts`.
- **The CLI** loads the config inside a `try` that turns `FileNotFoundError` and
  `ValueError` into `r3con: <message>` and exit 2, as a source that matches nothing
  already does. Today a bad config file prints a traceback.
- Model-facing text: none. R2's `reasoning: v2` pin is checked like any other.

Tests, all failing at `623d21b`:

- `test_config.py`: `test_a_config_pinning_a_prompt_that_does_not_exist_is_refused_on_load`
  checks for `FileNotFoundError` naming the stage, the version and both places looked.
  `test_the_label_is_every_axis_of_the_resolved_identity` writes an overlay
  `prompts/reasoning/v9.yaml` for its `exp2`.
- `test_pipeline.py`: `test_a_config_that_cannot_render_its_prompts_is_refused_before_any_request`,
  parametrized over a missing file (`FileNotFoundError`) and a missing stage
  (`ValueError`). Both leave `llm.requests == []` and no manifest.
- `test_r3con.py`: `test_a_config_missing_a_prompt_is_refused_before_a_run_folder_exists`
  (§3.3 item 11).
- `test_cli.py`: `test_a_config_pinning_a_missing_prompt_exits_2_before_any_request`, with
  an overlay `configs/exp.yaml` in the working directory and `--config exp`.

## 10 · R1.13 · `seed` is removed from r3con

### 10.1 Why

r3con sends `seed` on every call. litellm refuses it for Anthropic and Gemini models
before any network call: `UnsupportedParamsError` for `anthropic/claude-sonnet-5-5` and
`gemini/gemini-3.8-flash`, the README's examples, at 1.101 and 1.104 (*run*). So
`r3con.run` fails on its first call with providers the README names.

### 10.2 What goes

| Where | Today | After |
| --- | --- | --- |
| `RunConfig` | `seed: int = 42` | no field (and `extra="forbid"`, §7) |
| `configs/default.yaml` | `seed: 42`, and "seed" in its header comment | gone |
| the label | `default[model=gpt-6-luna,seed=42,rounds=2,prompts=(…)]` | `default[model=gpt-6-luna,rounds=2,prompts=(…)]` |
| `run()` | `seed: int \| None = None` | gone |
| `load_config()` | `seed` override | gone (§7) |
| CLI | `--seed` | gone: argparse exits 2, "unrecognized arguments" |
| `pipeline.py` | `seed = config.seed`; `{**config.params, "seed": seed}`; `StageRun(seed=)` | `llm_kwargs = {**config.params}` plus transport |
| `runtime/llm.py` | `seed` parameter of both functions; `request["seed"]`; `call_seed = seed + attempt` | gone. A `seed=` passed to either function goes through `**kwargs` to the provider, as any parameter does. |
| `StageRun` | `seed` parameter and attribute; `"seed"` in `transcript.yaml` | gone |
| docstrings and comments | "model, seed, relevance rounds" in `config.py`, `settings.py`, `runs.py`, `pipeline.py`, `cli.py`, `r3con.py`; the re-roll's perturbation | gone |

**The re-roll** of an empty structured reply stays (§3.2 item 10). It re-sends the same
request up to `LLM_EMPTY_CONTENT_RETRIES` times, and its error reads "the model returned
empty text after N attempt(s)", without "(seed-perturbed re-rolls)".

**A seed is still possible**, through `params={"seed": 42}`, which reaches every request
untouched and shows in the label as `params={seed=42}`.

**`examples/options.ipynb`** (§3.2 item 11):

- Cell 3's `seed=42,  # default 42` line goes, and its `params` becomes
  `params={"temperature": 0.3, "seed": 42}`, with the trailing comment naming a seed
  among the things `params` passes on. The requests the cell sends are the ones its
  stored answer came from, since 0.1 sent `seed=42` and `temperature=0.3` on every call.
  So the stored output stays true and the cell is not re-run, which would need the
  self-hosted endpoint.
- Cell 5 calls only `load_config(...).label()`. Its code is run from `examples/` on the
  branch's package, and the printed output replaces the stored one.

`quickstart.ipynb` and `vllm.ipynb` never mention a seed.

Model-facing text: none. Requests lose the `seed` parameter, and no message changes.

### 10.3 Tests

- `test_r3con.py`: `test_a_run_on_a_provider_that_takes_no_seed_completes`, parametrized
  over `anthropic/claude-sonnet-5-5` and `gemini/gemini-3.8-flash`, on `answering_llm`.
  It fails today with `UnsupportedParamsError` and passes once `seed` is gone (*run*,
  both models, on a fake that drops `seed`).
- `test_pipeline.py`: `test_no_request_carries_a_seed_unless_params_sets_one` replaces
  `test_the_configs_params_and_seed_are_in_every_request`. The default config sends no
  `seed`, and `params={"seed": 7}` on `openai/gpt-4o` puts `7` in every request.
  `CONFIG` drops `"seed"`.
- `test_llm.py`:
  - `test_an_empty_structured_reply_is_rerolled_with_the_same_request`: 3 requests with
    equal messages, and none carries `seed`.
  - `test_a_seed_the_caller_sends_is_resent_unchanged_on_a_reroll`: `seed=5` through
    `**kwargs` gives `[5, 5, 5]`. Today it gives `[5, 6, 7]`.
  - The other tests drop their `seed=` arguments.
- `test_config.py`: the labels lose `seed=42`. The override table loses its seed rows, and
  `load_config(seed=7)` is `TypeError` (§7's test covers it).
- `test_cli.py`: the label assertions lose `seed=`.
  `test_the_flags_override_the_config_and_show_in_the_label` drops `--seed 7` and its
  request check. `test_seed_is_not_an_option` asserts exit 2 and "unrecognized
  arguments: --seed".
- `test_runs.py`: `StageRun` without `seed`, `"seed" not in transcript`, and
  `"seed" not in manifest["config"]`.
- `test_schema.py`'s `test_the_callers_request_options_reach_the_request` passes `seed=42`
  as an example of an extra keyword. It still reaches the request through `**kwargs`, so
  that test stays as it is.

## 11 · R1.14 · `run_codeact` keeps the timeout it is given

In `run_codeact`, `timeout_int` goes and the executor gets `timeout_s` itself:

```python
executor = LocalPythonExecutor(
    additional_authorized_imports=additional_authorized_imports or [],
    # Annotated int upstream, but it reaches Future.result(timeout=), which takes a float.
    timeout_seconds=timeout_s,  # pyright: ignore[reportArgumentType]
    additional_functions={**(tools or {}), "final_answer": _identity_final_answer},
)
```

pyright refuses `float | None` for the vendored `timeout_seconds: int | None`, and the
vendored file is not edited. The ignore is scoped to that one argument, and with it
pyright and ruff are clean (*run*). At run time a float works: a 0.2 s program finishes
under 0.5, and times out under 0 (*run*). `reason` and `run_pipeline` already pass
`float | None`. A program that does time out is told "Program timed out after 0.5s",
which is now true. That observation is the model-facing text concerned, and only on a
timeout.

Test (`test_codeact.py`): `test_a_timeout_under_a_second_is_kept_as_given`. A first turn
of `import time`, `time.sleep(0.2)`, `print('done')` under `timeout_s=0.5` runs without
error and feeds back `done`, then a commit follows. Today `int(0.5)` is 0 and the turn
times out. It is deterministic, because 0.2 s is above 0 and below 0.5 s with a margin
of 0.3 s either way.

## 12 · R1.15 · A model that refuses `stop` is asked again without it

### 12.1 Why

The reasoning loop sends `stop=["</code>", "<observation>"]` when litellm's parameter
map says the model takes it (`_supports_stop_parameter`). Client-side clipping
(`_clip_assistant_response`) is the correctness path. `stop` saves tokens, and a
self-hosted Qwen behind vLLM runs on past `</code>` without it. When the map is wrong
(litellm before 1.101 for `gpt-6-luna`, or any OpenAI model newer than the installed
map), the provider answers 400 and the whole run fails in its last stage.

### 12.2 What the refusal looks like (*run*, litellm 1.101 and 1.104, against a local server answering OpenAI's body)

- It arrives as `litellm.BadRequestError`, `status_code=400`, with
  `str(error) == "litellm.BadRequestError: OpenAIException - Unsupported parameter: 'stop'
  is not supported with this model."`, `.param == "stop"` and
  `.code == "unsupported_parameter"`.
- litellm sends it `num_retries + 1` times before raising: 11 at today's
  `LLM_NUM_RETRIES = 10`, and 3 after R2.1.
- A refused `temperature` is the same class, with `'temperature'` in the message.
- A context-window refusal is `ContextWindowExceededError`, a subclass of
  `BadRequestError`, whose message names no parameter.
- litellm's own client-side refusal (`UnsupportedParamsError`, also a
  `BadRequestError`) reads "does not support parameters: ['stop']".

### 12.3 The rule

```python
_STOP_REFUSED = re.compile(r"""['"]stop['"]""")


def _refuses_stop(error: Exception) -> bool:
    """Whether the provider refused a request because it carried ``stop``."""
    return isinstance(error, litellm.BadRequestError) and bool(
        _STOP_REFUSED.search(str(error))
    )
```

The match is on the parameter's name in quotes. That covers the provider's wording and
litellm's own, and leaves out every other 400: a refused `temperature`, a context-window
error (which R2 handles), and a 400 that merely contains the word "stop".

In `run_codeact`, the memory of the refusal is one local, `stop`:

- Before the loop, `stop` is `_STOP_SEQUENCES` when the caller passed no `stop` in
  `llm_kwargs` and `_supports_stop_parameter(model)`. Otherwise it is `None`. A caller's
  own `stop` travels in `llm_kwargs`, as today.
- Each turn sends `llm_kwargs`, plus `stop=stop` when `stop` is not `None`.
- If that call raises and `stop is not None and _refuses_stop(error)`, the loop logs
  `_log.info("%s refused `stop`; this run's turns go without it", model)`, sets
  `stop = None`, and sends the same turn again: same messages and same `kind`, so no
  turn is spent. Any other error, or a refusal of a caller's own `stop`, raises as
  today.
- Later turns send no `stop`. The max-turns synthesis call never sent one.

The memory lasts for one `run_codeact` call. That is one run's reasoning stage, and one
model, so it holds "per run, per model string" with no module-level state to leak
across runs or tests. The refused request raised before `add_step`, so `calls.json`
holds the successful retry only. The next run learns again, at a cost of up to 11
refused requests at today's retry count, or 3 after R2.1. A 400 is normally not billed.
Model-facing text: none, because the messages are identical and only the `stop`
parameter is dropped.

### 12.4 Tests (`test_codeact.py`, on `openai/gpt-4o`, which the map says takes `stop`)

`REFUSED = litellm.BadRequestError("litellm.BadRequestError: OpenAIException - Unsupported
parameter: 'stop' is not supported with this model.", model="gpt-4o",
llm_provider="openai")`, queued in the fake. The fake raises a queued exception for
whichever request comes next, here the first turn, which carries `stop`.

- `test_a_model_that_refuses_stop_is_asked_again_without_it_for_the_rest_of_the_run`:
  `solve(llm, REFUSED, prints("1"), final("ok"), model="openai/gpt-4o")`. It asserts
  `["stop" in r for r in llm.requests] == [True, False, False]`, equal messages in the
  first two requests, `len(result.turns) == 2` and `result.answer == "ok"`. Today it
  raises.
- `test_a_refusal_that_is_not_about_stop_is_raised`, parametrized over
  `BadRequestError("… Unsupported value: 'temperature' does not support 0.7 …")` and
  `ContextWindowExceededError("… maximum context length is 4000 tokens …")`. Each is
  raised after 1 request. This passes today, as a guard.
- `test_a_callers_own_stop_is_never_dropped`: `solve(llm, REFUSED, model="openai/gpt-4o",
  stop=["END"])` raises `BadRequestError` after 1 request, and that request's `stop` is
  `["END"]`. This passes today, as a guard.
- The existing `test_stop_sequences_are_sent_only_to_models_that_take_them` and the
  synthesis test's "no `stop`" assertion stay.

## 13 · The release: 0.2.0, and what breaks

`uv version --bump minor`, which gives 0.2.0 and updates `uv.lock`'s entry, as its own
commit at the end of the build. 0.2.0 is not on PyPI, so the release dry run's check
passes. After Merged, the Conductor tags `v0.2.0` on `main`, and Michael approves the
`pypi` environment, as in R1's §10.3.

What 0.2.0 breaks, for whoever writes the release note:

- **Seed (R1.13):** `run(seed=)`, `load_config(seed=)`, `r3con run --seed`,
  `RunConfig(seed=)` and `RunConfig.seed`, `StageRun(seed=)`, and a config file with
  `seed:`, whose message says where the seed goes. Each now raises. Pass
  `params={"seed": N}` instead. Run labels lose `seed=42`,
  so a 0.1 label and a 0.2 label never match.
- **Settings (R1.7):** `from r3con.settings import settings` raises. Use
  `from r3con import settings` or `import r3con` and `r3con.settings`.
  `R3CON_DOC_WORKERS=0` and `--doc-workers 0` are refused.
- **Config (R1.10):** `load_config` takes `model`, `relevance_rounds` and `params` by
  keyword only. Any other name raises `TypeError`, as does a `params` that is not a
  mapping (it was `ValueError`). A config file field r3con does not read raises
  `ValueError`, and so does `RunConfig` with a field it does not have.
- **Removed (R1.11):** `render_relevance(doc_ids=)`, `parse_documents(doc_ids=)`,
  `ParseResult.doc_ids` and `ParseResult.doc_label()`, plus the accidental
  `r3con.version` and `r3con.PackageNotFoundError`.
- **Behaviour:**
  - The model's schema runs in the restricted interpreter, so a schema with a decorator,
    a nested class or an import outside the allowlist is refused and retried (R1.8).
  - A failing stage stops sending documents not yet started (R1.11).
  - A config pinning a missing prompt fails on load (R1.12).
  - Exceptions from a stage carry a note naming the run folder (R1.9).

## 14 · The tests, by file

The R1 machinery stays as it is: `tests/conftest.py` (`FakeLLM`, `llm`, `answering_llm`,
`_isolated`), pytest-socket, pytest-httpserver on 127.0.0.1, `ci.yml` and `release.yml`.
One fixture moves: `configs`, from `test_config.py` to `conftest.py` (§3.3 item 11).
"New" means the test fails at `623d21b`, which is E1. "Guard" means it passes before
and after.

| File | Added | Changed or removed |
| --- | --- | --- |
| `test_settings.py` (new) | snapshot as set; floors table (R1.7) | — |
| `test_pipeline.py` | turn cap, from settings or passed in; attempt caps; refused cap sends nothing (R1.7, new) · failing stage ×4; midway calls kept; interrupt recorded; no note without a logger (R1.9, new) · prompts refused before any request (R1.12, new) · no seed unless `params` (R1.13, new) | `from r3con import settings`; `CONFIG` without seed; the reasoning-failure test folded into R1.9's |
| `test_config.py` | misspelt override; `params` type; unread field; `RunConfig` extra (R1.10, new) · missing prompt on load (R1.12, new) | `FULL`, the labels and the override table without seed; `exp2`'s v9 overlay; the `params` row moves out of the ValueError table |
| `test_parsing.py` | cannot reach outside; decorators refused (R1.8, new) · rich and plain methods (R1.8, guard) | syntax row matches `SyntaxError`; `test_doc_ids_label_the_documents` removed |
| `test_schema.py` | — | syntax row matches `SyntaxError` |
| `test_codeact.py` | settings turn cap, direct (R1.7, new) · sub-second timeout (R1.14, new) · `stop` refused and retried (R1.15, new) · other refusals raised; caller's `stop` kept (R1.15, guard) | — |
| `test_reasoning.py` | prompt is v1 with the parse as JSON (R1.11, guard) | the tokenizer-fallback test, added by R1.11 and removed by §3.3 item 10 |
| `test_parallel.py` | not-yet-started calls never run (R1.11, new) · lowest-indexed failure raised (guard) | — |
| `test_relevance.py` | — | render table without `doc_ids` |
| `test_llm.py` | same-request re-roll; caller's seed resent unchanged (R1.13, new) | `from r3con import settings`; `seed=` arguments dropped |
| `test_runs.py` | manifest records given settings (R1.7) | `StageRun`, the `config(...)` helper and the manifest without seed |
| `test_r3con.py` | no-seed providers complete (R1.13, new) · namespace without leftovers (R1.11, new) · missing prompt before a run folder (R1.12, new) | — |
| `test_cli.py` | `--seed` is not an option (R1.13) · missing prompt exits 2 (R1.12, new) | labels without seed; "partial artifacts" printed once (R1.9) |
| `test_prompts.py` | — | `from r3con import settings` |
| `test_live.py` | — | unchanged |

## 15 · Experiments and the live tier

- **E1, second half: every reproducing test fails before its fix.** The Implementer runs
  each "new" test of §14 at `623d21b` before writing the fix, and the as-built document
  lists which failed. Null: a "new" test that passes at `623d21b`, unless the as-built
  document names why.
- **E8: the schema in the interpreter keeps the answer.** The offline half was measured
  here: 1,918 of 1,920 archived schemas are unchanged, and 2 are refused by both (§3.2
  item 9). The live half is the Proof Green dispatch at R1b's head. The baseline is the
  last green live run on `main`, PR #13's on `gpt-6-luna`. From both runs' `live-run`
  artifact, compare `len(structuring/schema/result.json["attempts"])` and the answer.
  Null: the schema stage needs more attempts than the baseline, or the answer fails the
  live assertion. If the null shows, read the refused attempt's `error` in the artifact
  before re-running.
- **The live tier is unchanged.** `tests/test_live.py` asks its question over
  `examples/memos` with the default config and asserts "halloran" and "11". The default
  config's model, rounds and prompt versions do not change. Its requests lose `seed`
  (R1.13). `gpt-6-luna` is not sent `stop` (litellm's map says no), so R1.15 does not
  act. The schema runs in the interpreter (E8).

## 16 · Changes to `src/`, by module

| Module | Items |
| --- | --- |
| `__init__.py` | R1.7 (`from r3con import settings`), R1.11 c (`_metadata`) |
| `settings.py` | R1.7 (constants, `settings_snapshot`, `_FLOORS`), R1.11 h, R1.13 (docstring) |
| `config.py` | R1.7 (import), R1.10 (`load_config`, `_CONFIG_FIELDS`, `extra="forbid"`, `TypeError`), R1.12 (`check_prompts`), R1.13 (`seed` field and label), R1.11 h |
| `configs/default.yaml` | R1.13 |
| `prompts.py` | R1.7 (import), R1.12 (`require_prompt_path`) |
| `pipeline.py` | R1.7 (caps), R1.9 (`_StageRecord`, `_recorded_stage`), R1.12 (`check_prompts`), R1.13, R1.11 h |
| `runs.py` | R1.7 (`write_manifest(settings=)`), R1.11 c (`_version` out), R1.13 (`StageRun.seed` out), R1.11 h |
| `parallel.py` | R1.11 d |
| `r3con.py` | R1.13 (`run(seed=)` out), R1.9 (docstring), R1.7 and R1.12 (caps and prompts checked before the run folder) |
| `cli.py` | R1.13 (`--seed` out), R1.9 (notes), R1.12 (exit 2) |
| `runtime/llm.py` | R1.7 (import), R1.13 (`seed` out, re-roll), R1.11 h |
| `runtime/codeact.py` | R1.7 (`max_turns=None`), R1.11 a, R1.14, R1.15 (`_STOP_REFUSED`, `_refuses_stop`, the loop), R1.11 h |
| `stages/relevance.py` | R1.11 f (`render_relevance`), R1.11 h |
| `stages/structuring/schema.py` | R1.7 (`max_attempts=None`) |
| `stages/structuring/parsing.py` | R1.7 (`max_attempts`), R1.8 (`check_schema`, `_decorators`, `_SCHEMA_IMPORTS`, `_SCHEMA_GLOBALS`), R1.11 f, g and h |
| `stages/reasoning.py` | R1.7 (`max_turns=None`), R1.11 b, e and h |
| `runtime/python_executor.py` | unchanged (vendored) |
| `logging_setup.py` | R1.11 h |

Outside `src/`: `pyproject.toml` (version 0.2.0; tiktoken out, §3.3 item 10),
`uv.lock`, `examples/options.ipynb` (R1.13), `tests/` (§14), and this file.

## 17 · What R2 inherits

- **R1.7.** `LLM_NUM_RETRIES` is still read when each call runs, and recorded in the
  manifest. R2.1 changes its value from 10 to 2.
- **R1.9.** `_recorded_stage` writes a failing stage's `calls.json`, `error.txt` and note.
  R2's stop-rule error is recorded like any failure. R2's own note (the TASK-36
  explanation) and the run-folder note both reach the CLI, which already prints every
  note, so R2 has nothing left to do in `cli.py`. `splits.json` is R2's, written as
  splits happen.
- **R1.11.** `parallel_map` is `ex.map`: a failing document stops the documents not yet
  started, so most of R2.4's dropped behaviour arrives anyway. `_merge_with_source_docs`
  keeps its `(index, parse)` pairs. `render_relevance` takes only the snippets, so R2
  adds its own parameter for the `Document 7.1` headings.
- **R1.12.** R2's `reasoning: v2` pin is checked on load and before the first call.
- **R1.13.** E6's "× 3 seeds" in R2's spec can no longer use r3con's seed. It passes
  `params={"seed": …}`, or repeats unseeded runs.
- **R1.15.** A `ContextWindowExceededError` is never taken for a `stop` refusal, and a
  test pins that.

## 18 · Outside r3con, and risks

**Depends on** (no collaborator's code):

- **litellm 1.101 to 1.104:**
  - `UnsupportedParamsError` for `seed` on Anthropic and Gemini.
  - `get_supported_openai_params` returns `None` for an unmapped model and does not
    raise on the chat path.
  - A provider 400 maps to `litellm.BadRequestError` carrying the provider's message.
  - `ContextWindowExceededError` is a subclass of `BadRequestError`.
  - `num_retries` re-sends a 400.
  - `litellm.encode(text=)` counts cl100k_base tokens with `disallowed_special=()`, from
    the vocabulary litellm ships (§3.3 item 10).
- **Python 3.11 to 3.14:** `Executor.map` cancels unstarted calls on the first exception
  in order, and `BaseException.add_note` exists.
- **pydantic 2.11 to 2.13:** `ConfigDict(extra="forbid")` and
  `model_rebuild(_types_namespace=)`.
- **The vendored executor:**
  - It ignores decorators.
  - Its class bodies accept only fields, assignments, methods, `pass` and a docstring.
  - Its import allowlist.
  - It wraps errors as `InterpreterError`.
  - `timeout_seconds` reaches `Future.result(timeout=)`.

**Risks, and what would show them:**

- **A live schema the interpreter refuses.** E8 shows it as an extra attempt. The
  offline half makes it unlikely: no archived schema uses a refused shape.
- **Unseeded live runs vary more.** The live assertion (Q3 (a)) is about the facts, and
  `gpt-6-luna` treats a seed as best-effort anyway. A red live run is read from its
  artifact, not re-run until green.
- **The cancellation test's timing.** It allows up to nine started calls of ten, and the
  prototype starts three. The main thread would have to be descheduled for 0.2 s to
  break it.
- **`_refuses_stop` misses a provider that words its refusal without quotes.** That run
  fails as today, with the provider's error in `reasoning/error.txt` (R1.9). The fix is
  one regex.
- **An empty run folder** after a refused cap or prompt, only for a caller that builds
  its own `TaskLogger` (§4.4). It is cosmetic, and the error says what to change.

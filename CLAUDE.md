# CLAUDE.md

## Where to start

Read `STATUS.md` first — the live snapshot (what's built, what's next, the open TODOs).
It is **gitignored**: this repo is public, `STATUS.md` is not. When `STATUS.md` and the
code disagree, `STATUS.md` is the intent and the code is the lagging snapshot.

`README.md` is for *users of the package*. `CLAUDE.md` (this file) is the working
discipline for agents. Keep the two apart — user-facing prose goes in the README.

## What this is

`r3con` answers a question whose evidence is spread across a **collection of documents**,
by building the reasoning apparatus **per task, at the moment the task is asked**.

**R³ — the three moves the method makes.** This vocabulary is the paper's, and the code,
the prompt folders, the config keys and the run-folder layout all use it. Keep them
aligned; if the paper's wording changes, the code follows.

| | stage | question it answers | in the code |
|---|---|---|---|
| 1 | **surfacing relevance** | what, in this corpus, matters for *this* question? | `stages/relevance.py` |
| 2 | **structuring** | what form does that mattering take, and what are its instances? | `stages/structuring/{schema,parsing}.py` |
| 3 | **reasoning** | what follows, over what was surfaced and what was structured? | `stages/reasoning.py` |

Each per-document note from stage 1 is a **relevance snippet**; all of them together are the
**relevant context**. Stage 2 is two steps that are one idea — propose the
schema, then **parse** every document into it. Stage 3 loads the parse into a Python
runtime and reasons there.

```
(question, documents)                 # each document assumed to fit in context; NO chunking
      │
      ▼
RELEVANCE   (relevance_rounds=2 default; each round fans out over docs in parallel)
   R1:  state(q, doc)                              # the document read against the question alone
   R2:  snippet(q, doc, OTHER docs' R1 snippets)       → the relevant context
      │   round k reads the FROZEN round k-1 set; a doc never sees its own prior state
      ▼
STRUCTURING · schema    schema(q, relevant context) → a per-task `Parse` class
STRUCTURING · parsing   per-doc in parallel: records(q, snippets, doc) against that schema
                        merge → one Parse (list-concat), each record tagged with its source doc
      ▼
REASONING   the sandboxed multi-turn Python loop, over
            (question, parse, relevant context) → the answer
```

Orchestrated in plain Python in `run_pipeline` — no coordinator class.

### The core thesis: just-in-time

Build the schema **and** the reasoning logic **per question** — not before. The anti-pattern
this explicitly opposes is pre-building a knowledge graph or fixed ontology that
"encompasses everything" before the question is seen. **If a design choice violates this
thesis, it is wrong by definition.**

## The shape of the project (so you don't re-add complexity)

This is a **deliberately small** codebase — ~4.7k lines, 39% of which is one vendored
sandbox file. It was simplified hard on purpose, and it arrived here by *deleting* an
evaluation harness several times its size. The bias is **toward deletion**.

- **One run config = the identity of a run.** A `RunConfig` (`config.py`) bundles
  everything that shapes the OUTPUT — model, seed, relevance_rounds, the prompt version of
  each stage, and `params` (handed straight to litellm) — and flows through the pipeline as
  one object. Everything
  output-shaping lives there; **runtime** knobs (parallelism, retry caps, paths,
  `REASONING_MAX_TURNS`) live in `settings.py` because they don't change a correct answer.
  To add an experiment axis, add it to `RunConfig._identity_parts()` — don't smear it into
  paths or env vars.
- **Versioned prompts, one folder per stage.** `prompts/<stage>/<version>.yaml`, shipped
  inside the package. The version of each stage is pinned in the run config and passed
  explicitly to `load_prompt(stage, version=…)` — there is NO env default. A user can add
  their own versions in an overlay (`R3CON_PROMPTS_DIR`, else `./prompts`), which is
  searched first; the packaged copy is the fallback. Same for configs
  (`R3CON_CONFIGS_DIR`, else `./configs`).
  - **NEVER edit a `prompts/<stage>/<v>.yaml` in place to change/fix/redesign a prompt —
    not even a "small" tweak, not even when the owner says "fix the prompt".** A run's
    manifest records the prompt *version string*; that file must keep meaning exactly what
    it meant when runs recorded it, or you destroy the provenance of what ran, what
    worked, and what didn't. **Always** create the next `vN+1.yaml` and bump the config's
    `prompts:` pin. The old version stays on disk, untouched, forever. (In-place edits are
    allowed ONLY for a typo with no behavioral effect; when in doubt, bump.)
- **One flat, opaque run-folder per run.** `<logs>/<UTC-timestamp>_<hex>/`, carrying **no
  identity in the name**. The question and every run parameter live *inside* in
  `manifest.json` (a `config` block, a `settings` block, and `r3con_version`), written
  **before stage 1** so a crashed run is still identifiable. Stage subfolders mirror the
  three moves: `relevance/`, `structuring/{schema,parsing}/`, `reasoning/`.
  Runs **accumulate**, never overwrite.
  - `r3con_version` is load-bearing: prompts ship *inside the wheel*, so `v1` names
    content only relative to a release. Without it, "which prompt actually ran?" is
    unanswerable after an upgrade.
- **Standard logging.** `logging_setup.get_logger("<stage>")`; enable with
  `R3CON_LOG_LEVEL=INFO` or `--verbose`. No bespoke stdout printer.
- **No coordinator class, no registry, no plugin system, no framework.** Orchestration is
  plain control flow. If a change tempts you to add one — stop.

## This is a published package (the rules that follow from that)

**The distribution is `r3context`; the import package is `r3con`.** They differ because
the PyPI name `r3con` is refused as too similar to the existing `recon`. Only three places
know the distribution name — `[project] name`, and the two `importlib.metadata` lookups in
`__init__.py` and `runs.py` — and **both lookups swallow their failure**, so a wrong one
does not raise: it silently freezes the recorded version at the in-source literal, and that
version is a run's provenance. `[tool.uv.build-backend] module-name` is what tells the
backend to ship `src/r3con/` under that distribution name. Everything else — the import,
the CLI verb, `R3CON_*`, the logger namespace, the manifest's `r3con_version` key — is
`r3con` and stays that way.

`r3con` is built to be published on PyPI and `pip install`ed by strangers. That constrains things the
research repo it came from never had to think about:

- **Nothing may write inside the package directory.** Site-packages is not writable and
  must be treated as read-only. Artifacts go to the *user's* chosen logs dir (default:
  `./logs` under the current working directory), never next to the code.
- **No path may be resolved at import time.** `active_logs_dir()` is a function, not a
  class attribute, because an attribute freezes the working directory of whoever imported
  first and ignores a later `R3CON_LOGS_DIR`. There is deliberately no `settings.ROOT`
  and no `settings.LOGS_DIR`.
- **Importing the package must not mutate the environment.** `load_dotenv` is called by
  the CLI and by `run()` — the application layer — never at import. And it needs
  `find_dotenv(usecwd=True)`: a bare `load_dotenv()` searches upward from the *calling
  file*, which is inside site-packages.
- **`prompts/` and `configs/` are package data**, shipped inside the wheel at
  `src/r3con/prompts/` and `src/r3con/configs/`, and resolved from the **package
  directory** — never from a "repo root". A path computed by walking up from `__file__` to
  a parent directory is a bug: it silently points outside the installed package.
  (Verified: the `uv_build` backend ships nested YAML under `src/r3con/` with no extra
  configuration.)
- **The package version is part of a run's provenance.** Prompts ship in the wheel, so
  "prompt v1" means whatever v1 was *in that release*. Never repurpose a version number
  across releases (see the prompt rule above), and note prompt changes in the changelog.
- **Public repo hygiene.** No API keys, no `.env`, no `logs/`, no benchmark results, no
  internal notes. `STATUS.md` is gitignored for exactly this reason. Before committing
  anything that quotes a run, check it carries no key and no private document text.
- **The connection is litellm's, not ours.** We call `litellm.completion` directly: the
  model string is litellm's, credentials come from the provider's own environment
  variable, and `params` is passed through verbatim. Do **not** grow a model registry, a
  provider abstraction, or an r3con-branded credential variable — litellm already is the
  abstraction over providers, and re-implementing it here would be building a framework
  inside a package that exists to not be one. The one escape hatch is `completion=`,
  the callable that makes each request (default `litellm.completion`); it is the
  litellm-shaped version of what smolagents does with `ApiModel(client=…)` and pydantic-ai
  with `OpenAIProvider(openai_client=…)`. It is transport, so it must never enter the run
  identity.
- **Dependencies are a cost the user pays.** Current set: `litellm`, `jinja2`, `pydantic`,
  `pyyaml`, `python-dotenv`, `tiktoken`, `tenacity`. Adding one needs a reason. `rich` is
  *not* a dependency — it arrives transitively via litellm and must not be imported here.
  **Do not "clean up" `tenacity` because nothing imports it**: `runtime/llm.py` sets
  `num_retries` on every call and litellm imports tenacity lazily on the retry path, so
  dropping it breaks every call that hits a transient error — and only on the error path,
  which is exactly where you won't notice it.
- **This package executes model-written Python, conditioned on the user's documents.**
  Stage 2 `exec`s the proposed schema in-process to validate it — no sandbox at all. Stage 3
  runs the agent's code in the vendored restricted AST interpreter (import allowlist, blocked
  dunder access, operation/loop/wall-clock caps), which raises the bar but is not a boundary
  to bet a production secret on. Don't weaken either without saying so; documents are
  untrusted input. (Deliberately not in the README — the owner cut it as noise for readers.)

- **Attribution is not optional.** `runtime/python_executor.py` is vendored from
  smolagents (Apache-2.0, HuggingFace Inc.). Its header, the upstream link, the list of
  modifications, and the `NOTICE` file must all stay intact. Don't "tidy" that header.

## Running it

```bash
# Answer a question over your own documents (writes ./logs/<run-folder>/, prints the answer)
r3con run "your question" ./docs --verbose
```

```python
from r3con import r3con

docs = r3con.read_documents("./docs")              # folder/glob/files -> list[str]
result = r3con.run("your question", docs)          # documents ARE a list of strings
result.answer, result.relevant_context, result.structured_context, result.schema_code, result.run_dir
```

The lower level, when you want a prebuilt config, your own `TaskLogger`, or your own
connection:

```python
from r3con import load_config, run_pipeline
from r3con.runs import TaskLogger, new_run_folder

result = run_pipeline(
    task="your question",
    documents=[doc1_text, doc2_text],       # each must fit in the model's context
    config=load_config("default"),
    task_logger=TaskLogger(new_run_folder()),
)
```

The run config (`--config <name>`, default `default`) carries the identity — model, seed,
relevance rounds, prompt versions, generation params. Override single fields with
`--model`/`--seed`/`--relevance-rounds`; `--base-url`/`--api-key`/`completion=` are transport
and are **not** part of the identity. A document (or the relevance-state block plus the
document) that doesn't fit raises `ContextWindowExceededError` — that is the accepted, measurable
degradation, not a bug to paper over with chunking.

**Don't edit a `prompts/<stage>/<v>.yaml` while a run is in flight** — `load_prompt` reads
the YAML on every call. Add a new `vN.yaml` and pin it instead.

## Iteration is empirical

Unit tests prove plumbing; they do not tell you whether the pipeline answers correctly.
Changing a prompt or a stage means **running it on real documents and reading the
artifacts** — `relevance/result.json`, `structuring/schema/result.json`,
`structuring/parsing/result.json`, `reasoning/transcript.yaml` — and writing down
what you saw. A change justified
only by reasoning about the prompt text is not justified.

When an answer is wrong, read the artifacts in order and attribute the failure to a
**stage** before touching anything: did the relevance snippets stay faithful descriptions
(or collapse into premature per-document verdicts)? did the schema capture the right
concepts? are the answer-bearing records present and populated? did the agent use both
views, or anchor on the parse alone? Then fix *that* stage.

Benchmark evaluation does **not** live here. It lives in a separate research repo
(`grounded-reasoning`) which keeps its **own** copy of this pipeline — the two are
independent, and that duplication is deliberate. Don't add a benchmark harness, a
scoreboard, a judge, or a results database here, and don't go modify that repo to depend
on this one.

## Code hygiene as you go

When you read or edit a file, if you see obviously loose code in *that file*, clean it if
it fits the diff; otherwise log it in `STATUS.md` under `## TODOs`. Don't open files you
wouldn't have opened anyway; don't refactor for cohesion; don't introduce new
abstractions.

## Working in a checkout

```bash
uv sync            # .venv with the package + ipykernel/pip, ready for examples/
```

`examples/quickstart.ipynb` runs the whole thing over `examples/memos/` — five short
documents arranged so no single one holds the answer. It executes end to end against a
real provider, and states the correct answer in a comment so a reader can check the run
rather than trust it. Keep it that way: an example that can only be believed is not an
example.

## Tests

Don't run everything for every change — run the test file(s) for the modules you touched.
Run the **full** suite for shared-infra changes (`config.py`, `runs.py`, `pipeline.py`,
`runtime/`, `settings.py`, `prompts.py`, packaging) or before a multi-commit chunk lands.

Module → test: `stages/relevance.py`→`test_relevance`; `stages/structuring/schema.py`→
`test_schema`; `stages/structuring/parsing.py`→`test_parsing` + `test_parse_documents`;
`stages/reasoning.py`→`test_reasoning`; `runtime/codeact.py`→`test_codeact`;
`runtime/python_executor.py`→`test_python_executor`; `runtime/llm.py`→`test_llm`;
`runs.py`→`test_runs`; `pipeline.py`→`test_pipeline`; `config.py`→`test_config`;
`prompts.py`→`test_prompts`; `r3con.py`/`cli.py`→`test_r3con`;
`parallel.py`→`test_parallel`; the `completion=` hook→`test_completion_override`;
`logging_setup.py`→`test_logging_setup`.

Run one with `uv run python tests/test_<x>.py` — they are plain scripts with a `__main__`
runner, not pytest-driven.

Anything touching packaging must also be checked against a **built wheel**, not just the
source checkout — `uv build` then install it into a scratch venv and run the CLI from a
directory that is not this repo. A test that passes in the checkout and fails after
`pip install` is the failure mode this repo is most exposed to.

## Committing

You do NOT need an explicit "commit" instruction — when a unit of work is complete and its
tests pass, commit it. Keep commits atomic; delete the `STATUS.md` TODO bullet in the same
commit that lands the work. Stage files explicitly (don't `git add -A`). End commit
messages with the `Co-Authored-By` trailer. Routine work lands on `main`; branch only for
risky or long-lived changes. Pushing is worth a quick heads-up unless told otherwise.

**Do not publish to PyPI.** A paper has to appear first, so releasing is blocked until the
owner explicitly asks for it *in the moment* — a checklist, a TODO, or a finished version
bump is not permission. Building a wheel locally to verify it is fine; uploading is not.
A release is outward-facing and irreversible.

## Non-goals (do not propose)

- **Chunking**, or any size-gated path that splits a document. Each document is assumed to
  fit; a non-fitting one is an accepted `ContextWindowExceededError`.
- **A benchmark/eval harness, scoreboard, or judge** in this repo — that is
  `grounded-reasoning`'s job, and the split is deliberate.
- **A registry, a plugin system, or a framework** of any kind.
- **Identity in log folder names.** Folders stay opaque; the run config in the manifest is
  the identity.
- **Vendoring more code** to avoid a dependency, or removing the smolagents attribution.

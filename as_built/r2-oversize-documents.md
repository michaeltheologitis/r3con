# TASK-35 · R2 as built: a document too long for the model's window is read in parts

Cartographer, for TASK-35. This is what exists at `20b8b3c` on
`claude/tender-shannon-eq4lq7-r2`, checked against `design/r2-oversize-documents.md` at
that commit. The base is `main` at `cfc17bc`, r3context 0.2.0. The code is the same at
`71ac060`, `5e716f2` and `20b8b3c`: the last two commits change only the design. So E3
and E6, run at the earlier two, apply to the head.

**Evidence.** *[run]* means I executed it: the suite, an offline probe (a fake
completion, or litellm against a server on 127.0.0.1), or a CI run whose artifact or log
I downloaded and read. *[read]* means I read it in the code and did not execute it.
Nothing I ran reached a provider.

**Reading it.** §1 is what a user of 0.2.0 sees differently. §2 is the map of the code.
§3 is the divergences from the design. §4 is the measured results. §5 is what I could
not verify.

## 1 · What a 0.2.0 user sees on this branch

**Now works**

- **A document too long for the model's window is read in parts** instead of failing the
  run. It is cut in 2 near its middle, at a paragraph break where there is one, then
  each part in 2 again, until every part fits. Each part is its own call, in relevance
  (both rounds) and in parsing [run].
- **The answer still uses the caller's numbering.** `relevant_context` has one entry per
  document passed in, with a split document's notes joined. Every record from a part
  says `"document": N`. Only the summaries in the reasoning prompt say `Document 4.1`,
  `4.2`, with one added sentence explaining them [run: E2, E6].
- **There are two triggers.** The first is an estimate, before anything is sent. It
  applies when litellm's model map gives the model's input window, which it does for
  OpenAI, Anthropic and Gemini models at the lock. The second is the provider's refusal.
  That is the only trigger for a self-hosted model, a Router alias, and the default
  model at litellm 1.101 to 1.103, where `gpt-6-luna` is not in the map [run at 1.101 and
  1.104].
- **The user is told.** Each split logs one WARNING, and the run folder gains
  `splits.json`. A library user gets these on stderr for the registry grown to 40,000
  characters on a mapped 5,000-token window [run]:

  ```text
  relevance-r1-d3: documents[3] (Document 4) is estimated at 8,441 tokens, over the 4,250-token line; reading it in 2 parts
  relevance-r1-d3c0: documents[3] (Document 4) is estimated at 4,555 tokens, over the 4,250-token line; reading it in 4 parts
  ```

**Fails differently**

- **A refused request is sent 3 times, not 11**: a context-window refusal, and a wrong
  key [run].
- **A server that fails 3 times in a row now fails the call.** Two 500s then a 200
  recovers; three 500s raise `InternalServerError` after 3 requests. In 0.2.0 the same
  call recovered [run, §4.1].
- **When the prompt and the other documents' notes fill the window by themselves**, more
  parts cannot help, and the run stops. It raises `litellm.ContextWindowExceededError`
  with a note. With a known window it stops before sending anything. This is E5's run,
  40 short reports on a 6,000-token window, as the error and its first note read [run]:

  ```text
  r3con estimated relevance-r2-d0 at 8,293 tokens, over the 5,100-token line (the 6,000-token input window litellm's model map gives hosted_vllm/window-6000, less 15%); it was not sent.
  r3con: reading documents[0] (Document 1) in more parts cannot help in relevance-r2-d0: the prompt and notes sent with it are about 8,218 tokens, over the 5,100-token line by themselves. The relevant context has outgrown the model's window.
  ```

  The run folder keeps `splits.json`, the stage's `calls.json` and `error.txt` [run].

**Changes in every run**

- The default config pins reasoning prompt `v2`, so the label reads `reason=v2`. A run
  that splits nothing sends exactly what 0.2.0 sends, apart from `num_retries` [run, E3].
- The manifest's `settings` gain `window_margin_percent: 15` [run: live artifact].
- The manifest still says `r3con_version: 0.2.0`; the branch has no version bump [run].

**Changes some runs**

- **A document estimated at 85% to 100% of a mapped window** was sent whole, and is now
  read in 2 parts. Setting `r3con.settings.WINDOW_MARGIN_PERCENT = 0` sends it whole
  again [run, §4.3]. On the lock's map, the line is 783,700 tokens for `gpt-6-luna` and
  170,000 for `claude-sonnet-5-5`.
- **A Router alias makes litellm print to stdout.** For a model string litellm cannot
  place with a provider (`my-router-alias`), it prints its "Provider List" banner
  twice per run, from r3con's one window lookup. 0.2.0 printed nothing. An unmapped
  `hosted_vllm/…` string prints nothing. `litellm.suppress_debug_info = True` silences
  it [run].

**Breaks**

- `RelevantContext.snippets`, `.rounds` and `relevance/result.json` hold a list of part
  notes for a document read in parts [run]. Only runs that split are affected, and 0.2.0
  raised on those, apart from the 85% to 100% band above [read].
- Nothing is removed from the API [read].

## 2 · How it is built

**Size.** `src/` changes by +800/−122 lines over 15 files, 178 of them the new prompt
file. r3con's own Python goes from 3,765 to 4,265 lines, and the new `splitting.py` is
395 of them. `tests/` changes by +989/−27 lines, with 39 new test functions [run].

**Where the complexity is.** In one place: `Splits.read_in_parts` in
`r3con/splitting.py`. The stages hand it two things and know nothing else.
`runtime/llm.py` changes by one comment.

### 2.1 The seam: `r3con.splitting.Splits`

```python
class Splits:
    max_input_tokens: int | None  # litellm's map; None when unmapped
    margin_percent: int
    line: int | None  # window * (100 - margin) // 100

    def __init__(
        self,
        documents: Sequence[str],
        *,
        model: str,
        margin_percent: int | None = None,
        task_logger: TaskLogger | None = None,
    ) -> None: ...

    def parts(self, doc: int) -> list[str]: ...

    def read_in_parts(
        self, doc: int, *, call: str, rest: str, send: Callable[[str, str], T]
    ) -> list[T]: ...
```

- **What crosses the seam.** A stage passes `rest`, everything sent beside a part: the
  rendered system prompt, plus the response schema's JSON for a parse. It also passes
  `send(part, kind)`, which makes the call. It gets one result per part back, in order.
  The kind is `parse-d3` while the document is whole and `parse-d3c0`, `parse-d3c1`, …
  once split; `calls.json`'s existing `chunk` field picks up the part index [run].
- **One `Splits` per run**, built by `run_pipeline` after the manifest, and shared by
  every stage. A document's cuts persist, so parsing starts from the parts relevance
  left, and a later call may cut further but never merges parts back [run].
- **The window** is `litellm.get_model_info(model)["max_input_tokens"]`, looked up once.
  An unmapped model gets `None` and one INFO line, which a library user does not see by
  default [run].
- **The estimate** is cl100k_base tokens through `litellm.encode`, of the rest plus the
  part. A request whose rest and part together are no larger than the line in UTF-8
  bytes is not counted at all, since every token is at least one byte. On the default
  model, that means nothing short of 783,700 bytes is counted [run].
- **The cut** (`halve`) looks only in the middle half of a part, for a paragraph break,
  then a line break, then a sentence end (`。！？` included), then a space, else the
  exact middle. So each half is at least a quarter, and the halves rejoin exactly [run].
- **The loop, with a known window.** If the rest alone is over the line, it stops. If
  not, it halves until every part's estimate fits, then sends the parts in order. A
  refusal halves again and re-measures. The only other stop is a one-character part that
  still does not fit [run].
- **The loop, with an unknown window.** It sends. On a refusal, it stops if the refused
  part has fewer tokens than the rest; otherwise it halves and sends again [run].
- **In both modes**, a refusal discards the parts of that level already accepted. They
  stay in `calls.json`, since they were sent. The refused request is not in `calls.json`:
  it shows only as an event in `splits.json` [run]. One document's parts go one after
  another, and the documents still run in parallel [read]. Any other error passes
  through and never splits [run].
- **`splits.json`** is written on every split or stop, and again when a split document
  finishes a call. A run that splits nothing has none. E6's 4-part run, the error text
  shortened:

  ```json
  {"model": "openai/gpt-6-luna", "max_input_tokens": 922000, "margin_percent": 15, "line": 783700,
   "documents": {"3": {"cuts": [10068, 20092, 30017],
     "parts": {"relevance-r1": 4, "relevance-r2": 4, "parse": 4},
     "events": [
       {"call": "relevance-r1-d3", "cause": "refusal", "estimate": 8469, "rest": 720,
        "discarded": 0, "action": "split", "parts": 2, "error": "litellm.ContextWindowExceededError: …"},
       {"call": "relevance-r1-d3c0", "cause": "refusal", "estimate": 4583, "rest": 720,
        "discarded": 0, "action": "split", "parts": 4, "error": "…"}]}}}
  ```

### 2.2 The stages

- **`stages/relevance.py`.** `Snippet = str | list[str]` and `join_parts` live here.
  `render_relevance` gives each part a `### Document N.k` heading. In round 2 a part sees
  the other documents' notes, a split one's joined and unnumbered, and never its own
  document's other parts [run].
- **`stages/structuring/parsing.py`.** One `rest` is shared by all documents. Every
  record from a part carries its document's index, in part order [run]. A validation
  retry keeps the part's kind (`parse-d3c0-retry-1`). Its added error text is not in the
  estimate [read].
- **`stages/structuring/schema.py`.** Only the type changes. The schema call carries no
  document, so it is never split, and a refusal there fails after 3 requests [read].
- **`stages/reasoning.py` and `prompts/reasoning/v2.yaml`.** v2 is v1 plus one sentence,
  shown only when relevance read a document in parts. Without one, v2 renders to v1's
  text byte for byte. A document that only parsing split leaves no `N.k` heading and no
  sentence [run]. The schema and parsing prompts show `N.k` headings unexplained [read].
- **`settings.py`.** `LLM_NUM_RETRIES = 2`. `WINDOW_MARGIN_PERCENT = 15`, from 0 to 99,
  is checked and recorded with the caps [run].

### 2.3 Public surface, as the code has it

- `r3con.run` and `run_pipeline`: unchanged signatures [read].
- `r3con.surface_relevance(…, splits: Splits | None = None)` and
  `r3con.parse_documents(…, splits: Splits | None = None)`: new keyword. `None` builds a
  `Splits` without a record. `Splits` is imported from `r3con.splitting`; the top-level
  `r3con` does not export it [read].
- `relevance_snippets: Sequence[Snippet] | None` on `propose_schema`, `parse_documents`,
  `parse_one_document` and `reason`. `RelevantContext.snippets: list[Snippet]`.
  `Snippet` and `join_parts` are in `r3con.stages.relevance`, not exported [read].
- `r3con.settings.WINDOW_MARGIN_PERCENT`, and `read_in_parts` as a variable a user's
  reasoning prompt overlay may use [read].
- CLI: help text only [read].

### 2.4 CI

- A marker `experiment` (also `live`), so the default `-m "not live"` excludes it.
- The `live` job now runs `-m "live and not experiment"`, the same single test.
- A new `experiment` job runs only when `ci.yml` is dispatched with `experiment: true`,
  after the four check jobs, with a 45-minute timeout, uploading `experiment-run`. The
  input is declared under `workflow_dispatch` only. On a push both jobs skip [run:
  37289930961]. Through `release.yml`'s call it is unproven (§5).

### 2.5 The tests, by item

| Item | Tests [run, all pass] |
| --- | --- |
| R2.1 retries | `test_llm.py`: 3 new, against a 127.0.0.1 server; the caller-override test now passes 7 |
| `halve` | `test_splitting.py`: 3 |
| Window, line, margin | `test_splitting.py`: 3; `test_settings.py`: 1; `test_pipeline.py`: the manifest |
| The loop and the record | `test_splitting.py`: 14, both modes, every stop, the byte shortcut |
| Stages | `test_relevance.py`: 3; `test_parsing.py`: 2; `test_reasoning.py`: 2 |
| E2 to E5, and a stopped run's folder | `test_pipeline.py`: 6 functions, 8 cases |
| E6 | `test_experiments.py`: 9 cases, live, dispatched once |

The suite at the head: 393 passed, 10 deselected (the live test and E6) [run].

**What the tests leave unpinned.**

- **The stdout banner.** `conftest.py` sets `litellm.suppress_debug_info = True` for the
  whole suite.
- **A server failing 3 times.** Only "two 500s then a 200 recovers" is pinned.
- **E4's relevance count, by refusal.** In
  `test_a_later_stage_splits_further_without_renumbering`,
  `assert len(…) == 12 if measured else 13` parses as `assert (len(…) == 12) if measured
  else 13`, so the by-refusal row asserts `13`, which is always true. The count is in
  fact 13 [run], so the row holds, unpinned.

## 3 · Divergences from the design

**The design's own list holds.** Its §2.5 names ten departures found while building. I
checked each against the code; none is stale. The 401's class, the measured stop naming
part 0, the E4 and stopped-run sizes, and the banner I ran; the rest I read.

**No new divergence in behaviour.** I found no place where the build contradicts the
design. One wording mismatch is inside the design: §11's last risk says the Router alias
banner prints "once per run", and §2.5 item 9 says twice. Twice is what prints, from one
lookup per run, so the build matches §2.5. I did not edit the design. The gaps in the
tests are in §2.5 above.

## 4 · Measured results

### 4.1 E1 · R2.1: a refused request is sent 3 times

One call through r3con's `litellm_chat_completion` to a 127.0.0.1 server answering in
vLLM's words, litellm 1.104.0, at the head. The 0.2.0 column passes `num_retries=10`,
0.2.0's default, through the caller override [run]:

| Server answers | 0.2.0 (`num_retries=10`) | Head (`num_retries=2`) |
| --- | --- | --- |
| 400, context window | 11 requests, `ContextWindowExceededError` | 3 requests, same |
| 401, wrong key | 11 requests, `AuthenticationError` | 3 requests, same |
| 500 twice, then 200 | recovers on request 3 | recovers on request 3 |
| 500 three times, then 200 | recovers on request 4 | `InternalServerError` after 3 |
| 500 four times, then 200 | recovers on request 5 | `InternalServerError` after 3 |

The head column's first three rows are also the suite's tests, which CI's `lowest` job
runs at litellm 1.101 [run: 37289930961].

### 4.2 E2 to E5, and E3 against 0.2.0

- **E2 to E5** are the `test_pipeline.py` tests of §2.5, all passing [run].
- **E3 against `cfc17bc`.** This is a default run over the five memos, with the suite's
  answering fake, one worker and `gpt-6-luna` mapped. I ran it on 0.2.0's source and on
  the head's, and compared every request with `num_retries` removed. Both sides sent 17
  requests, byte-identical. `num_retries` is 10 against 2, the prompts read `reason=v1`
  against `reason=v2`, and neither run wrote `splits.json` [run]. The Implementer's run
  at `71ac060` (design §9) says the same.

### 4.3 The margin band, and a rest just under the line

These are offline probes of `Splits` with a mapped 10,000-token window and a 9-token
rest. The document is the grown registry, trimmed to each size [run]:

| Request, share of the window | Parts at margin 15 | Parts at margin 0 |
| --- | --- | --- |
| 80% | 1 | 1 |
| 86%, 90%, 95%, 99% | 2 | 1 |
| 101% | 2 | 2 |

With a known window and a rest close to the line, the run does not stop. Every part
then carries the whole rest [run]:

| Line | Rest | Document | Parts sent | Rest tokens sent in all |
| --- | --- | --- | --- | --- |
| 8,500 | 8,000 | 7,749 | 32 | 256,000 |
| 8,500 | 8,400 | 7,749 | 128 | 1,075,200 |
| 100,000 | 90,000 | 469,127 | 64 | 5,760,000 |

The last row is the design's §5.4 example (64 parts). Counting took 0.3 s.

### 4.4 The live tier at the head

CI run [37290154711](https://github.com/michaeltheologitis/r3con/actions/runs/37290154711)
was dispatched at `20b8b3c` with `live: true` and `experiment: false`. Every job is
green and `experiment` skipped. The push run 37289930961 at the same commit is green
[run: both read through GitHub]. From the `live-run` artifact [run]:

- The answer: "Halloran Services Ltd logged the most equipment incidents in Q3, with 11
  in total across its sites."
- The manifest records `reasoning: v2`, `llm_num_retries: 2` and
  `window_margin_percent: 15`. There is no `splits.json`.
- 18 calls (10 relevance, 1 schema, 5 parse, 2 reasoning), 27,803 tokens.

### 4.5 E6 · a document read in parts still answers

- **The claim and the rule.** A document read in 2 or 4 parts still answers. The null
  is any of three things: a seed that is right at 1 part and wrong at 2 or 4, a missing
  CT-118 to Halloran record from Document 4, or a doubled one.
- **Conditions.** CI run
  [37288918145](https://github.com/michaeltheologitis/r3con/actions/runs/37288918145),
  job `experiment`, at `5e716f2` (code identical to the head). It ran `openai/gpt-6-luna`
  at litellm 1.104.0 with the default config (`reason=v2`) and seeds 1, 2 and 3 through
  `params`, one run per cell. The registry is grown to 40,081 characters, with its code
  lines in part 2 of 2 and part 3 of 4. A wrapper around `litellm.completion` refuses
  user messages over 24,000 (2 parts) or 12,000 (4 parts) characters before they leave
  the runner, so the parts are set by refusal, not by the model's window.
- **Reproduce.** Dispatch `ci.yml` on the branch with `live=false`, `experiment=true`.
  Locally: `OPENAI_API_KEY=… uv run --locked pytest -m experiment -rA`.
- **Run by** the Implementer. I recomputed every column but seconds from the artifact;
  the seconds come from the job log, 267 s for the 9 runs [run].

| Parts | Seed | Refusals | Halloran, 11 | CT-118 records from Doc 4 | Calls | Prompt tokens | Completion tokens | Reasoning turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | yes | 1 | 17 | 44,864 | 2,369 | 1 |
| 1 | 2 | 0 | yes | 1 | 17 | 45,037 | 2,482 | 1 |
| 1 | 3 | 0 | yes | 1 | 17 | 45,151 | 3,110 | 1 |
| 2 | 1 | 1 | yes | 1 | 20 | 49,322 | 3,003 | 1 |
| 2 | 2 | 1 | yes | 1 | 23 | 60,816 | 2,989 | 4 |
| 2 | 3 | 1 | yes | 1 | 20 | 50,032 | 3,293 | 1 |
| 4 | 1 | 2 | yes | 1 | 27 | 60,963 | 3,428 | 2 |
| 4 | 2 | 2 | yes | 1 | 26 | 57,927 | 3,477 | 1 |
| 4 | 3 | 2 | yes | 1 | 27 | 60,644 | 3,382 | 2 |

| Parts | Mean calls | Mean prompt tokens |
| --- | --- | --- |
| 1, the baseline | 17.0 | 45,017 |
| 2 | 21.0 (1.24x) | 53,390 (1.19x) |
| 4 | 26.7 (1.57x) | 59,845 (1.33x) |

- **Result: the null does not show.** Every split run read the registry in K parts in
  both relevance rounds and in parsing (`parse-d3c0` to `c{K-1}`, never `parse-d3`). Its
  reasoning prompt had headings 4.1 to 4.K and v2's sentence; unsplit runs had neither.
  Cost: 194 calls, $0.061 at litellm's mapped prices [run: artifact].
- **Noise.** At 1 part, the three seeds' prompt tokens vary by 0.6%. One 2-part seed
  took 4 reasoning turns, which adds about 11,000 prompt tokens to that run.
- **What it does not show.** It is one run per cell, on one model and one question. The
  registry's mapping lines sit whole inside one part, so a fact cut across a part
  boundary is not tested. Nor does it test a real provider's refusal, or a split by
  estimate: `gpt-6-luna`'s window never splits these sizes.

## 5 · What I could not verify

- **`release.yml`'s call of `ci.yml`.** It passes only `live: false`, and `experiment`
  is not declared under `workflow_call`. Whether the job then skips cleanly has not run;
  actionlint finds nothing wrong with either file [read].
- **A real provider's refusal.** No refusal from a real provider has gone through the
  splitter. The tests use vLLM's words on a local server, and E6 uses a wrapper. That
  Anthropic's and Gemini's refusals map to `ContextWindowExceededError` is the design's
  *read* claim.
- **The 15% margin against Claude's and Gemini's tokenizers.** The estimate is
  cl100k_base; how far each provider's count runs above it is unmeasured.
- **litellm 1.102 and 1.103.** I ran only 1.101 (the floor) and 1.104 (the lock).

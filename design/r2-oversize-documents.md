# TASK-35 · R2 design: a document too long for the model's window is read in parts

System Designer, for TASK-35. It works from the
[approved spec](https://app.notion.com/p/3ef62fb2223781218375f54417d7e1ec), read with its
two amendments: 2026-10-04 (R2.1 lowers `num_retries` to 2, Q1 (a), Q2 (a), R2.4
dropped) and 2026-10-05 (R2.2 measures first when it can). Where the body and an
amendment disagree, this design follows the amendment. Branch
`claude/tender-shannon-eq4lq7-r2`, cut at `main` `cfc17bc` (R1 released as 0.1.1, R1b
merged as 0.2.0). Line numbers below are at `cfc17bc`.

**Revisions**

- 2026-10-05 · first version.
- 2026-10-05 · the Conductor's three rulings (§2.4). With a known window, the run stops
  only when the rest alone is over the line (§5.4). The stop note names no task ID
  (§5.6). A request under the line in UTF-8 bytes is not counted (§5.2). §1, §2.1
  item 5, §2.3, §7, §8, §9 and §11 follow them. Stop trusting the first version's three
  stop forms and its "names TASK-36". The prototype, re-run with the rulings, passes E2 to
  E5 and the new tests of §8 at 1.101 and 1.104.
- 2026-10-05 · built (Implementer). §2.5 lists where the build departs from this design
  and what it found; §4's 401 row is corrected; §9 gains E3's diff against `cfc17bc` and
  E6's measured table, both run.

**Reading it.** §1 and §2 are the Gate B read: what changes, and every place this design
decides something the spec left open or departs from it. §4 is R2.1. §5 is the new module
that holds R2.2's algorithm, and §6 wires it into the stages. §7 is what a user sees, §8
the tests by file, §9 the experiments and the live tier, §10 the changes to `src/` by
module (the Gate C map), and §11 the outside dependencies and risks.

**Evidence.** *Run* means I executed it at design time on scratch copies of `cfc17bc`:
at the lock (litellm 1.104.0, tiktoken 0.14.0, Python 3.13) or at the lowest bounds
(litellm 1.101.0, pydantic 2.11.0, Python 3.11). Nothing reached a provider: litellm
calls went to a server on 127.0.0.1 or to the suite's `FakeLLM`. *Prototype* means a
sketch of §5 and §6 in a scratch copy, run with the suite and with E2 to E5. *Read*
means from the code, not executed.

**Where this file lives.** `design/` at the repo root, on the task branch only, as R1's
and R1b's did. pytest reads `tests/`, the wheel carries `src/`, pyright checks `src/`.
`ruff format --check` does format the Python blocks inside Markdown, so every Python
block here is valid, formatted Python; after editing, run `uv run ruff format design/`.
The PR split leaves `design/` out of the stack.

## 1 · What R2 changes, in one screen

| Item | The change | What a user sees | What breaks | Model-facing text |
| --- | --- | --- | --- | --- |
| R2.1 | `settings.LLM_NUM_RETRIES` goes from 10 to 2. litellm keeps its own retries. | A refused request (400, 401, context window) is sent 3 times, not 11. A server that fails 3 times in a row now fails the call; it used to get 10 retries. | nothing in the API | none |
| R2.2 | A new module, `r3con/splitting.py`. Before a relevance or parse call, a document whose request is estimated over 85% of the model's mapped window is cut in 2, then 4, … at paragraph breaks. Any document the provider refuses with `ContextWindowExceededError` is cut the same way. Parts are numbered N.1, N.2, … in the summaries only. | A document too long for the model is read in parts instead of failing the run. `splits.json` in the run folder says how. When the relevant context itself fills the window, the run stops with a note naming the document and saying so. | nothing in the API. A document estimated between 85% and 100% of a mapped window, which was sent whole before, is now read in 2 parts. `RelevantContext.snippets` entries can be lists for a document read in parts. | `prompts/reasoning/v2.yaml`: v1 plus one sentence, shown only when a summary is a part. `configs/default.yaml` pins it, so the default label reads `reason=v2`. |

Three rules shape R2.2:

- **The caller's index space never moves.** `Document N`, `"document": N`,
  `relevant_context[i]` and `source_docs` keep one number per document passed in. Only
  the summary headings say `N.1`, `N.2`.
- **Measure when the window is known, and trust a refusal always.** The estimate decides
  before anything is sent. A refusal splits even when the estimate said the request fits.
- **An unsplit run sends what it sends today**, byte for byte. Measuring sends nothing,
  and v2's sentence renders to nothing (E3, *prototype*: 17 of 17 requests identical to
  `cfc17bc` over the memos, apart from `num_retries`, which litellm never sends).

## 2 · Divergences and decisions

### 2.1 What the spec deferred, decided here

1. **How far from the middle a break may lie.** Each kind of break is looked for in the
   middle half of the part only, from ¼ to ¾ of its length. Kinds are tried in order:
   paragraph break, then line break, then sentence end, then any space, then the exact
   middle. So every part is at least a quarter of its parent, and the parts shrink
   geometrically (§5.3).
2. **Do one document's parts fan out in parallel?** No. They go one after another,
   inside that document's worker. A refusal of the first part then costs the level
   nothing else. The other documents still run in parallel.
3. **The type that carries a split document's notes between stages.**
   `Snippet = str | list[str]`: a document's note, or one note per part, in order
   (§6.1). It flows from relevance to schema, parsing and reasoning in the existing
   `relevance_snippets` parameter, whose type widens. `Answer.relevant_context` stays
   `list[str]`, with a split document's notes joined.
4. **The shapes of `splits.json` and `relevance/result.json`.** §5.5 and §6.1. A split
   document's entry in `relevance/result.json` is the list of its part notes, and an
   unsplit run's file is unchanged.
5. **The stop threshold.** It is measured in tokens, by the same counter as the estimate,
   and depends on whether the window is known (§5.4). With a known window, the run stops
   only when the rest alone is over the line, the amendment's measured form. With an
   unknown window, it stops when a refused part is shorter than the rest, the spec's
   form. In both, a part of one character that still does not fit stops the run (the
   cut floor). That is how the loop bounds itself. This is the Conductor's ruling 1
   (§2.4); the first version had a third form.
6. **Does r3con's retry loop wrap a caller's `completion=`?** The question is moot: the
   2026-10-04 amendment removed r3con's own loop. `num_retries=2` reaches whatever
   `completion` is, as `num_retries=10` does today. A Router gets it as a keyword.
7. **The encoding** (the amendment's ask): `litellm.encode(text=...)`, which is cl100k_base
   from the vocabulary litellm ships. It is the counter the reasoning stage's parse guard
   already uses, it loads offline at 1.101 and 1.104 (*run*, in a process with no network
   at all), and it costs about 50 ms per million characters (§5.2). o200k_base also loads
   offline from litellm's copy, and counts 1.4% fewer tokens on the memos (*run*). The
   difference is inside the margin, so I kept the counter r3con already has.
8. **The margin is an integer percent**, `WINDOW_MARGIN_PERCENT = 15`. The amendment asks
   for "a setting, not a config field", recorded with the caps. `settings_snapshot`
   checks every cap as an exact `int`, so a percent fits that machinery and a float
   would not. It gets a floor of 0 and a ceiling of 99 (§6.6).

### 2.2 Where this design departs from the spec's text

1. **The stop rule counts tokens, not characters.** The spec's mock-up compares
   characters. Tokens are what the window is made of, and the counter is already
   needed. The rule is unchanged where it applies, with an unknown window (ruling 1): a
   refused part shorter than everything sent with it.
2. **A sentence longer than half the part can be cut inside, not only one longer than
   the whole part.** This follows from the middle-half window (§2.1 item 1). A split
   that kept every sentence whole could land 1% from one end, and halving would then no
   longer shrink the parts.
3. **v2's sentence shows when a summary the reasoning prompt carries is a part**, that is
   when relevance read a document in parts. A document that only parsing split leaves
   no `N.k` heading for the sentence to explain, so the sentence is not shown. This is
   inside Q2 (a), "only when a document was split", and narrower than it.
4. **The measured stop raises a `ContextWindowExceededError` that r3con builds.** The
   spec says the run "stops with the provider's error", and there is one only after a
   refusal. Using the same class means a caller catches one type for both forms (§5.6).
5. **E1's transient case is two 500s then a 200**, which recovers in 3 requests. The
   spec's "500 four times, then 200, must still recover" cannot hold at
   `num_retries=2`. It fails after 3 requests (*run*). That is the amendment's own
   trade, made explicit.
6. **E6 runs once as a dispatched experiment job, and the live tier is unchanged.** The
   spec put E6's K = 2 run into the live tier for every dispatch. The Conductor's brief
   keeps `tests/test_live.py` as it is, so all of E6 (K = 1, 2, 4 × 3 seeds) is one test
   file run by a new `experiment` job (§9).
7. **The spec's mock-up of `splits.json`** (`parts` per stage, flat `refused` pairs)
   becomes §5.5's shape. It records the window and margin once, and per document the
   cuts, the parts each call read, and one event per split or stop with its cause, its
   estimate, the tokens of the rest, and the provider's message for a refusal.

### 2.3 Findings for the Conductor

1. **At the floor, the default model has no mapped window.** litellm 1.101 does not map
   `openai/gpt-6-luna`, and 1.104 maps it at 922,000 tokens (*run*). A user on 1.101 to
   1.103 gets the refusal path for the default model. That is correct, but unmeasured.
   Raising the floor is not R2's call.
2. **The approved stop rule can stop a run that more parts would have saved.** Take a
   window of 10,000 tokens, a rest of 6,000 and a refused part of 5,000. The part is
   shorter than the rest, so the run stops, yet a 2,500-token half would have fit. With a
   known window, ruling 1 removed this rule (§2.4). With an unknown window, it stays as
   approved, because nothing else bounds the refused requests there.
3. **The stop note named TASK-36 in a message every user reads.** Ruling 2 removed it.
4. **`ollama/*` is mapped (`ollama/llama3`: 8,192 tokens), but Ollama truncates instead
   of refusing.** Measuring now splits such a document before Ollama would have
   truncated it silently. That is better than today, and still out of R2's scope.

### 2.4 Conductor rulings (2026-10-05, under Michael's waiver)

1. **With a known window, the run stops only on the measured form**: the rest alone is
   over the line. When the rest is under the line, halving always ends, so a part
   shorter than the rest, or over the line, is cut again rather than stopping the run.
   The first version's form (c) and its use of the spec's rule with a known window would
   only stop runs that more splits would save (§2.3 item 2). With an unknown window, the
   spec's rule stays as approved: a refused part shorter than the rest stops. The loop
   bounds itself by halving down to the cut floor (§5.4). E5 keeps both forms.
2. **No internal task ID in user-facing text.** r3con is a public library. The stop note
   names the document and says that the relevant context has outgrown the model's
   window, with no "TASK-36" (§5.6). E5 asserts that wording.
3. **The counting shortcut is adopted.** Every cl100k token is at least one byte, so a
   text's tokens are at most `len(text.encode("utf-8"))`. A request whose rest and part
   together are at most the line in UTF-8 bytes is under the line without calling
   `litellm.encode` (§5.2). Characters are not a bound, because one CJK character can be
   several tokens. Tests pin both sides (§8).

### 2.5 Found while building (Implementer, 2026-10-05)

1. **A 401 raises `litellm.AuthenticationError`**, not `BadRequestError` (§4's table, now
   corrected). It is still sent 3 times. `test_a_refused_key_is_sent_three_times` asserts
   the class litellm raises.
2. **The caller-override test passed 2, the new default.** At `LLM_NUM_RETRIES = 2` it
   would pass with the caller's value ignored. `test_a_callers_retry_count_wins` now
   passes 7.
3. **The widened types land with relevance (step 4), not in steps 5 to 7.** Once
   `RelevantContext.snippets` is `list[Snippet]`, pyright rejects the pipeline's calls to
   `propose_schema`, `parse_documents` and `reason`, and `Answer(relevant_context=…)`. So
   step 4 also widens those three parameters and joins `Answer.relevant_context`; steps 5
   and 7 add only behaviour, and every commit type-checks.
4. **The measured stop names the first part.** When the rest alone is over the line,
   every part is, so the event, the error's message and the note name part 0's kind, and
   the estimate is the rest plus part 0.
5. **`splitting.py` imports `ContextWindowExceededError` from `litellm.exceptions`.**
   pyright reports `litellm.ContextWindowExceededError` as a private import
   (`reportPrivateImportUsage`). It is the same class.
6. **Docstrings §6.7 did not list.** `r3con/__init__.py` ("every document is parsed,
   whole"), `stages/structuring/__init__.py` ("parse each document, whole") and
   `parse_one_document` ("one whole document") now say a document may be read in parts.
7. **E4's measured row is `window(6_000)`, a 5,100-token line.** Measured on the suite's
   fake: the registry is 7,749 tokens, its halves 3,863 and 3,886, its quarters 1,925 to
   1,959; relevance's rest is 692 tokens in round 1 and 828 in round 2; parsing's rest is
   about 1,780 with the 2,000-character notes. §8's inequality then needs a line from
   4,714 to about 5,640.
8. **`test_a_stopped_run_keeps_its_calls_and_splits` stops in parsing.** The schema call
   carries every note under a prompt about 3,600 characters longer than parsing's, and it
   is never split, so the test sizes the run to keep the schema under the limit: two
   5,000-character documents, 3,000-character notes, a 13,500-character limit.
9. **The window lookup silences litellm's provider banner, for its own call only**
   (the Conductor's ruling, after the as-built). For a model string whose provider
   litellm does not recognise (`my-router-alias`), `get_model_info` prints litellm's
   "Provider List" banner to stdout, twice per lookup, unless
   `litellm.suppress_debug_info` is set. 0.2.0 never asked litellm about the model
   string, and printed nothing. `_max_input_tokens` now sets `suppress_debug_info` and
   the `LiteLLM` logger's level around its call and restores both, as
   `runtime/codeact.py`'s `_supports_stop_parameter` does. The window it returns is
   unchanged, mapped or not. `test_looking_up_the_window_prints_nothing` pins it with the
   suite's global setting turned off.
10. **Tests §8 did not name**: `test_a_text_of_one_character_cannot_be_halved`,
    `test_the_measured_stop_says_what_was_estimated_against_which_window` (§5.6's
    message, exactly) and `test_a_split_documents_notes_join_in_order_without_the_empty_ones`
    (`join_parts`). E6's job has a 45-minute timeout, and each E6 case prints one line of
    measurements before it asserts.

## 3 · Order of work

Each step is test-first, and CI is green after every step.

| Step | Item | Lands |
| --- | --- | --- |
| 1 | R2.1 | `LLM_NUM_RETRIES = 2` and its comment; `runtime/llm.py`'s retry comment; E1's tests |
| 2 | R2.2 | `settings.WINDOW_MARGIN_PERCENT`, its floor, ceiling and snapshot key |
| 3 | R2.2 | `splitting.py`: `halve`, then `Splits` (window, parts, `read_in_parts`, the record) |
| 4 | R2.2 | `relevance.py`: `Snippet`, `join_parts`, `render_relevance`, `surface_relevance(splits=)` |
| 5 | R2.2 | `parsing.py`: `parse_documents(splits=)`, the widened types; `schema.py`'s type |
| 6 | R2.2 | `prompts/reasoning/v2.yaml`, `reason`'s flag, `configs/default.yaml` |
| 7 | R2.2 | `pipeline.py`: one `Splits` per run, `Answer.relevant_context` joined |
| 8 | R2.2 | the "no chunking" sentences, `runs.py`'s layout, the README's tree |
| 9 | E6 | `tests/test_experiments.py`, the `experiment` marker, `ci.yml`'s `experiment` job |
| — | proof | one dispatch of `ci.yml` with `live`: every job green. One dispatch with `experiment` for E6. |

## 4 · R2.1 · A refused request is sent 3 times, not 11

**Why.** litellm resends any failed call `num_retries` times, a 400 included, with no
filter. At r3con's 10, a document the provider refuses as too long costs 11 requests,
and so does a wrong key. With R2.2, every halving would pay that.

**What I measured.** A server on 127.0.0.1 answering in vLLM's words, one
`litellm.completion` call on `hosted_vllm/…`, at litellm 1.101 and 1.104 (*run*, same
counts on both):

| Server answers | `num_retries=10` (today) | `num_retries=2` (R2.1) |
| --- | --- | --- |
| 400, context window | 11 requests, `ContextWindowExceededError` | 3 requests, same error |
| 401, wrong key | 11 requests, `AuthenticationError` (§2.5) | 3 requests, same error |
| 500, 500, then 200 | recovers on request 3 | recovers on request 3 |
| 500 four times, then 200 | recovers on request 5 | fails after 3, `InternalServerError` |

**The change.**

- `settings.py`: `LLM_NUM_RETRIES = 2`. The comment says what it is: "How many times
  litellm resends a failed call. It resends any error, a refused request (400) included,
  so each refusal costs this many extra requests." The floor (0) and the snapshot key
  are unchanged.
- `runtime/llm.py:143–145`: the comment that credits the backoff to tenacity is wrong.
  It becomes one line: "litellm resends a failed call `num_retries` times, whatever the
  error; setdefault so a caller can still override." The module docstring's tenacity
  paragraph stays true, because litellm's resend loop still imports tenacity.
- No new signature. A caller's own `num_retries` still wins, through `setdefault`.

**Model-facing text:** none. `num_retries` is litellm's own keyword and never reaches
the provider.

## 5 · `splitting.py` · reading a document in parts

### 5.1 Shape

One new module, `src/r3con/splitting.py`, owns everything R2.2 adds. It holds the
halving rule, the window lookup and estimate, the loop that sends a document's parts and
re-cuts them, the stop rule, and the record. The stages know none of it. Each stage
gives the loop two things, `rest` (the text it sends beside the document) and `send`
(how to send one part), and gets one result per part back. `runtime/llm.py` is untouched
by R2.2.

```python
T = TypeVar("T")


class Splits:
    """Where each document of one run is cut into parts, kept for the rest of the run.

    The window is ``model``'s ``max_input_tokens`` in litellm's model map, looked up
    once; ``line`` is that window less ``margin_percent``. Both are ``None`` when the
    map does not know the model. With a ``task_logger``, every split and stop is
    written to ``splits.json`` in the run folder as it happens.
    """

    max_input_tokens: int | None
    margin_percent: int
    line: int | None

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


def halve(text: str) -> int: ...
```

`Splits` is a class because it holds state across calls: each document's cuts, which
every later stage starts from, and the record. It is thread-safe, because the documents
of a stage run in parallel. Each document's cuts are touched only by its own worker, and
the record and its file are written under a lock.

- **`__init__`.** It copies `documents`. `margin_percent=None` reads
  `settings.WINDOW_MARGIN_PERCENT` now. A value outside 0 to 99 raises
  `ValueError(f"margin_percent must be from 0 to 99, got {value}.")`. Then it looks up the
  window (§5.2) and computes `line = max_input_tokens * (100 - margin_percent) // 100`.
- **`parts(doc)`** returns `documents[doc]` cut at its current cuts:
  `[text[a:b] for a, b in itertools.pairwise([0, *cuts, len(text)])]`, which is
  `[text]` until it is split. The parts never overlap and always rejoin to the document.
- **`read_in_parts`** is §5.4.
- **Exported?** No. The module is imported as `r3con.splitting`; `r3con/__init__.py`
  does not change.

### 5.2 The window and the estimate

**The window** is `litellm.get_model_info(model)["max_input_tokens"]`, read in a private
`_max_input_tokens(model) -> int | None`. What it returns (*run*, local map):

| `model` | litellm 1.101 | litellm 1.104 |
| --- | --- | --- |
| `openai/gpt-6-luna` (the default) | raises `Exception` (unmapped) | 922,000 |
| `anthropic/claude-sonnet-5-5` | 200,000 | 200,000 |
| `gemini/gemini-3.8-flash` | 1,048,576 | 1,048,576 |
| `hosted_vllm/Qwen/Qwen3.5-35B-A3B` | raises `Exception` | raises `ModelNotMappedError` |
| a Router alias (`my-router-alias`), `""` | raises `Exception` | raises `ModelNotMappedError` |
| `ollama/llama3` | 8,192 | 8,192 |
| any of these after `litellm.register_model({model: {"max_input_tokens": 32768, …}})` | 32,768 | 32,768 |
| an entry with `max_tokens` and no `max_input_tokens` | `None` | `None` |

- An unmapped model raises a bare `Exception` at 1.101, and `ModelNotMappedError` (an
  `Exception` subclass) at 1.104. So the lookup catches `Exception`, logs it, and returns
  `None`:
  `except Exception:  # noqa: BLE001 — litellm before 1.104 raises a bare Exception for a model its map lacks`.
  The log line is
  `_log.info("litellm's model map gives no input window for %s; a document is split only when the model refuses it", model)`.
- Only `max_input_tokens` counts. `max_tokens` is often the output limit, so it is not a
  fallback.
- **The window follows the model string, not the transport.** A caller's `completion=`
  or a Router alias is looked up by the string r3con is given. An alias is usually
  unmapped, so it takes the refusal path. An alias that happens to name a mapped model
  is measured with that model's window, and a refusal still splits.
- **A self-hosted model can be measured.** Its user registers the window with litellm,
  for example `litellm.register_model({"hosted_vllm/Qwen/Qwen3.5-35B-A3B":
  {"max_input_tokens": 32768, "litellm_provider": "hosted_vllm", "mode": "chat"}})`.
  r3con adds no knob of its own (Code Guide: the installed library covers it). The
  README does not teach this; the INFO line names the behaviour.
- **Which map.** litellm uses its shipped copy when `LITELLM_LOCAL_MODEL_COST_MAP=True`,
  which is how the suite and CI run. Otherwise it downloads its map on import. A user's
  window can be newer than CI's.

**The estimate** is a private `_count_tokens(text) -> int`, which is
`len(litellm.encode(text=text))`. With no `model`, `litellm.encode` always takes
cl100k_base and never a Hugging Face tokenizer (*read*: `_select_tokenizer_helper`). It
counts the same tokens as `stages/reasoning.py:112`. The request's estimate is
`_count_tokens(rest) + _count_tokens(part)`. Per-message overhead and the response
format's own encoding are left out: closeness is enough, and the margin absorbs them.

**The shortcut (ruling 3).** `_utf8_size(text)` is `len(text.encode("utf-8"))`. Every
cl100k token is at least one byte (*run*: all 100,261 of them), so a text's tokens never
exceed its UTF-8 size. Before counting, the loop compares sizes: a rest whose UTF-8 size is
at most the line is not over it, and a part whose `_utf8_size(rest) + _utf8_size(part)`
is at most the line fits. Either is decided without `litellm.encode`. Only a request
larger than the line in bytes is counted. The rest is counted at most once per call. On
`gpt-6-luna`'s 783,700-token line, nothing short of 783,700 bytes is ever counted.
Characters would be wrong: Japanese contract text runs at 1.29 tokens per character, and
emoji at 2 (*run*). A single character is at most 4 bytes, so at most 4 tokens.

| Measured (*run*) | 1.101 | 1.104 |
| --- | --- | --- |
| loads with no network (`unshare -rn`) | yes | yes |
| first call | under 1 ms | 113 ms (loads lazily) |
| 1,000,000 characters of memo text | 86 ms, 215,807 tokens | 48 ms, 215,807 tokens |
| characters per token (memos) | 4.63 | 4.63 |
| tokens per character (random CJK) | — | 2.35 |

### 5.3 `halve`: where a part is cut

```python
_BREAKS = (
    re.compile(r"\n\s*\n\s*"),  # paragraph break
    re.compile(r"\n"),  # line break
    re.compile(r"(?<=[.!?])\s+|(?<=[。！？])"),  # sentence end
    re.compile(r"\s+"),  # any space
)
```

`halve(text)` returns the offset `c` at which `text` is cut into `text[:c]` and
`text[c:]`:

1. Let `n = len(text)`. If `n < 2`, raise `ValueError`. The loop never calls it on one
   character: that is the cut floor (§5.4).
2. The window is `max(1, n // 4) <= c <= min(n - 1, n - n // 4)`, and the middle is
   `n // 2`.
3. For each pattern in `_BREAKS`, in order, the candidates are the `m.end()` of its
   matches that lie in the window. The first pattern with a candidate wins, and the
   candidate nearest the middle is the cut (the earlier one on a tie).
4. With no candidate at all (a part with no whitespace in its middle half), cut at the
   middle.

The cut lies after the break, so the next part starts with text. Nothing is stripped, so
`"".join(parts) == document` always holds. Halving a level halves every part: the cut
offset is the part's start plus `halve(part)`, and the new cuts merge into the
document's sorted list. *Prototype*: the 40,000-character registry of E2 is cut at
10,091, 20,140 and 30,317.

### 5.4 `read_in_parts`: the loop

`read_in_parts(doc, *, call, rest, send)` returns one result per part of `documents[doc]`,
in order, after splitting the document as far as it must. Its arguments:

- `call` names the call, for example `"relevance-r2"` or `"parse"`. A part's kind is
  `f"{call}-d{doc}"` while the document is whole and `f"{call}-d{doc}c{k}"` for part `k`
  (0-based) once it is split. `runs._DOC_CHUNK_RE` already reads `c{k}` into
  `calls.json`'s `chunk` field.
- `rest` is everything the stage sends beside a part: the rendered system prompt, plus
  the response schema for a parse.
- `send(part, kind)` sends one part and returns its result. It raises
  `litellm.ContextWindowExceededError` when the provider refuses it.

**Halving a level** halves every part of two characters or more and leaves a
one-character part as it is (§5.3). **The cut floor**: a part of one character cannot be
cut, so when such a part is over the line or refused, the run stops.

The loop, **with a known window** (`line` is not `None`), the measured mode:

1. **Measure.** If the rest alone is over the line, stop: the measured form (ruling 1).
   Otherwise find the first part whose request is over the line, using the shortcut of
   §5.2 before counting. If there is none, go to step 2. If that part has one character,
   stop at the cut floor. Otherwise halve the level, record a split with cause
   `"estimate"`, and measure again. Nothing has been sent yet.
2. **Send** the parts in order, through `send`.
3. **On `ContextWindowExceededError` for part `k`**, a refusal the estimate did not
   predict: if the part has one character, stop at the cut floor with the provider's
   error. Otherwise halve the level, and record a split with cause `"refusal"`, the
   provider's message, and `discarded = k` (the accepted parts of this level, now thrown
   away). Then go back to step 1. There is no other stop with a known window: a part
   shorter than the rest is cut again.

The loop, **with an unknown window**, the refusal mode:

1. **Send** the parts in order, through `send`. Nothing is counted.
2. **On `ContextWindowExceededError` for part `k`**, count the part and the rest. If the
   part is shorter than the rest, stop: the spec's rule. If the part has one character,
   stop at the cut floor. Otherwise halve the level, record a split with cause
   `"refusal"` and `discarded = k`, and send again.

In both modes:

- **A stop** records a `"stop"` event, adds the note of §5.6 to the error, and raises
  it. After a refusal, the error is the provider's own. Before sending, r3con builds one.
- **When every part is accepted,** the loop records `parts[call] = len(parts)` for a
  document already in the record, and returns the results.
- **Any other exception** from `send` passes through untouched. It is never split.

**How it bounds itself.** Every halving leaves each part at most ¾ of its parent (§5.3).
So after at most ⌈log₄⁄₃ n⌉ levels every part of an `n`-character document has one
character: 49 levels for a million characters.

- **With a known window**, the rest is under the line once step 1 passes its first
  check. A one-character part is at most 4 tokens, so every part fits by then, unless
  the rest leaves less than 4 tokens of room. That case is the cut floor's stop.
  Measuring sends nothing, so those levels cost only counting.
- **A refusal with a known window** comes when the provider counts more than the
  estimate. It moves the loop down one level, so a document meets at most one refusal
  per level: at most 49 refusals, 3 requests each through litellm, for a million
  characters, before the cut floor.
- **With an unknown window**, the spec's rule stops the loop once a refused part is
  shorter than the rest. Every stage's prompt is far longer than one character, so the
  cut floor is not reached in practice.

**What it costs.** A refusal through litellm is 3 requests (R2.1). A refused part at
level K also discards the parts accepted before it at that level. They stay in
`calls.json` (they were sent), and the event's `discarded` count says how many. A
document read in K parts costs K calls per relevance round and K parse calls, one after
another. Each of its K notes then rides in every later call's prompt. With a known window
and a rest close to the line, K can be large. Take a rest of 90,000 tokens under a
100,000-token line and a 500,000-token document: that is 64 parts, each sent with the
whole rest. Ruling 1 accepts that cost rather than stopping such a run.

### 5.5 `splits.json`

It is written to the run folder's root through `task_logger.write_json("splits", …)`:
on every event, and again whenever a recorded document finishes a call. So a run that
stops keeps it, beside the failing stage's `calls.json` and `error.txt`. A run that
never splits or stops has no `splits.json`. *Prototype*, E2 by refusal:

```json
{
  "model": "hosted_vllm/Qwen/Qwen3.5-35B-A3B",
  "max_input_tokens": null,
  "margin_percent": 15,
  "line": null,
  "documents": {
    "3": {
      "cuts": [10091, 20140, 30317],
      "parts": {"relevance-r1": 4, "relevance-r2": 4, "parse": 4},
      "events": [
        {"call": "relevance-r1-d3", "cause": "refusal", "estimate": 8083, "rest": 704,
         "discarded": 0, "action": "split", "parts": 2,
         "error": "litellm.ContextWindowExceededError: … maximum context length is …"},
        {"call": "relevance-r1-d3c0", "cause": "refusal", "estimate": 4389, "rest": 704,
         "discarded": 0, "action": "split", "parts": 4, "error": "…"}
      ]
    }
  }
}
```

- The window is recorded once, because a run has one model.
- Document keys are 0-based indices, as strings, like `source_docs` and `calls.json`'s
  `doc`.
- `cuts` are offsets into the document where parts 2, 3, … begin.
- `parts` maps each call a recorded document completed to the number of parts it read.
- An event has the kind that was refused or estimated over, its cause (`"estimate"` or
  `"refusal"`), the estimate of the whole request and of the rest in tokens, the number
  of parts discarded, and the action (`"split"` or `"stop"`). `parts` is the count after
  a split, or at a stop. `error` is the provider's message for a refusal, and `null` for
  an estimate.

### 5.6 The stop's error, the note, the logs

**The error.** A stop after a refusal re-raises the provider's
`ContextWindowExceededError`. A stop before sending (the measured form, or the cut floor
in step 1) raises
`litellm.ContextWindowExceededError(message=…, model=model, llm_provider="r3con")`, with
this message:

```text
r3con estimated relevance-r2-d0 at 25,601 tokens, over the 3,400-token line (the 4,000-token input window litellm's model map gives hosted_vllm/window-4000, less 15%); it was not sent.
```

**The note**, added with `error.add_note` before raising. `_recorded_stage` then adds
the run-folder note after it, and the CLI and tracebacks print both:

```text
r3con: reading documents[0] (Document 1) in more parts cannot help in relevance-r2-d0: the part is about 120 tokens and the prompt and notes sent with it about 20,359. The relevant context has outgrown the model's window.
```

The clause after the colon depends on the stop:

| Stop | Clause |
| --- | --- |
| the spec's rule (unknown window) | "the part is about 120 tokens and the prompt and notes sent with it about 20,359" |
| the measured form | "the prompt and notes sent with it are about 20,359 tokens, over the 3,400-token line by themselves" |
| the cut floor | "the part is one character, and the prompt and notes sent with it (about 3,398 tokens) leave it no room" |

The note names no task ID and no internal plan (ruling 2).

**Logs.** A split logs a WARNING on `r3con.splitting`. The CLI shows WARNING by default,
and a library user gets it on stderr through logging's last-resort handler:

```text
relevance-r1-d3: documents[3] (Document 4) was refused as too long; reading it in 2 parts
parse-d3c0: documents[3] (Document 4) is estimated at 5,183 tokens, over the 4,250-token line; reading it in 4 parts
```

An unknown window logs INFO once per `Splits` (§5.2).

## 6 · The stages and the pipeline

### 6.1 `stages/relevance.py`

```python
Snippet = str | list[str]
"""A document's relevance snippet, or, for a document read in parts, one per part."""


def join_parts(snippet: Snippet) -> str: ...


def render_relevance(relevance_snippets: Sequence[Snippet] | None) -> str: ...


def surface_relevance(
    *,
    task: str,
    documents: list[str],
    model: str,
    prompt_version: str,
    rounds: int = 2,
    run: StageRun | None = None,
    workers: int | None = None,
    splits: Splits | None = None,
    **llm_kwargs: Any,
) -> RelevantContext: ...


@dataclass
class RelevantContext:
    snippets: list[Snippet]
    rounds: list[list[Snippet]]
```

- **`join_parts`** returns a `str` as it is. For a list, it returns the stripped
  non-empty notes joined by a blank line, which is `""` when every part had nothing.
- **`render_relevance`** gives a `str` entry `### Document N` exactly as today. It gives
  a list entry one `### Document N.k` section per part, `k` from 1, each with today's
  "(no relevant summary for this task)" for an empty note. A list always means parts,
  even of length 1, which the loop never produces.
- **`surface_relevance`.** `splits=None` builds `Splits(documents, model=model)` (no
  record). A `splits` passed in must be built over the same `documents`: the parts are
  read from it. In round `r`, document `i`'s worker computes
  `others = [join_parts(s) for j, s in enumerate(prev) if j != i]`, then
  `rest = _system_prompt(task, others, prompt_version)`, and calls
  `splits.read_in_parts(i, call=f"relevance-r{r}", rest=rest, send=…)`, where `send`
  calls `relevance_snippet(document=part, other_snippets=others, kind=kind, …)`. One
  result becomes a `str`, and several a `list`. The round's progress count reads
  `join_parts(s)`.
- **A part never sees its own document's other parts.** The others block holds documents
  `j != i` only, each split one joined, without numbers, as today (spec item 5).
- **`_system_prompt(task, other_snippets, prompt_version) -> str`** is the existing
  `load_prompt("relevance", …)` call, moved into a private helper so that
  `relevance_snippet` and the worker render the same text. `relevance_snippet`'s
  signature does not change.
- **`relevance/result.json`** keeps its shape. A split document's entry in
  `rounds[k].snippets` is a JSON list of its part notes.
- **The module docstring's** "assumed to each fit in context (no chunking)" becomes "a
  document too long for the model's window is read in parts (`r3con.splitting`), and its
  snippet is then one note per part".

### 6.2 `stages/structuring/schema.py`

`propose_schema(relevance_snippets: Sequence[Snippet] | None = None, …)`: the type widens,
and nothing else changes. The schema call carries no document, so a refusal there is
raised after litellm's 3 requests, never split (spec scope). Schema prompt v1 shows the
`N.k` headings without explaining them, and is not versioned.

### 6.3 `stages/structuring/parsing.py`

```python
def parse_documents(
    *,
    documents: list[str],
    schema_code: str,
    parse_cls: type[BaseModel],
    task: str,
    prompt_version: str,
    relevance_snippets: Sequence[Snippet] | None = None,
    model: str,
    max_attempts: int | None = None,
    run: StageRun | None = None,
    workers: int | None = None,
    splits: Splits | None = None,
    **llm_kwargs: Any,
) -> ParseResult: ...
```

- `parse_one_document(relevance_snippets: Sequence[Snippet] | None = None, …)`: the type
  widens, and nothing else changes.
- `splits=None` builds `Splits(documents, model=model)`. Before the workers start, it
  computes one `rest`, the same for every document:
  `_system_prompt(task, schema_code, relevance_snippets, prompt_version) + json.dumps(parse_cls.model_json_schema())`.
  The private helper is the existing `load_prompt("structuring/parsing", …)` call, shared
  with `parse_one_document`.
- Document `i`'s worker logs `parse doc i/n` as today, then returns
  `splits.read_in_parts(i, call="parse", rest=rest, send=…)`, where `send` calls
  `parse_one_document(document=part, kind=kind, …)`. A part's validation retries keep
  their kinds (`parse-d3c0-retry-1`). A refusal on a retry splits like any other.
- **Parsing starts from the parts relevance left**, through the shared `splits`, and may
  cut them further. It never renumbers: the headings in its prompt are relevance's.
- **The merge** takes `[(i, p) for i, parses in enumerate(per_doc) for p in parses]`.
  `_merge_with_source_docs` already concatenates several pairs under one index, in
  order. So every record from a part carries the document's index (Q1 (a)), and which
  part it came from is in `calls.json`.
- The docstrings' "no chunking" (module `:4–6`, `parse_documents` `:328`) become "read
  whole, or in parts when too long for the model's window". The log at `:347` drops
  ", no chunking".

### 6.4 `stages/reasoning.py`, `prompts/reasoning/v2.yaml`, `configs/default.yaml`

- `reason(relevance_snippets: Sequence[Snippet] | None = None, …)`: the type widens.
  `reason` passes one more prompt variable,
  `read_in_parts=any(isinstance(s, list) for s in relevance_snippets or ())`. A prompt
  that does not reference it, v1 or a user's overlay, renders as before. The
  variables a prompt version may use are now `task`, `schema_code`, `relevance`,
  `parse_json`, `samples_block`, `parse_block` and `read_in_parts`; the comment above
  them says so.
- `configs/default.yaml`: `reasoning: v2`. The default label reads
  `prompts=(rel=v1,schema=v1,parse=v1,reason=v2)`, split or not. R1.12's check finds
  `v2.yaml` in the package.

`v2.yaml` is `v1.yaml` with one change, at line 16, and v1 is not touched:

```diff
-  {{ relevance }}
+  {% if read_in_parts %}Some documents were too long to read whole and were read in parts, in order: a summary headed **Document N.1**, **N.2**, … covers one part of Document N, and every record from Document N still says `document: N`.
+
+  {% endif %}{{ relevance }}
```

With `read_in_parts` false or absent, v2 renders to v1's text byte for byte, with or
without summaries. With it true, the sentence is its own paragraph between the section's
description and the first heading (*run*).

### 6.5 `pipeline.py`

`run_pipeline`'s signature does not change. After the manifest, it builds one `Splits`
for the run:

```python
splits = Splits(
    documents,
    model=model,
    margin_percent=caps["window_margin_percent"],
    task_logger=task_logger,
)
```

It passes `splits=splits` to `surface_relevance` and `parse_documents`, and
`relevance.snippets` (a `list[Snippet]`) to the schema, parsing and reasoning stages, as
today. `Answer.relevant_context` becomes `[join_parts(s) for s in relevance.snippets]`:
one string per document passed in, a split document's notes joined (spec item 5), and
`Answer`'s docstring says so. The module docstring's "nothing is chunked" becomes "a
document too long for the model's window is read in parts (`r3con.splitting`)". A stop
raises from the stage through `_recorded_stage` like any failure: `<stage>/calls.json`,
`<stage>/error.txt`, and the run-folder note after R2's note.

### 6.6 `settings.py`

```python
# The share of the model's input window kept free when r3con estimates whether a
# request fits; a request estimated over the rest is split before it is sent.
WINDOW_MARGIN_PERCENT = 15

_CEILINGS: dict[str, int] = {"window_margin_percent": 99}
```

- `_FLOORS` gains `"window_margin_percent": 0`.
- `settings_snapshot` gains the key `window_margin_percent`, last.
- After the floor check, a value above its ceiling raises
  `ValueError(f"{KEY} must be <= {ceiling}, got {value}.")`.
- So the manifest's `settings` block records the margin with the caps. The CLI's exit 2
  covers a bad margin, as it does a bad cap.

### 6.7 The rest

- `runs.py`: the module docstring's layout gains `splits.json` ("only when a document was
  read in parts or splitting stopped"), and says that `chunk` in `calls.json` is the part
  index. The comment above `_DOC_CHUNK_RE` (`:71–73`) drops "unused … never chunked".
- `r3con.py`: `run`'s `documents` doc (`:221–223`) says a document too long for the
  model's window is read in parts. The `Raises` entry (`:265–266`) says
  `litellm.ContextWindowExceededError` comes only when the prompt and notes sent beside a
  document leave it no room. `__init__.py:24–26` says each document is read on its own,
  in parts when it is too long.
- `cli.py:41–42`: the help's "each document must fit in the model's context (there is no
  chunking)" becomes "a document too long for the model's context is read in parts".
- `README.md`: the run-folder tree gains `├── splits.json   <- only when a document was
  read in parts`.

## 7 · What a user sees

**Now works**

- **A document longer than the model's window is read in parts** instead of failing the
  run (spec E2). On a model litellm maps (OpenAI, Anthropic, Gemini), the split happens
  before anything is sent. On a self-hosted model, it happens on the provider's refusal,
  or before sending once the model's window is registered with litellm.
- **The answer still cites the caller's own numbers.** `len(result.relevant_context)`
  equals `len(documents)`, every record says `"document": N`, and only the summaries say
  `Document N.1`, `N.2`.
- **Each refused request is sent 3 times, not 11**, and so is a request with a wrong key
  (R2.1). R1.15's `stop` refusal costs 3 requests instead of 11.

**Fails differently**

- **A server that fails 3 times in a row fails the call.** It used to get 10 retries.
- **When the relevant context fills the window, the run stops** with
  `litellm.ContextWindowExceededError` and a note naming the document and saying that
  the relevant context has outgrown the model's window. That happens before sending
  when the window is known. The run folder keeps `splits.json`,
  the stage's `calls.json` and `error.txt`.

**Changes in every run**

- The default label reads `reason=v2`. A run that split nothing sends v1's reasoning text
  byte for byte.
- The manifest's `settings` gain `window_margin_percent: 15`.
- With a known window, a document call is counted only when its request is larger than
  the line in UTF-8 bytes, which on a large window means never (§5.2).

**Breaks**

- **A document estimated between 85% and 100% of a mapped window** was sent whole, and is
  now read in 2 parts. Those runs change. That is the margin's purpose, and
  `WINDOW_MARGIN_PERCENT = 0` restores sending whole up to the full window.
- **`RelevantContext.snippets`, `.rounds` and `relevance/result.json`** can hold a list
  for a document read in parts. A run that reads one in parts raised before R2, unless
  the document was in the 85% to 100% band above. `render_relevance` accepts lists as
  well as strings.
- Nothing is removed from the API.

## 8 · Tests, by file

They use R1's machinery: `FakeLLM` through `completion=`, `llm`, `answering_llm`,
`_isolated`, pytest-socket, and pytest-httpserver on 127.0.0.1. "New" means the test
fails at `cfc17bc`. "Guard" means it passes before and after.

**`conftest.py` additions** (shared by several files):

- `FakeLLM.refuses_over(chars: int) -> FakeLLM`. From then on, a request whose
  messages' contents total more than `chars` characters is recorded in `requests` and
  answered by raising
  `litellm.ContextWindowExceededError(message=f"This model's maximum context length is {chars} tokens. However, you requested {size} tokens in the messages.", model=…, llm_provider="hosted_vllm")`,
  without consuming a scripted reply. It is the spec's "model that refuses above a
  stated size": one request per refusal, since the fake replaces litellm's resend loop.
- Fixture `window`, a function `register(tokens: int) -> str`. It runs
  `monkeypatch.setitem(litellm.model_cost, model, {"max_input_tokens": tokens, "litellm_provider": "hosted_vllm", "mode": "chat"})`
  with `model = f"hosted_vllm/window-{tokens}"` and returns `model`, so `get_model_info`
  maps it at 1.101 and 1.104 (*run*). This fakes litellm's map, the outside world, at its
  boundary. The name carries the size because `get_model_info` caches per name: a
  changed or deleted entry keeps answering its first value (*run*, both versions; only
  `register_model` clears the cache). So one name always means one window, across tests.
- Fixture `grown_registry`, a function `grow(chars: int, at: float = 0.6) -> str`. It
  returns `examples/memos/04_contractor_registry.txt` grown to about `chars` characters
  by deterministic filler paragraphs (procurement and insurance boilerplate, with no
  contractor codes or names and no incident counts). The registry's three code lines and its footer sit at fraction
  `at`, and its heading stays first. E2, E4 and E6 use it.
- Unknown-window tests use `QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"`, which is unmapped
  at both bounds.

| File | Added | Changed |
| --- | --- | --- |
| `test_llm.py` | `test_every_call_asks_litellm_for_two_retries` (new: 10 today) · `test_a_request_refused_as_too_long_is_sent_three_times`, httpserver answering vLLM's 400, raises `ContextWindowExceededError`, `len(httpserver.log) == 3` (new: 11) · `test_a_refused_key_is_sent_three_times`, a 401 (new: 11) | `test_a_transient_server_error_is_retried` (500, 500, 200, 3 requests) stays, as E1's recovery guard |
| `test_settings.py` | `test_a_margin_above_its_ceiling_is_refused` (100 gives `WINDOW_MARGIN_PERCENT must be <= 99, got 100.`) | the snapshot's key list ends with `window_margin_percent`; the floor table gains `("WINDOW_MARGIN_PERCENT", -1, 0)` |
| `test_splitting.py` (new file) | see the list below | — |
| `test_relevance.py` | render rows: `[["a", ""], "c"]` gives `### Document 1.1\na\n\n### Document 1.2\n(no relevant summary …)\n\n### Document 2\nc` · `test_a_refused_document_is_read_in_parts_and_keeps_one_note_per_part` · `test_a_parts_later_round_sees_the_other_documents_notes_but_never_its_own_parts` | — |
| `test_parsing.py` | `test_a_refused_document_is_parsed_in_parts_and_every_record_keeps_its_index` (`source_docs` repeats the index, records in part order) · `test_parsing_starts_from_the_parts_it_is_given` (a pre-split `Splits`: kinds `parse-d0c0`, `parse-d0c1`, no whole request) | `test_progress_is_logged_per_document_at_info` stays as written (guard) |
| `test_reasoning.py` | `test_without_a_part_v2_is_the_v1_prompt_byte_for_byte` (new, since v2 does not exist yet) · `test_v2_says_documents_were_read_in_parts_only_when_a_summary_is_a_part` (a list snippet gives the sentence and `### Document 1.1`; strings give neither; v1 with a list gives the headings and no sentence) | — |
| `test_pipeline.py` | E2 to E5, below · `test_a_stopped_run_keeps_its_calls_and_splits` · `test_the_manifest_records_the_margin` | — |
| `test_config.py` | — | `DEFAULT_PROMPTS` ends `reason=v2`; `test_the_default_config_loads_with_its_shipped_values` expects `reasoning: v2` |
| `test_experiments.py` (new, live and experiment) | E6 (§9) | — |
| `test_runs.py`, `test_cli.py`, `test_live.py` | — | unchanged |

**`test_splitting.py`**, with `send` a plain function that refuses parts over N
characters. It stands for the provider at the loop's boundary, which is the loop's own
parameter:

- `test_a_part_is_cut_at_the_break_nearest_its_middle`, table-driven:
  - a paragraph break beats a nearer line break;
  - a paragraph break outside the middle half loses to a line break inside it;
  - a sentence end when there is no line break;
  - `。` as a sentence end;
  - a space;
  - the exact middle for a text with no whitespace.
- `test_halves_rejoin_to_the_text_and_neither_is_under_a_quarter`, over those texts.
- `test_the_line_is_litellms_window_less_the_margin`: 10,000 tokens at 15 gives 8,500;
  at 0 gives 10,000.
- `test_a_model_litellm_does_not_map_has_no_window_and_no_line`: rows `QWEN`,
  `"my-router-alias"`, `""`.
- `test_a_margin_outside_0_to_99_is_refused`.
- `test_a_document_that_fits_is_sent_whole_under_its_plain_kind`: kinds
  `["parse-d0"]`, no `splits.json`.
- `test_a_refused_document_is_read_in_2_then_4_parts_in_order`: the sent kinds are
  `d0`, `d0c0` (refused), then `d0c0` to `d0c3`. The results are in part order, and the
  record holds 3 cuts and 2 refusal events.
- `test_a_refusal_after_accepted_parts_discards_them_and_rereads_the_level`: `c0`
  accepted, `c1` refused, then `discarded == 1` and 4 parts sent.
- `test_a_document_once_split_starts_the_next_call_from_its_parts`.
- `test_over_the_line_a_document_is_split_before_anything_is_sent` (`window`): no send
  of the whole, and `"estimate"` events.
- `test_a_refusal_splits_even_when_the_estimate_said_it_fits` (a large `window`):
  `"refusal"` event.
- `test_splitting_stops_where_more_parts_cannot_help`, one row per stop:
  - the spec's rule (unknown window, a refused part shorter than the rest);
  - the measured form (`window`, a rest alone over the line);
  - the cut floor (unknown window, `rest="s"`, `send` refusing everything, document
    `"abcd"`).

  Each row asserts:
  - the error is a `ContextWindowExceededError`, the provider's own after a refusal;
  - its first note names `documents[0] (Document 1)`, says "the relevant context has
    outgrown the model's window", and contains no `TASK-`;
  - for the measured form, nothing was sent;
  - the record's last event is a `"stop"`.
- `test_with_a_known_window_a_part_shorter_than_the_rest_is_cut_again` (ruling 1): a rest
  at about 90% of the line and a document three times the room left. The loop splits by
  estimate until every part fits, sends them all, and raises nothing.
- `test_with_a_known_window_a_refused_part_shorter_than_the_rest_is_cut_again`
  (ruling 1): `send` refuses parts over N characters, with N under the estimate's room.
  The loop splits on the refusals and ends with every part accepted.
- `test_a_request_under_the_line_in_bytes_is_not_counted` (ruling 3): `litellm.encode`
  wrapped to record its calls (the outside world's boundary), a memo under
  `window(1_000_000)`. No `encode` call, sent whole.
- `test_a_request_over_the_line_in_bytes_is_counted`, with two rows:
  - ASCII text whose UTF-8 size is over the line but whose tokens are under it.
    `encode` is called, and the text is sent whole, with no split.
  - Japanese text (`"契約書の第三条に基づき、当事者は誠実に協議する。" * k`, 1.29
    tokens per character), sized so that the characters of rest and part are under the
    line and their tokens over it. `encode` is called and the text is split by estimate,
    which a character bound would have missed.
- `test_another_error_passes_through_and_never_splits`: a `BadRequestError` from `send`.
- `test_splits_json_is_written_as_each_split_happens`, with a `TaskLogger`: the file
  exists after a stop.

**E2 to E5 in `test_pipeline.py`.** Each runs `run_pipeline` with a `TaskLogger`, on
`CONFIG` with `reasoning: v2`. Sizes are what the prototype passed with. The relevance prompt is about 3,300 characters in
round 1 and 3,900 in round 2, and the parse prompt about 3,800.

- **E2 · `test_a_refused_document_is_read_in_parts_and_keeps_its_number`.** `QWEN`, the
  five memos with `documents[3] = grown_registry(40_000)`, and `refuses_over(16_000)`.
  It asserts:
  - 5 entries in `relevant_context`;
  - the summary section of the reasoning prompt has headings 1, 2, 3, 4.1 to 4.4 and 5
    (the section only: v1's examples carry headings of their own), and no `Document 6`;
  - v2's sentence is there;
  - every record from the registry says `"document": 4` and is sourced 3;
  - no round-2 request whose user message is a registry part carries another registry
    part's note;
  - each over-limit request appears once;
  - `splits.json` has 3 cuts, 2 refusal events and
    `parts == {"relevance-r1": 4, "relevance-r2": 4, "parse": 4}`.
- **E2, measured · `test_a_document_over_the_line_is_split_before_sending`.**
  `window(5_000)` (line 4,250) and no refusal limit. It asserts the same headings, no
  request over the line, and `"estimate"` as every event's cause.
- **E3 · `test_an_unsplit_run_sends_what_a_v1_reasoning_run_sends`.** `window(1_000_000)`,
  so the estimate runs. The test runs the five memos twice, with configs that differ
  only in `reasoning: v1` or `v2`. Every request's `messages` and `response_format` are
  equal, and there is no `splits.json`.
- **E4 · `test_a_later_stage_splits_further_without_renumbering`.** `refuses_over(25_000)`.
  The registry's part notes are 2,000 characters each, so that parsing's rest exceeds
  relevance's. It asserts `parts == {"relevance-r1": 2, "relevance-r2": 2, "parse": 4}`
  and the headings 4.1 and 4.2 only. The parse kinds in `structuring/parsing/calls.json`
  are `parse-d3c0` to `c3` (plus the other documents'), with no second relevance run,
  and every registry record is stamped 4. This refusal row also needs, in tokens, the
  half the parser is refused larger than parsing's rest (the spec's rule). The prototype
  had about 3,700 tokens against 1,500. A measured row needs, with `R` the registry,
  `rest_relevance + R/2 <= line < rest_parse + R/2` and `rest_parse + R/4 <= line`. With
  a known window, nothing else stops it (ruling 1). The Implementer sizes it with the
  same long notes.
- **E5 · `test_splitting_stops_at_the_relevant_contexts_line`.** `QWEN`, 40 reports of
  about 400 characters, relevance notes of 1,000 characters, `refuses_over(20_000)`, and
  `R3CON_DOC_WORKERS=1`, so the order is fixed. It asserts:
  - `ContextWindowExceededError`, with notes `[<R2's note naming documents[0] (Document 1) and saying "the relevant context has outgrown the model's window">, "r3con: partial artifacts in …"]`, and no `TASK-` in either;
  - no request carries a part of document 0;
  - `splits.json` holds one `"stop"` event and no cuts.
- **E5, measured.** `window(6_000)`: 40 requests (round 1 only). Round 2 is never sent,
  and the note has the same wording, with the measured clause.

## 9 · Experiments and the live tier

| E | Where | Null (false if) |
| --- | --- | --- |
| E1 | `test_llm.py`, hermetic against httpserver in vLLM's words | the 400 or the 401 is sent more than 3 times; 500, 500, 200 does not recover |
| E2 | `test_pipeline.py`, by refusal and by estimate | §8 E2's assertions |
| E3 | `test_pipeline.py` (v1 against v2), plus one diff against `cfc17bc` | any byte of any request's `messages` or `response_format` differs (`num_retries` aside) |
| E4 | `test_pipeline.py` | relevance runs again; headings other than 4.1 and 4.2; parse kinds other than `parse-d3c0` to `c3`; a record not stamped 4 |
| E5 | `test_pipeline.py`, both forms | (unknown window) a refused part shorter than its rest is cut again; (known window) a round-2 request is sent; either run ends in anything but `ContextWindowExceededError` with a note naming the document and saying the relevant context has outgrown the model's window |
| E6 | `test_experiments.py`, dispatched once | below |

**E3 against `cfc17bc`.** The Implementer runs a script once with the suite's
`FakeLLM`, the `answering_llm` replies, the five memos, the default config and one
worker. It runs on `cfc17bc`'s `src/` and on the head's (`PYTHONPATH` to each), dumps
every request without `num_retries`, and diffs. The as-built document reports it. The
prototype gave 17 requests, all identical; the labels differ in `reason=` only.
*Run* (Implementer, head `71ac060`, litellm 1.104): 17 requests on each side, identical
apart from `num_retries` (10 at `cfc17bc`, 2 at the head); the labels are
`default[model=gpt-6-luna,rounds=2,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]` and
`…reason=v2)]`, and the answers are equal.

**E6 · a document read in parts still answers.** `tests/test_experiments.py` holds one
test, marked like `test_live.py` (`live`, `enable_socket`, skipped without
`OPENAI_API_KEY`) and also `experiment`:
`test_a_document_read_in_parts_still_answers[parts-seed]`, parametrized over
`parts in (1, 2, 4)` × `seed in (1, 2, 3)`. Each case:

- reads `examples/memos` with `documents[3] = grown_registry(40_000, at=0.6)`, so that the
  code lines fall in the second half, in part 2 of 2 and part 3 of 4;
- runs `r3con.run(QUESTION, docs, params={"seed": seed}, completion=refusing(limit), logs_dir=tmp_path)`
  with the live tier's question and the default config. `refusing(limit)` raises
  `ContextWindowExceededError` for a two-message request whose user message is over
  `limit` characters, and otherwise calls `litellm.completion`. The limit is none for
  1 part, 24,000 for 2 and 12,000 for 4. `gpt-6-luna`'s mapped window never splits these
  sizes, so the wrapper sets K;
- checks the precondition: `splits.json`'s `parts["relevance-r1"] == parts`, and no
  `splits.json` at 1 part;
- asserts R1.5's answer ("halloran", `\b11\b`), and exactly one record with
  `"document": 4` whose string values mention both `CT-118` and `Halloran`.

The null is that a seed right at 1 part is wrong at 2 or 4, or that the CT-118 to
Halloran record is missing or doubled. The schema is the model's own, so a seed whose
schema has no such record fails the record assertion at 1 part as well. The as-built
document reads that as "no mapping record in this schema", not as the null. It reports
the 3 × 3 table from the run's artifact. If the null shows, the spec's fallback (telling relevance and
parsing "part 2 of 4") is a new prompt version, and it goes back to the Conductor.

**E6, run** (Implementer, once): CI run
[37288918145](https://github.com/michaeltheologitis/r3con/actions/runs/37288918145),
job `experiment`, at `5e716f2`, artifact `experiment-run`.

- **The claim and the rule, as above, before the result.** A document read in 2 or 4
  parts still answers: a seed right at 1 part is right at 2 and 4, with exactly one
  CT-118 to Halloran record from Document 4. Any seed wrong at 2 or 4 parts while right
  at 1, or a missing or doubled record, is the null.
- **Conditions.** `openai/gpt-6-luna` (922,000-token mapped window, so the wrapper alone
  sets the parts), litellm 1.104.0, Python 3.13, the default config (`reason=v2`), seeds
  1, 2 and 3 through `params`, one run per cell, 9 runs in all. The registry is 40,081
  characters (7,749 tokens), cut at 20,092 for 2 parts and at 10,068, 20,092 and 30,017
  for 4; its code lines sit in part 2 of 2 and part 3 of 4. A refused request is refused
  by the wrapper before it leaves the runner, so it costs no tokens. Cost at litellm's
  mapped prices ($0.10 in, $0.50 out per million tokens): $0.061 for the 9 runs, 194
  calls, 267 s of test time. The noise floor, from the three seeds at 1 part: 17 calls
  each, prompt tokens 44,864 to 45,151 (0.6%), and a fourth reasoning turn in one
  2-part seed moves a run by 3 calls and 11,000 prompt tokens.

| Parts | Seed | Parts read (relevance r1, r2, parse) | Refusals | Answer: Halloran, 11 | CT-118 to Halloran records from Document 4 | Calls | Prompt tokens | Completion tokens | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | whole | 0 | yes, yes | 1 | 17 | 44,864 | 2,369 | 19 |
| 1 | 2 | whole | 0 | yes, yes | 1 | 17 | 45,037 | 2,482 | 24 |
| 1 | 3 | whole | 0 | yes, yes | 1 | 17 | 45,151 | 3,110 | 26 |
| 2 | 1 | 2, 2, 2 | 1 | yes, yes | 1 | 20 | 49,322 | 3,003 | 24 |
| 2 | 2 | 2, 2, 2 | 1 | yes, yes | 1 | 23 | 60,816 | 2,989 | 34 |
| 2 | 3 | 2, 2, 2 | 1 | yes, yes | 1 | 20 | 50,032 | 3,293 | 28 |
| 4 | 1 | 4, 4, 4 | 2 | yes, yes | 1 | 27 | 60,963 | 3,428 | 39 |
| 4 | 2 | 4, 4, 4 | 2 | yes, yes | 1 | 26 | 57,927 | 3,477 | 40 |
| 4 | 3 | 4, 4, 4 | 2 | yes, yes | 1 | 27 | 60,644 | 3,382 | 33 |

| Parts | Mean calls | Mean prompt tokens | Mean seconds |
| --- | --- | --- | --- |
| 1 (baseline) | 17.0 (1x) | 45,017 (1x) | 23.0 (1x) |
| 2 | 21.0 (1.24x) | 53,390 (1.19x) | 28.7 (1.25x) |
| 4 | 26.7 (1.57x) | 59,845 (1.33x) | 37.3 (1.62x) |

- **Result: the null does not show.** All 9 runs answer Halloran with 11 and hold exactly
  one CT-118 to Halloran record from Document 4. Every split run's reasoning prompt has
  headings 1, 2, 3, 4.1 to 4.K and 5 and v2's sentence; every unsplit run has neither.
  Parsing read the parts relevance left (`parse-d3c0` to `c{K-1}`, never `parse-d3`).
  Each schema was accepted on its first attempt and every one had a registry field
  (`contractor_registry_entries`, `contractor_mappings` or `contractor_code_mappings`).
  Reasoning took 1 turn in 6 runs, 2 in two 4-part runs, and 4 in one 2-part run.
- **What it does not show.** One run per cell, one model, one question, and a registry
  whose mapping lines sit whole inside one part. A fact cut across a part boundary is
  not tested.

**The CI wiring for E6:**

- `pyproject.toml` registers
  `experiment: a measured run against a real provider, dispatched once; also marked live`.
  `addopts` keeps `-m "not live"`, which already excludes it.
- `ci.yml` gets a `workflow_dispatch` input, `experiment` (boolean, default false).
- The `live` job's command becomes
  `uv run --locked pytest -m "live and not experiment" -rA --basetemp="$RUNNER_TEMP/live"`.
  It selects the same single test as today.
- A new job, `experiment`, has `if: inputs.experiment` and
  `needs: [checks, test, lowest, wheel]`. Its steps are the `live` job's, with
  `-m experiment`, `--basetemp="$RUNNER_TEMP/experiment"`, and an `experiment-run`
  artifact.
- The Conductor dispatches `ci.yml` on the branch's head with `live: false` and
  `experiment: true`, once. That is 9 runs on `gpt-6-luna`, about 200 calls.
  `release.yml` passes only `live: false`, so it never runs the job.

**The live tier is unchanged.** `tests/test_live.py` is not edited. Its run changes in
four ways that do not touch the assertion:

- its label reads `reason=v2`, and the reasoning prompt is v1's text, since nothing
  splits;
- at the lock, `gpt-6-luna` is mapped (922,000), so each document call is estimated (the
  five memos are far under the line);
- its requests carry `num_retries=2`;
- the `live` job's `-m` expression changes as above.

## 10 · Changes to `src/` by module

| Module | Item |
| --- | --- |
| `settings.py` | R2.1 (`LLM_NUM_RETRIES = 2`, comment), R2.2 (`WINDOW_MARGIN_PERCENT`, `_FLOORS`, `_CEILINGS`, snapshot key) |
| `runtime/llm.py` | R2.1 (the retry comment) |
| `splitting.py` (new) | R2.2 (`Splits`, `halve`, `_max_input_tokens`, `_count_tokens`, `_utf8_size`, the stop, the record) |
| `stages/relevance.py` | R2.2 (`Snippet`, `join_parts`, `render_relevance`, `_system_prompt`, `surface_relevance(splits=)`, docstring) |
| `stages/structuring/schema.py` | R2.2 (type) |
| `stages/structuring/parsing.py` | R2.2 (`parse_documents(splits=)`, `_system_prompt`, types, docstrings, log) |
| `stages/reasoning.py` | R2.2 (type, `read_in_parts`, variables comment) |
| `prompts/reasoning/v2.yaml` (new) | R2.2 |
| `configs/default.yaml` | R2.2 (`reasoning: v2`) |
| `pipeline.py` | R2.2 (one `Splits` per run, `Answer.relevant_context`, docstring) |
| `runs.py`, `r3con.py`, `cli.py`, `__init__.py` | R2.2 (docstrings and help only) |

Outside `src/`: `tests/` (§8), `pyproject.toml` (the marker), `.github/workflows/ci.yml`
(§9), `README.md` (one line), and this file. No version bump is designed here. A release
would be 0.3.0: new behaviour, a setting and a prompt version, and nothing removed.

## 11 · Outside r3con, and risks

**Depends on** (no collaborator's code):

- **litellm 1.101 to 1.104:**
  - `get_model_info` returns `max_input_tokens` for a mapped model and raises an
    `Exception` for an unmapped one, the 1.104 `ModelNotMappedError` included.
  - `model_cost` entries and `register_model` are read by `get_model_info`, which
    caches per model name. `register_model` clears that cache, and a direct
    `model_cost` edit does not.
  - `encode(text=)` is cl100k_base, offline.
  - `ContextWindowExceededError(message, model, llm_provider)` is a `BadRequestError`
    and takes notes.
  - `num_retries` resends any error.
  - A provider's context refusal maps to `ContextWindowExceededError` for vLLM's and
    OpenAI's words (spec, verified), and for Anthropic's, Gemini's and llama.cpp's
    (*read* by the spec).
- **jinja2:** an undefined variable in `{% if %}` is false.
- **Python:** `itertools.pairwise` (3.10+) and `BaseException.add_note` (3.11+).

**Risks, and what would show them:**

- **The estimate is cl100k.** Claude's and Gemini's tokenizers can count more tokens than
  cl100k. The 15% margin absorbs a tokenizer up to 1/0.85, about 18% hungrier. Beyond
  that, the request is refused and split, at 3 requests per refusal. `splits.json`
  shows it as a `"refusal"` event on a model with a known window. CJK text is
  over-counted (2.35 cl100k tokens per character on random CJK), so it splits early,
  which is safe.
- **Counting near the line.** The shortcut leaves only requests larger than the line in
  UTF-8 bytes to count. When round 2's rest (every other note) is itself near the line,
  every call is counted, which is O(N²) characters per round. At 1,000 documents with
  500-character notes, that is about 25 s of CPU per round, at a size where the
  relevant context is near the window anyway.
- **Many parts near the line (ruling 1).** With a known window and a rest just under
  the line, a document is cut into as many parts as the remaining room needs, each a
  call carrying the whole rest (§5.4). `splits.json`'s `parts` shows it.
- **Refusals with a known window** mean the provider counts more than the map's window
  less the margin allows. Each such refusal moves the loop down a level, so there is at
  most one per level, at 3 requests each. That is at most 49 for a million characters,
  before the cut floor. `splits.json` shows them as `"refusal"` events on a known
  window.
- **A `CUSTOM_TIKTOKEN_CACHE_DIR` that is empty and offline** makes `litellm.encode`
  raise at the first document call on a mapped model, as it already does in reasoning's
  parse guard (R1b removed that fallback). It is not handled.
- **A refusal litellm does not map** (a plain `BadRequestError`) is raised after 3
  requests and never split, as the spec states.
- **A server that truncates** (Ollama's default) never refuses. A mapped window still
  measures (§2.3 item 4).
- **E6's null** ends R2.2 as designed (spec §4), and it comes back to the Conductor.
- **The window lookup sets a litellm global for the length of its call** (§2.5 item 9).
  A litellm call another thread makes during that lookup also runs with
  `suppress_debug_info` on. r3con builds its `Splits` before any stage's workers start,
  so only a caller's own concurrent litellm calls can meet it, and the setting only
  silences litellm's debug banners.

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
| R2.2 | A new module, `r3con/splitting.py`. Before a relevance or parse call, a document whose request is estimated over 85% of the model's mapped window is cut in 2, then 4, … at paragraph breaks. Any document the provider refuses with `ContextWindowExceededError` is cut the same way. Parts are numbered N.1, N.2, … in the summaries only. | A document too long for the model is read in parts instead of failing the run. `splits.json` in the run folder says how. When the relevant context itself fills the window, the run stops with a note naming the document and TASK-36. | nothing that worked before. `RelevantContext.snippets` entries can now be lists, which only happens in a run that used to fail. | `prompts/reasoning/v2.yaml`: v1 plus one sentence, shown only when a summary is a part. `configs/default.yaml` pins it, so the default label reads `reason=v2`. |

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
   in three forms (§5.4, step 4). The spec's form, a refused part shorter than the rest of
   its request, applies to a refusal. The amendment's measured form, the rest alone over
   the line, applies before sending. A third form is needed so that measuring always
   ends: a part over the line that is already shorter than the rest also stops.
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
   needed. The rule is unchanged: a part shorter than everything sent with it.
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
   shorter than the rest, so the run stops, yet a 2,500-token half would have fit. The
   rule fires only when the rest is more than half the window, which is where R3 begins.
   I kept it as approved.
3. **The stop note names TASK-36 in a message every user reads.** E5's null requires it.
   When R3 ships, the note should name what to do instead of a task ID.
4. **`ollama/*` is mapped (`ollama/llama3`: 8,192 tokens), but Ollama truncates instead
   of refusing.** Measuring now splits such a document before Ollama would have
   truncated it silently. That is better than today, and still out of R2's scope.

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
| 401, wrong key | 11 requests, `BadRequestError` | 3 requests, same error |
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

| Measured (*run*) | 1.101 | 1.104 |
| --- | --- | --- |
| loads with no network (`unshare -rn`) | yes | yes |
| first call | under 1 ms | 113 ms (loads lazily) |
| 1,000,000 characters of memo text | 86 ms, 215,807 tokens | 48 ms, 215,807 tokens |
| characters per token (memos) | 4.63 | 4.63 |
| tokens per character (random CJK) | — | 2.35 |

Every cl100k token is at least one byte (*run*: all 100,261 tokens), so the estimate is
never above the request's UTF-8 size.

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

1. Let `n = len(text)`. If `n < 2`, raise `ValueError` (the stop rule ends splitting long
   before this).
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

The loop:

1. **Measure (only when `line` is known).** Count the rest once per call. If the rest
   alone is over the line, stop (form b below). Otherwise find the first part whose
   `rest + part` is over the line. If that part is shorter than the rest, stop (form
   c). Otherwise halve every part, record a split with cause `"estimate"`, and measure
   again. Nothing has been sent yet.
2. **Send.** Send the parts in order, through `send`.
3. **On `ContextWindowExceededError` for part `k`,** count the part and the rest. If the
   part is shorter than the rest, stop (form a). Otherwise halve every part and record a
   split with cause `"refusal"`, the provider's message, and `discarded = k` (the
   accepted parts of this level, now thrown away). Then go back to step 1, so the new
   level is measured before it is sent.
4. **The stop rule**, in three forms, in tokens:
   - (a) a refused part is shorter than the rest (the spec's rule);
   - (b) the rest alone is over the line (the amendment's measured form), checked
     before any part is sent;
   - (c) a part over the line is shorter than the rest. This is (a) predicted, and it is
     what makes measuring end when the rest is just under the line.

   A stop records a `"stop"` event, adds the note of §5.6 to the error, and raises it.
   For (a) the error is the provider's own. For (b) and (c) it is one r3con builds.
5. **When every part is accepted,** record `parts[call] = len(parts)` for a document
   already in the record, and return the results.

Any other exception from `send` passes through untouched. It is never split.

**Why it ends.** Every halving leaves each part at most ¾ of its parent (§5.3), while
the rest stays the same. Within a bounded number of levels, either every part fits or
one is shorter than the rest, which is a stop. The rest is never empty, because every
stage's prompt has text.

**What it costs.** A refusal through litellm is 3 requests (R2.1). A refused part at
level K also discards the parts accepted before it at that level. They stay in
`calls.json` (they were sent), and the event's `discarded` count says how many. A
document read in K parts costs K calls per relevance round and K parse calls, one after
another. Each of its K notes then rides in every later call's prompt.

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

**The error.** Form (a) re-raises the provider's `ContextWindowExceededError`. Forms (b)
and (c) raise
`litellm.ContextWindowExceededError(message=…, model=model, llm_provider="r3con")`, with
this message:

```text
r3con estimated relevance-r2-d0 at 25,601 tokens, over the 3,400-token line (the 4,000-token input window litellm's model map gives hosted_vllm/tiny, less 15%); it was not sent.
```

**The note**, added with `error.add_note` before raising. `_recorded_stage` then adds
the run-folder note after it, and the CLI and tracebacks print both:

```text
r3con: reading documents[0] (Document 1) in more parts cannot help in relevance-r2-d0: the part is about 120 tokens and the prompt and notes sent with it about 20,359. The relevant context has outgrown the model's window; r3con does not shrink it yet (TASK-36).
```

For form (b), the clause after the colon is "the prompt and notes sent with it are about
20,359 tokens, over the 3,400-token line by themselves".

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
  record). In round `r`, document `i`'s worker computes
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
`Answer`'s docstring says so. The
module docstring's "nothing is chunked" becomes "a document too long for the model's
window is read in parts (`r3con.splitting`)". A stop raises from the stage through
`_recorded_stage` like any failure: `<stage>/calls.json`, `<stage>/error.txt`, and the
run-folder note after R2's note.

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
  `litellm.ContextWindowExceededError` and a note naming the document and TASK-36. That
  happens before sending when the window is known. The run folder keeps `splits.json`,
  the stage's `calls.json` and `error.txt`.

**Changes in every run**

- The default label reads `reason=v2`. A run that split nothing sends v1's reasoning text
  byte for byte.
- The manifest's `settings` gain `window_margin_percent: 15`.
- With a known window, each document call's request is counted. That costs about 50 ms
  per million characters (§11).

**Breaks**: nothing that worked before. `RelevantContext.snippets`, `.rounds` and
`relevance/result.json` can hold a list for a document read in parts, which only happens
in a run that used to raise. `render_relevance` accepts lists as well as strings.

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
  by deterministic filler paragraphs (procurement and insurance boilerplate, no codes,
  no names, no counts). The registry's three code lines and its footer sit at fraction
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
- `test_splitting_stops_where_more_parts_cannot_help`, one row per form (a), (b), (c):
  - the error is a `ContextWindowExceededError`, the provider's own for (a);
  - its first note names `documents[0] (Document 1)` and `TASK-36`;
  - for (b) and (c) nothing was sent;
  - the record's last event is a `"stop"`.
- `test_another_error_passes_through_and_never_splits`: a `BadRequestError` from `send`.
- `test_splits_json_is_written_as_each_split_happens`, with a `TaskLogger`: the file
  exists after a stop.

**E2 to E5 in `test_pipeline.py`.** Each runs `run_pipeline` with a `TaskLogger`. Sizes
are what the prototype passed with. The relevance prompt is about 3,300 characters in
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
  and every registry record is stamped 4. A measured row needs
  `rest_relevance + R/2 <= line < rest_parse + R/2` and `R/2 > rest_parse`, in tokens;
  the Implementer sizes it with the same long notes.
- **E5 · `test_splitting_stops_at_the_relevant_contexts_line`.** `QWEN`, 40 reports of
  about 400 characters, relevance notes of 1,000 characters, `refuses_over(20_000)`, and
  `R3CON_DOC_WORKERS=1`, so the order is fixed. It asserts:
  - `ContextWindowExceededError`, with notes `[<R2's note naming documents[0] (Document 1) and TASK-36>, "r3con: partial artifacts in …"]`;
  - no request carries a part of document 0;
  - `splits.json` holds one `"stop"` event and no cuts.
- **E5, measured.** `window(6_000)`: 40 requests (round 1 only). Round 2 is never sent.

## 9 · Experiments and the live tier

| E | Where | Null (false if) |
| --- | --- | --- |
| E1 | `test_llm.py`, hermetic against httpserver in vLLM's words | the 400 or the 401 is sent more than 3 times; 500, 500, 200 does not recover |
| E2 | `test_pipeline.py`, by refusal and by estimate | §8 E2's assertions |
| E3 | `test_pipeline.py` (v1 against v2), plus one diff against `cfc17bc` | any byte of any request's `messages` or `response_format` differs (`num_retries` aside) |
| E4 | `test_pipeline.py` | relevance runs again; headings other than 4.1 and 4.2; parse kinds other than `parse-d3c0` to `c3`; a record not stamped 4 |
| E5 | `test_pipeline.py`, both forms | a part shorter than its rest is cut again; the run ends in anything but `ContextWindowExceededError` with a note naming the document and TASK-36 |
| E6 | `test_experiments.py`, dispatched once | below |

**E3 against `cfc17bc`.** The Implementer runs a script once with the suite's
`FakeLLM`, the `answering_llm` replies, the five memos, the default config and one
worker. It runs on `cfc17bc`'s `src/` and on the head's (`PYTHONPATH` to each), dumps
every request without `num_retries`, and diffs. The as-built document reports it. The
prototype gave 17 requests, all identical; the labels differ in `reason=` only.

**E6 · a document read in parts still answers.** `tests/test_experiments.py` holds one
test, marked `live` and `experiment`:
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
document reads that as "no mapping record in this schema", not as the null. The as-built document reports the 3 × 3 table
from the run's artifact. If the null shows, the spec's fallback (telling relevance and
parsing "part 2 of 4") is a new prompt version, and it goes back to the Conductor.

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
| `splitting.py` (new) | R2.2 (`Splits`, `halve`, `_max_input_tokens`, `_count_tokens`, the stop, the record) |
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
- **Counting costs O(N²) characters per relevance round on a mapped model.** Round 2's
  rest holds every other note. At 1,000 documents with 500-character notes, that is
  about 25 s of CPU per round, at a size where R3's line is near anyway. If it shows, a
  request whose UTF-8 size is at most the line can skip counting, because a cl100k
  token is at least one byte. I left that out: write less until it is needed.
- **A `CUSTOM_TIKTOKEN_CACHE_DIR` that is empty and offline** makes `litellm.encode`
  raise at the first document call on a mapped model, as it already does in reasoning's
  parse guard (R1b removed that fallback). It is not handled.
- **A refusal litellm does not map** (a plain `BadRequestError`) is raised after 3
  requests and never split, as the spec states.
- **A server that truncates** (Ollama's default) never refuses. A mapped window still
  measures (§2.3 item 4).
- **E6's null** ends R2.2 as designed (spec §4), and it comes back to the Conductor.

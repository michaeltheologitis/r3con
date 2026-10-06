# TASK-36 · R3 as built: notes that fill the window are read again shorter

Cartographer, for TASK-36; restated by the Refactorer after the literate refactor. This
is what exists at `92c06d7` on `claude/tender-shannon-eq4lq7-r3` (r3context 0.3.0), the
last commit to touch `src/` or `tests/`, checked against
`design/r3-notes-fit-the-window.md`. The base is `main` at `f4dae00`, r3context 0.2.0.

**Behaviour is `07aa539`'s**, which §4's measurements ran. `src/` did not change from
`56f7389`, the release commit, to `07aa539`; the commits between touch E7's harness,
`ci.yml`'s seed input and the design, so both E7 runs ran that code. The refactor,
`42735c8` to `92c06d7`, changed how the code reads and nothing it sends or writes: run
offline against `3d3ebd2`'s `src/`, seventeen runs (E1 to E6, the refused parse, the
40 reports' three, the v1 stop's two, the floor's two, R2's parse stop, a grown
registry) and ten direct `reason` calls give byte-identical requests (`messages`,
`response_format`), log records, `notes.json`, `splits.json`, `result.json` files,
`calls.json` kinds and outcomes. §2 says where code moved; §4.2 counts the head's tests.

**Evidence.** *[run]* means executed: the suite, an offline probe, or a CI run whose
artifacts I downloaded and recomputed. *[read]* means read in the code and not executed.
The probes ran each version's own `src/` (`f4dae00` extracted with `git archive`) with
the suite's `FakeLLM` as `completion=` and sockets disabled; nothing reached a provider.
Their corpus is E3's: the five memos plus quiet site memos, notes of about 33 words, the
suite's answering replies. Sizes below are that corpus's, not the design's §1.1 corpus's.

**Reading it.** §1 is what a 0.2.0 user sees differently. §2 is the map of the code. §3
is the divergences from the design. §4 is the measured results, E7 and the Conductor's
ruling on it in §4.4. §5 is what I could not verify.

## 1 · What a 0.2.0 user sees in 0.3.0

**In every run**

- **The label reads `rel=v2`**, because the default config pins relevance prompt v2, so
  0.3.0's runs group apart from 0.2.0's. An ordinary run sends 0.2.0's requests byte for
  byte: 17 of 17 identical for the five memos on the default model, on a mapped
  1,000,000-token window and on a mapped 8,192-token one; 362 of 362 for 120 documents
  on the default model [run]. v2's one extra sentence renders only under a word budget.
- **Nothing is removed.** `r3con.run`, `run_pipeline` and the CLI take what they took;
  the rest of the public surface only gains (§2.3) [read].

**Now works**

- **A collection whose notes outgrow the window answers.** Every document's note rides
  in every later request, so the notes grow with the collection. When a request would
  not fit and its notes are the bigger part of it, r3con reads the relevance round that
  wrote them again, each note asked to stay under W words, and sends the stage again. W
  starts at half the notes' mean length, halves again if needed, and is never under 10.
  On a mapped window this is decided by estimate before anything is sent; on an
  unmapped one, on the provider's refusal. On an 8,192-token window whose provider
  refuses over it, 120 documents answer in 482 requests, round 2 read again under 16
  words, where 0.2.0's reasoning is refused after 361; 200 answer; 215 stop (§4.3) [run].
- **Reasoning shows the whole parse only when its first turn fits.** Otherwise it shows
  one sample record per field, the view 0.2.0 kept for huge parses, and the whole parse
  stays bound as `parse` in the sandbox [read]. A whole-parse first turn the provider
  refuses is sent once more with samples (E2) [run].
- **A document over the line only because its notes are bigger is no longer cut**; the
  notes are read again instead. The registry grown to 12,000 characters among 100
  documents: 0.2.0 reads it in 4 parts, 0.3.0 whole (§4.3) [run].
- **The user is told.** Each read again logs one WARNING on `r3con.notes`, and the run
  folder gains `notes.json` (the window, then every event). In `relevance/`, a round
  read under a budget has `max_words`, and its calls are tagged `relevance-r2-w16-d7`
  [run]:

  ```text
  reasoning: the notes of 120 documents come to about 5,366 tokens, and the request with them to about 8,330, over the 6,963-token line; reading round 2 again with each note under 16 words
  ```

**Changes some runs**

- **A first reasoning turn between 85% and 100% of a mapped window** was sent with the
  whole parse and now shows samples: 58 documents on 8,192 tokens, 6,969 tokens whole
  against 5,364 with samples [run]. Both of E7's wrong answers came with the sample view
  (§4.4).

**Fails differently**

- **When even 10-word notes cannot fit, the run stops** with
  `litellm.ContextWindowExceededError` and a note; on a mapped window, before anything is
  read again. Never with `NotesTooLong`: that leaves neither `run_pipeline` nor the CLI
  [run]. The note, for 215 documents:

  ```text
  r3con: even at 10 words each, the notes of 215 documents would take about 3,618 tokens, over the 3,524 that reasoning leaves them, so reading them shorter cannot help. The relevant context has outgrown the model's window.
  ```

- **A config that pins relevance v1** gets the reasoning view and never reads notes
  again: it stops at its first notes trouble, with a note naming v2. On a mapped window
  that can be earlier than 0.2.0 fails: 120 documents stop after 240 requests, where
  0.2.0 sends 362, the last over the window (§3, item 11) [run].
- **Notes in a language written without spaces** count as one word each, so no halving
  shortens them; the run stops at the first notes trouble with the floor's note, which
  then speaks of "10 words each" [run, `Budget.shorten` alone].

**For a direct caller of a stage**

- **Without a `Budget`**, every stage behaves as in 0.2.0, except that `reason` builds a
  `Splits` for its model when given none, so it shows samples when the whole parse would
  not fit: on a mapped 3,000-token window, 2,607 tokens where 0.2.0 sent 5,647 [run].
- **With a `Budget`**, `propose_schema`, `parse_documents` and `reason` raise
  `r3con.notes.NotesTooLong`, a plain `Exception`, before sending or from the provider's
  refusal, for the caller to hand to `budget.shorten` [run]. `surface_relevance` reads its own rounds again and lets none
  out [run: suite].

**Costs**

- **Reading again costs a round**: at 8,192 tokens in E7, 1.33 to 1.34 times the calls
  and 1.20 to 1.24 times the prompt tokens of the same seed on the shipped window [run].
- **The suite takes three times as long**: 65.8 s against 22.3 s for `f4dae00` on this
  machine, 45 s of it the new tests, 13 s the 400-document E5 alone [run]. Near the line
  each request is counted twice, by the pre-pass and by the splitter [read].

## 2 · The map

### 2.1 The seam R3 adds

A request carries a document part, or none, and the notes. `r3con.splitting.Splits`
decides which is bigger. A bigger part is cut, as in R2; bigger notes raise
`r3con.notes.NotesTooLong`. Two loops handle it, `surface_relevance`'s round loop and
`run_pipeline`'s `fitting`; both hand it to `Budget.shorten`, which sets W or raises
the stop, then read a round again and send again. `NotesTooLong` is r3con's own
`Exception`, so R2's `except ContextWindowExceededError` around a document never takes it
for a refusal of that document (`test_a_notes_signal_raised_by_send_passes_through_uncut`).
**The `Budget` is the switch**: a stage given none passes no notes to the splitter and
measures nothing new [read].

What crosses it: `NotesTooLong` carries the request's name, why (`"estimate"` or
`"refusal"`), its notes, and the `room` they have under the line (`None` after a
refusal). The `Budget` holds W, the run's `Splits`, whether the relevance prompt can ask
for a length, and `notes.json`'s record.

### 2.2 Module by module

- **`notes.py`** (new, 271 lines with docstrings): `MIN_NOTE_WORDS = 10`,
  `NotesTooLong` (a dataclass over `Exception`), `count_notes`, `Budget`. `Budget.shorten` tries W at half, a quarter, …
  of the current W (of the notes' mean when none is set), floored at 10, keeping only
  values below the current W and below the notes' mean. By estimate it takes the first
  at which the notes, each cut at W words, fit every room it is given; after a refusal,
  the first. With none left, or a prompt that cannot ask, it stops: the
  provider's refusal, or r3con's own error with the request's message, raised from
  `None`, with a note saying why (`_why_not`).
- **`splitting.py`**: `read_in_parts(notes=)` compares notes with the part at R2's two
  decision points, before sending and after a refusal; a tie goes to the part. `measure`
  runs before a stage fans out: it cuts by estimate, collects every document whose notes
  must shrink, and raises the one with the least room, so W is sized for the tightest
  request. `check_notes` serves the requests without a document (the schema call,
  reasoning's first turn); `over_line` counts tokens only when bytes cannot settle it.
  `window` is the model and its window, the fields `splits.json` and `notes.json` open
  with.
- **`stages/relevance.py`**: rounds are read by `read_round(r)`, which, when round r's
  requests hand over round r−1's notes, shortens, reads round r−1 again (recursively)
  and reads r again. `final_check`, reasoning's first turn as far as it is known after
  relevance, runs after the last round and beside a round that did not fit. `reread=`
  reads only the last round again. `RelevantContext.words` holds each round's W.
- **`prompts/relevance/v2.yaml`** is v1 plus one sentence under `{% if max_words %}`;
  `configs/default.yaml` pins it. `takes_word_budget(version)` is whether a prompt
  renders differently with a budget, so a user's overlay without the sentence gets v1's
  stop [read].
- **`stages/structuring/schema.py`, `parsing.py`**: with a budget, the schema call is
  checked before every attempt and a refusal carrying notes becomes `NotesTooLong`;
  parsing measures every document before sending any.
- **`stages/reasoning.py`, `runtime/codeact.py`**: `reason` counts the parse once, applies
  the flood guard (`REASONING_PARSE_MAX_TOKS`) itself, renders the first turn with the
  whole parse and renders it again with `_sample_view` when `splits.over_line` says it is
  over the line; 0.2.0's `_render_parse_for_codeact` is gone. One `check_notes` partial
  serves the check before sending and the one after a refused samples turn. `run_codeact(on_first_turn_too_long=)` lets `reason` swap the
  system prompt after a refusal of the first turn: once, to samples; with samples
  already shown, the hook hands the notes over (with a budget) or re-raises. Only the
  first turn; the `stop` retry stays. `check_first_turn` is the lower bound relevance
  checks against.
- **`pipeline.py`**: one `Budget` per run, given to every stage. `fitting(send, run)`
  wraps the schema, parsing and reasoning calls: on `NotesTooLong` it shortens, reads
  the last round again inside relevance's record (`_recorded_stage(run=)` appends to
  `relevance/calls.json`), and sends **that stage** again. The stages before it keep what
  they produced from the longer notes [read]. `Answer.relevant_context` is the notes
  reasoning saw.

**Where the complexity sits**: in the control flow of reading again, where the stage
that raises decides which round is read again and which stage is sent again (relevance's
recursion and its final check, the pipeline's `fitting`), and in `Budget.shorten`'s
choice of W against several rooms. The pipeline's +160 −64 lines are mostly its three
stage calls moving into `fitting` [read].

### 2.3 Public surface, from the code

- `r3con.notes`: `MIN_NOTE_WORDS = 10`;
  `NotesTooLong(message, *, model, call, cause, estimate, notes, notes_tokens, room=None, refusal=None)`;
  `count_notes(notes) -> int`;
  `Budget(splits, *, can_shorten=True, task_logger=None)` with `.words`, `.splits` and
  `.shorten(trouble, *, round_idx, discarded=0, also=()) -> int`.
- `r3con.splitting.Splits`: `.model` (was private); `.window`; `read_in_parts(..., notes=())`;
  `measure(docs, *, call, rests)`; `check_notes(*, call, request, notes, refusal=None)`;
  `over_line(*texts)`, the tokens when over the line, else `None`.
- `r3con.stages.relevance`: `surface_relevance(..., budget=None, reread=None, final_check=None)`,
  which raises `ValueError` for `reread` or `final_check` without a budget;
  `relevance_snippet(..., max_words=None)`; `RelevantContext.words`, defaulting to
  empty; `note_texts(snippets)`; `takes_word_budget(prompt_version)`.
- `propose_schema(..., budget=None)`; `parse_documents(..., budget=None)`;
  `reason(..., splits=None, budget=None)`;
  `check_first_turn(*, task, relevance_snippets, prompt_version, budget)`;
  `run_codeact(..., on_first_turn_too_long=None)`, a callable from the refusal to the new
  system prompt.
- `r3con/__init__.py` exports nothing new. `notes.json`: `model`, `max_input_tokens`,
  `margin_percent`, `line`, and `events`, each with `call`, `cause`, `estimate`,
  `notes`, `room`, `sized_for`, `discarded`, `action` (`"read again"` or `"stop"`),
  `round`, `words` and `error` [run].

## 3 · Divergences from the design

The design's §2.5 records ten places the build departs from it, and the sections it
names already say what the code does. I checked each; item 11 is new, and this commit
adds it to the design (§2.5 item 11, with §2.2 item 4 and §8 corrected).

| § 2.5 | What the code does | Why | Checked |
| --- | --- | --- | --- |
| 1 | The relevance-v1 stop's note names the request that did not fit, not the binding one | the binding one is reasoning's check, which never asked for a shorter prompt | [run]: the v1 probe's note names `reasoning`, the trouble |
| 2 | A fifth stop: notes asked for 10 words that still overshoot them | no other row is true of it | [run]: suite |
| 3 | Every W is below the last W | otherwise a model overshooting at 10 words is asked for 10 forever | [read] `_halvings`; [run] suite |
| 4 | A stop event keeps the round it would have read again, `words` null | — | [run]: probes |
| 5 | E5's room is 2,599 tokens in the suite, 2,571 in the design | the suite asks "Who?" | [run]: suite |
| 6 | Tests the design did not name | each pins a line no named test reaches | [run]: every test the design names exists |
| 7 | src +1,074 −175 in 12 files; tests +1,640 −34 at `56f7389`, +1,645 −34 at `07aa539`; after the refactor, src +1,051 −183, tests +1,647 −38 | forecast: 500 to 650 and about 600 | [run] |
| 8 | litellm 1.101 does not map `openai/gpt-6-luna`; E7 needs the locked 1.104 | — | not run; CI's lowest-bounds job is green |
| 9 | E7's null shows at 8,192 | §4.4 | [run]: recomputed |
| 10 | E7's harness records a copy of each request | the loop's live list changed under it | [run]: the first run's nine rows recompute to the design's |
| **11** | **A relevance-v1 run stops at its first notes trouble, which on a mapped window can be the check after relevance**: 120 documents stop after 240 requests, before the schema call, nothing sent over the line; 0.2.0 sends 362, reasoning's first turn at 11,909 tokens over the 8,192 window. On an unmapped window it stops at reasoning one request after 0.2.0 (the samples turn, refused too) | the design's mechanism (every `shorten` stops when the prompt cannot ask; relevance's final check) does this; the design's "stops where R2 stopped" was measured on §9.2's 40 reports only, where both stop at the same request | [run] |

## 4 · Measured results

### 4.1 The proof

- **The suite** at `07aa539`: 478 passed, 19 deselected (the live tier and the
  experiments), 65.8 s; `f4dae00`: 395 passed, 22.3 s, both on this machine
  (`uv run --locked pytest`) [run].
- **Push CI** 37364699244 at `07aa539`: green on its third attempt. The jobs cancelled
  in the first two had never been given a runner; no step failed [run: jobs read].
- **The live tier** 37385149637 at `07aa539`: green. From its artifact: r3con 0.3.0, the
  default config at `rel=v2`, `reason=v2`; 18 calls; no `notes.json`, no `max_words`,
  no budget sentence; reasoning shown the whole parse; the answer "Halloran Services
  Ltd’s sites logged the most Q3 equipment incidents, with 11 in total." [run]

### 4.2 The new tests, by what they pin

The head collects 84 test ids that `f4dae00` does not, and retires 2 (the renamed v1
stop's rows) [run]:

| Item | Ids | Where |
| --- | --- | --- |
| R3.1: the first turn's view, the retry with samples | 12 | `test_codeact.py` 6; `test_reasoning.py` 4; E1, E2 in `test_pipeline.py` |
| R3.2: W, the stop, `notes.json`, the exception | 20 | `test_notes.py` 16; `test_pipeline.py::test_no_stop_leaves_a_run_as_notes_too_long` 3; `test_cli.py` 1 |
| R3.2: the bigger part is halved | 15 | `test_splitting.py` |
| R3.2: v2's sentence, rounds read again | 12 | `test_relevance.py` |
| R3.2: the stages hand their notes over | 13 | `test_schema.py` 5; `test_parsing.py` 3; `test_reasoning.py` 5 |
| R3.2: the run reads again and sends again | 11 | `test_pipeline.py`: E3, E4, E5, the 40 reports (3 rows), discarded calls, the notes reasoning saw, a failure while reading again, the v1 stop (2 rows) |
| E6: an ordinary run's requests | 1 | `test_pipeline.py` |

E7 adds 9 experiment ids (3 windows × 3 seeds by default).

### 4.3 E1 to E6, and 0.2.0 against 0.3.0

E1 to E6 pass in the suite, with the design's §10.1 sizes: E3 482 requests and one event,
`("reasoning", "estimate", "read again", 2, 16)`; E5 400 requests, all of round 1, and the
floor's note [run]. **E6 against `f4dae00`**: every request's `messages` and
`response_format` identical in four ordinary runs (§1) [run].

The probes of §1, side by side, on an 8,192-token mapped window ("refusing": its provider
refuses over it, as a real one would). The fakes' answers carry no meaning; the rows
compare what each version sends [run]:

| Run | 0.2.0 | 0.3.0 |
| --- | --- | --- |
| 58 documents | 176 requests; first turn whole, 6,969 tokens | 176; samples, 5,364 |
| 60 | 182; whole, 7,129 | 182; samples, 5,464 |
| 70, refusing | answers; whole, 7,925 | not run |
| 120, refusing | refused at reasoning, 362 requests | answers, 482; one read again |
| 120, relevance v1 | 362; first turn 11,909, over the window | stops after 240, by the check after relevance |
| 120, relevance v1, unmapped, refusing | refused at reasoning, 362 | stops at reasoning, 363 |
| 100, registry of 12,000 characters | registry in 4 parts; 306; first turn 10,622 | registry whole; round 1 again under 16 words; 402; samples, 5,369 |
| 200, refusing | not run | answers, 802; round 1 again under 10 words |
| 215 and 230, refusing | not run | stop after round 1 (215 and 230 requests) |

Where 0.2.0 stops depends on the size of the parse: here its parse is one short row per
document and it answers at 70 documents; the design's corpus, with a two-list schema,
gave about 55.

### 4.4 E7 · shortened notes still answer, against the real model

- **Claim and rule, as the design set them before the runs.** Shortened notes still
  answer. The null: a seed right on the shipped window and wrong at 16,384 or 8,192, or
  the CT-118 to Halloran record missing or doubled. Precondition: a read again at 8,192
  and none elsewhere.
- **Conditions.** `openai/gpt-6-luna` through litellm 1.104.0, its window registered
  in-process (the provider never refuses); the default config; the five memos and 115
  quiet site memos; `params={"seed": s}`; 8 documents at a time; the live tier's
  question. Run 37352518283 at `1d8e0a3`, seeds 1 to 3 (its job failed on a harness bug
  fixed in `15212ff`; the nine runs completed); run 37358511251 at `b337d30`, seeds 1 to 6
  (`e7_seeds`), whose job reports 1 failed and 26 passed: R2's E6 9 of 9, E7 17 of 18.
  Both commits' `src/` is the head's. E7 cost about $1.81 and $3.64 at litellm's mapped
  prices, from the runs' tokens. Reproduce: dispatch `ci.yml` on the branch with
  `experiment: true` and `e7_seeds`.
- **Read from**: every case's run folder in the two `experiment-run` artifacts,
  recomputed by a script kept outside the repo; all 27 rows match the design's §10.1.

| Window (line) | Right | Read again | Notes over W, of 120 | Reasoning saw | Largest request | Calls, prompt tokens (vs shipped) |
| --- | --- | --- | --- | --- | --- | --- |
| shipped, 922,000 (783,700) | 9 of 9 | none | — | the whole parse | 13,817 to 17,653 | 362 to 363 calls |
| 16,384 (13,926) | 9 of 9 | none | — | samples | 8,405 to 8,835 | 1.00 to 1.01x, 0.97 to 1.02x |
| 8,192 (6,963) | **7 of 9** | round 2, once, under 16 or 17 words, by estimate, sized for reasoning | 34 to 54 | samples | 6,118 to 6,343 | 1.33 to 1.34x, 1.20 to 1.24x |

The CT-118 to Halloran record is there once in all 27. **The two wrong answers** are
8,192 seed 2 of the first run (22) and 8,192 seed 3 of the second (16) [run]:

- **The notes kept the facts** in all nine runs at 8,192: each 16- or 17-word note on
  Northgate says 5 (or five) incidents under CT-118, Riverside's says 6, and the
  registry's maps CT-118 to Halloran.
- **The parse kept a site's breakdown beside its total** in 5 of the 27 runs (Northgate's
  5 beside 3 and 2; in two of them Riverside's 6 beside 4 and 2 as well): 2 on the shipped
  window, 1 at 16,384, 2 at 8,192. That is the schema the run proposed, not the window.
  Without such records, 22 of 22 runs answered 11.
- **With them**, the 2 runs shown the whole parse answered 11; of the 3 shown samples,
  the 16,384 run answered 11 after four turns, and both 8,192 runs summed every CT-118
  record (5 + 3 + 2 + 6 + 4 + 2 = 22; 5 + 3 + 2 + 6 = 16). The sample view shows one
  record per field, so the prompt does not show reasoning that the parse holds sub-counts
  as records of their own.

**The Conductor's ruling** (Michael waived his gates): **R3.2 stands.** The shortened
notes, 16 or 17 words each, kept every fact in all nine runs at 8,192. Both wrong answers
came where the proposed schema kept a site's breakdown beside its total *and* reasoning
saw R3.1's sample view, which hides those records; shown the whole parse, reasoning never
summed them. R3.1 still does better than 0.2.0, which cannot answer these runs at 8,192: it
shows reasoning the whole parse, and the shipped-window runs, whose requests are 0.2.0's
(E6), put that first turn at 13,817 to 17,653 tokens [run]. The sample view's weakness is
filed as its own task, at Medium.

## 5 · What I could not verify

- **A real provider's refusal of a reasoning turn** reaching `run_codeact` as
  `ContextWindowExceededError` (design §12): the suite's refusals are the fake's litellm
  errors, and no live run refused [read].
- **litellm 1.101's map** (§3, item 8) [not run]; the CI job at the lowest bounds is green.
- **`Budget` used from one thread**: the stages raise in worker threads and
  `parallel_map` carries the exception to the caller, which alone calls `shorten`
  [read].
- **A user's reasoning overlay that renders `parse_json`** rather than `parse_block`
  keeps the whole parse when R3.1 switches to samples, since only `parse_block` changes
  [read].
- **How often samples cost the answer** beyond the breakdown case: E7 has 3 runs where
  the parse kept a breakdown and reasoning saw samples.

# TASK-36 · R3 design: when the notes fill the window, they are read again shorter

System Designer, for TASK-36. It works from the
[approved spec](https://app.notion.com/p/3f062fb2223781848cbcd181ae01a6db), approved at
Gate A by the Conductor in Michael's place: Q1 (a), read the round again with each note
asked to stay under W words; Q2 (a), only when a request does not fit and the notes are
the bigger of what it carries. Branch `claude/tender-shannon-eq4lq7-r3`, cut at `main`
`f4dae00` (r3context 0.2.0: R1, R1b and R2, released). Line numbers below are at
`f4dae00`. It builds on R2's design and as-built (`design/r2-oversize-documents.md` and
`as_built/r2-oversize-documents.md` on `claude/tender-shannon-eq4lq7-r2`).

**Revisions**

- 2026-10-05 · first version.
- 2026-10-05 · the Conductor's ruling (§2.4): `NotesTooLong` is r3con's own exception,
  not a `ContextWindowExceededError`, and a stage hands notes over only when it is given a
  `Budget`. Stop trusting the first version's §2.1 item 2, `NotesTooLong`'s base class
  (§5.1), `propose_schema(splits=)` (now `budget=`), `surface_relevance` building a
  default budget, and the hazard text in §7.3. §3, §4.2, §5, §6, §7, §8, §9, §11 and §12
  follow; §12 gains the known limits.
- 2026-10-05 · the build (Implementer, `a17196b` to `56f7389`): §2.5 lists where the
  code departs from this design and why; §5.3 and §5.4 say what the code does; §10.1
  holds E6's diff against `f4dae00` and E7's measured result.
- 2026-10-05 · E7, run once (run 37352518283): **its null shows at 8,192 tokens for one
  seed of three** (§10.1, §2.5 item 9). By §10's rule R3.2 as designed goes back to the
  Conductor.
- 2026-10-05 · E7 again, six seeds, at the Conductor's call (run 37358511251): 17 of 18
  right; the null shows again at 8,192 (seed 3). In both runs every wrong answer is a
  parse that kept sub-counts as records of their own, seen by reasoning as samples
  (§10.1, §2.5 item 9).

**Reading it.** §1 and §2 are the Gate B read: what changes, what was measured, and every
place this design decides something the spec left open or departs from it, with the
Conductor's ruling in §2.4. §4 is R3.1.
§5 is the new module that holds the notes' budget, §6 the rule R3 adds to R2's splitter,
and §7 wires both into the stages and the pipeline. §8 is what a user sees, §9 the tests
by file (the R2 tests that change are in §9.2), §10 the experiments, §11 the changes to
`src/` by module and the version, and §12 the outside dependencies and risks.

**Evidence.** *Run* means I executed it on a scratch copy of `f4dae00`, with a prototype
of §4 to §7 (about 690 lines of `src/`, no docstrings) and the tests of §9.4. Runs went
through `run_pipeline` with a fake `completion=` and models registered in litellm's map;
nothing reached a provider. *Read* means from the code, not executed. The corpus for every
measured row is §1.1's: the five memos plus quiet site memos, notes of about 200
characters, and a two-list schema like the real run's.

**Where this file lives.** `design/` at the repo root, on the task branch only, as R1's,
R1b's and R2's did. pytest reads `tests/`, the wheel carries `src/`, pyright checks `src/`.
`ruff format` formats the Python blocks inside Markdown, so every Python block here is
valid, formatted Python; after editing, run `uv run ruff format design/`. The PR split
leaves `design/` out of the stack.

## 1 · What R3 changes, in one screen

| Item | The change | What a user sees | What breaks | Model-facing text |
| --- | --- | --- | --- | --- |
| R3.1 | Reasoning shows the whole parse only when its first turn, with it, is estimated under the line. Otherwise it shows the sample records it already shows for a huge parse. A first turn refused as too long while the whole parse was shown is sent once more with the samples. | A run whose reasoning prompt filled the window now answers. The full parse stays bound as `parse` in the sandbox. | A run whose first reasoning turn was estimated between 85% and 100% of a mapped window, sent whole before, now shows samples. | none |
| R3.2 | A new module, `r3con/notes.py`, holds the notes' budget. When a request does not fit, the bigger of what it carries is halved: the document part, as R2 does, or the notes, by reading the round that wrote them again with each note asked to stay under W words. | Runs whose notes outgrow the window answer, with shorter notes, where they stopped or were refused. `notes.json` in the run folder says what was read again. When even 10-word notes cannot fit, the run stops with a note saying so. | nothing in the API. R2's stop hands over to R3 where the notes are the bigger part. A document whose request is over the line only because its notes are bigger is no longer cut. | `prompts/relevance/v2.yaml`: v1 plus one sentence, shown only under a budget. `configs/default.yaml` pins it, so the default label reads `rel=v2`. |

**The rule, in one sentence** (R3.2): when a request does not fit, the bigger of what it
carries is halved, the document part into parts or the notes by reading their round again
half as long, by estimate as many times as it takes before anything is read again, or
once per refusal.

Four rules shape R3:

- **The caller's index space never moves.** Every document keeps its `### Document N` and
  R2's `N.k`, every record its `"document": N`, and `relevant_context[i]` describes
  `documents[i]`. A round read again keeps one note per document (or part).
- **Measure when the window is known, and trust a refusal always.** Before a stage sends
  anything, its requests are estimated with R2's window, line and counter. A refusal hands
  the notes over even when the estimate said the request fits.
- **Once short, notes stay short.** W is one value per run. Every round read after it is
  set, and every round read again, is read under it, and it only ever falls.
- **The budget is the switch.** A stage hands its notes over only when it is given a
  `Budget`, and the signal, `NotesTooLong`, is r3con's own exception, caught only by
  `surface_relevance`'s round loop and the pipeline's stage loop. `r3con.run` always
  passes one; a stage called without one takes 0.2.0's path (§2.4).
- **An ordinary run sends what 0.2.0 sends**, byte for byte. Measuring sends nothing, the
  default model's line (783,700 tokens) is never reached, and v2's sentence renders to
  nothing without a budget (E6, *run*: 17 of 17 requests identical to `f4dae00`, and
  identical between `rel=v1` and `rel=v2`).

### 1.1 Measured: where a growing corpus fails, before and after

*Run.* `run_pipeline` on the default config, the five memos plus N − 5 site memos of about
230 characters that log no incidents, alternately under CT-204 and CT-311 (so the answer
cannot change). Relevance notes of about 200 characters (33 words, about 45 tokens), a
fake that keeps to a word budget when asked, and a fake provider that refuses over the
window in cl100k tokens. "Mapped" registers the window in litellm's map; "unmapped" leaves
it out, so only the refusal says the request is too long. One request per refusal (the
fake replaces litellm's resends; a real provider costs 3).

| N | Window | Today (`f4dae00`) | R3 (prototype) |
| --- | --- | --- | --- |
| 50 | 8,192, mapped | answers, 152 requests; reasoning's first turn is sent at 7,291 tokens, over the 6,963-token line and under the window | answers, 152 requests; the first turn shows samples, 5,064 tokens |
| 60 | 8,192, mapped | reasoning's first turn refused at 8,261 tokens, after 181 requests | answers, 182 requests; the first turn shows samples, 5,562 tokens (E1) |
| 60 | 8,192, unmapped | the same refusal | answers, 183 requests: the whole-parse turn is refused once, the samples turn accepted (E2) |
| 108 | 8,192, mapped | reasoning refused at 12,917 tokens, after 325 requests | answers, 434 requests; round 2 read again once, each note under 16 words |
| 120 | 8,192, mapped | R2's stop before parsing, 241 requests | answers, 482 requests; round 2 read again once, under 16 words, sized for reasoning's first turn (E3) |
| 120 | 8,192, unmapped | reasoning refused at 14,081 tokens, after 361 requests | answers, 485 requests; reasoning refused with samples, round 2 read again under 16 words, reasoning sent again (E4) |
| 200 | 8,192, mapped | R2's stop before round 2, 200 requests | answers, 802 requests; round 1 read again under 10 words, round 2 read under 10 |
| 230, 250, 400 | 8,192, mapped | R2's stop before round 2, after round 1 | stops before reading anything again, after round 1: 10-word notes would not fit reasoning's first turn (E5) |
| 280 | 32,768, mapped | answers; the first turn is sent at 29,601 tokens, over the 27,852-token line | answers; the first turn shows samples, 16,526 tokens |
| 120 + a 3,000-character registry | unmapped, refusing over 7,700 | — | `parse-d3` refused with its notes the bigger part; 46 accepted parse calls discarded and kept in `calls.json`; round 2 read again under 16 words; parsing sent again; answers, 530 requests |
| 100 + a 12,000-character registry | 8,192, mapped | R2 cuts the registry in 2, then 4; reasoning refused at 12,364 tokens, after 305 requests | answers, 402 requests; round 1 read again under 16 words, because the notes (4,430 tokens) are bigger than the registry (about 2,500); the registry is read whole (§2.3 item 2) |

- **The reach.** On an 8,192-token window, R3 answers up to about 200 of these memos, where
  0.2.0 answers up to about 55. The spec estimated 230 and 52; the difference is the
  schema and parse sizes of its corpus.
- **Which call binds.** Reasoning's first turn is the biggest request the notes ride in at
  every N, so the final notes are sized for it (§2.1 item 3). Round 2's calls bind for the
  round-1 notes. The schema call never binds with the shipped prompts.

## 2 · Decisions and divergences

### 2.1 What the spec deferred, decided here

1. **Where the budget lives.** A new module, `r3con/notes.py`, holds the exception that
   hands notes over (`NotesTooLong`), the run's budget (`Budget`: W, the choice of W, the
   stop, `notes.json`), and the two counting helpers. The rule that compares a part with
   its notes stays in `splitting.py`, beside R2's halving, because it is one decision
   about one request: halve the part or hand the notes over (§6). `notes.py` imports
   nothing from `splitting.py` at run time, so there is no cycle.
2. **How a stage hands the notes' refusal to the pipeline.** It raises `NotesTooLong`,
   r3con's own exception in `notes.py`, a plain `Exception` (the Conductor's ruling,
   §2.4). It is raised before sending (cause `"estimate"`) or from the provider's refusal
   (cause `"refusal"`, the refusal as its `__cause__` and its `refusal`), and only by a
   stage given a `Budget`. `surface_relevance`'s round loop and the pipeline's stage loop
   catch it and hand it to `Budget.shorten`, which reads the notes again or raises the
   stop, a plain `ContextWindowExceededError` (§5.3). So none escapes `r3con.run`,
   `run_pipeline` or the CLI; a direct caller who passes a `Budget` can see it, and the
   stages' Raises say so (§7.6).
3. **Whether the final notes are checked once after relevance or at each stage's start.**
   Both. After the last round, the notes are measured against reasoning's first turn as
   far as it is known then: its prompt with the notes and the task, no schema and no
   records (`reasoning.check_first_turn`, §7.4). That is a lower bound, so it never
   shortens notes that would fit, and it is the binding request at every N measured.
   Then each stage measures its own requests exactly before sending (the schema call at
   every attempt, every parse request before the fan-out, reasoning's first turn after
   R3.1's choice) and hands the notes over if one does not fit. *Run*: without the check
   after relevance, the final notes are first sized at the schema call, which is smaller
   than reasoning's turn, so 230 memos read round 1 again, then round 2 again for the
   schema, then stop at reasoning, after 1,151 requests. With it (and §2.2 item 2), they
   stop after round 1, and E3 reads round 2 once either way.
4. **The words-to-tokens conversion.** There is none. The estimate of the notes at W
   words cuts each note the request carries at its W-th word and counts that, followed
   by the paragraph break every rendering puts after a note (§5.2). *Run*: scaling the
   notes' tokens by W over their mean words under-counts short notes, whose first words
   (a site's name, a code, a number) are the dense ones; 400 memos then read round 1
   again before stopping, which E5's null forbids.
5. **The 10-word floor.** `MIN_NOTE_WORDS = 10`, a constant in `notes.py`, not a setting:
   it is part of the algorithm, as `halve`'s quarter is, and `notes.json` records every W
   asked. No note is asked for fewer. A halving that would go below 10 asks for 10 once;
   when 10 cannot fit (by estimate) or was already asked (by refusal), the run stops.
6. **The kinds of the calls read again.** A round read under a budget is called
   `relevance-r{r}-w{W}`, so its calls are `relevance-r2-w16-d7` and, for a part,
   `relevance-r2-w16-d7c1`. `runs._DOC_CHUNK_RE` already reads `doc` and `chunk` out of
   both (*run*). R2's `splits.json` keys a split document's `parts` by this call name.
7. **The shape of `notes.json`.** §5.4. It is written only when notes were read again or
   the run stopped for them, on every event, so a run that stops keeps it beside R2's
   `splits.json`.
8. **How `relevance/result.json` keeps a round read again.** Each round's entry holds the
   notes that round last produced, the ones later rounds and stages read, and gains
   `"max_words": W` when it was read under a budget. An earlier read of the same round is
   in `relevance/calls.json` (the outputs of its calls). An ordinary run's file does not
   change shape.
9. **Whether a stage sent again waits for its in-flight calls.** It does, through R2's
   `parallel_map`: the first failure propagates once every started call has finished, and
   calls not started are cancelled. The finished calls stay in the stage's `calls.json`,
   and the event's `discarded` counts them (*run*: 46 parse calls in §1.1's refused row).
   With a known window nothing is discarded, because every stage measures all its
   requests before sending any (§6.2).
10. **How reasoning learns the line (R3.1).** `reason(..., splits=)`, as
    `surface_relevance` and `parse_documents` already take one; `None` builds
    `Splits([], model=model)`, which looks the window up. R3.1 does not need a `Budget`:
    a direct caller of `reason` gets it too. The refusal path needs a seam
    in the CodeAct loop: `run_codeact(on_first_turn_too_long=)`, a callable that gets the
    first turn's refusal and returns the system prompt to send the first turn with
    instead, or raises (§4.1). The loop stays generic: it knows nothing about parses.

### 2.2 Where this design departs from the spec's text

1. **W is the notes' mean halved as many times as the estimate needs, not the largest W
   that fits.** The spec's §2 says the measured path "picks the W that fits from the
   room left". This design halves, as the spec's rule sentence and title say, and lets the
   estimate skip halvings: 120 memos are read under 16 words, where the spec's mock-up
   says 22. The reason: the final notes are sized before the schema and the records
   exist, so notes sized exactly to the line overflow it at reasoning by the schema and
   the samples, and a second round is read again. Halving leaves room for them (E3 reads
   round 2 once, *run*), and it mirrors R2, which halves a document by estimate before
   sending.
2. **A round's notes that must shrink are also sized for reasoning's first turn**, with
   them standing in for the final notes, because every later round is read under the same
   W (§7.2). The spec checks the final notes only after the last round. Without this, 200
   memos read rounds 1 and 2 again (1,002 requests), and 230 memos read round 1 again and
   round 2, then stop (690 requests); with it, 802 requests and a stop after round 1
   (*run*). It is what makes the spec's "with a known window this is known before
   anything is read again" true.
3. **`notes.json` records no tokens after the read.** The spec lists "the notes' tokens
   before and after". The event records the notes' tokens when they did not fit, the room,
   and W; the notes read again are in `relevance/result.json` with their `max_words`, and
   E7 computes how many came back over W from there. One record of the notes, not two.
4. **A config that pins relevance v1 gets R3.1 and not R3.2.** v1 cannot ask for a length,
   so reading again would send the same prompt. When a request's notes do not fit, such a
   run stops where R2 stopped (the same number of requests, *run*), with a note naming the
   reason and v2. The spec does not say what happens to a v1 config.
5. **R2's unit stop rows in `test_splitting.py` do not change.** The spec expects them
   rewritten. They pass no notes, which is still R2's case exactly; new rows pin the
   notes (§9.3). The pipeline tests that do change are in §9.2.
6. **E5's error names the request about to be sent, and its note the request that binds.**
   400 memos: `r3con estimated relevance-r2-d3 at 19,147 tokens, …; it was not sent.`,
   then `r3con: even at 10 words each, the notes of 400 documents would take about 6,733
   tokens, over the 2,571 that reasoning leaves them, so reading them shorter cannot
   help. …` (*run*). The spec's mock-up names round 2's prompt; the binding request is
   reasoning's.
7. **R2's unknown-window stop remains where the part is bigger than the notes.** The spec
   says R2's stop "remains only where the stage's own prompt leaves no room". Precisely:
   with an unknown window, a refused part shorter than everything sent with it still
   stops the run when the part is bigger than the notes, as R2's §2.3 item 2 records,
   because nothing else bounds the refusals there.

### 2.3 Findings for the Conductor

1. **The spec's size is a third of the measured one.** It estimated about 175 lines of
   `src/` (R3.1 25, R3.2 150). The prototype is about 690 changed lines without
   docstrings, 44 of them the prompt file and much of the rest the pipeline's stages
   moving into a loop. Expect 500 to 650 lines of `src/` and about 600 of tests, which
   moves the Gate C reading budget.
2. **"Halve the bigger" can choose the costlier action.** When a document is long but
   shorter than the notes, R3 reads a round again (N calls) where R2 would have cut the
   document (one more call per stage). 100 memos and a 12,000-character registry on an
   8,192-token window: R3 answers with 402 requests; R2 with R3.1 alone would answer with
   about 306 (estimated from today's run, which is refused only at reasoning). This is
   Q2 (a) as approved, and it reads the registry whole; it is named because it costs more.
3. **Near R3's ceiling, the notes are what the run has left.** At 200 memos on 8,192
   tokens every note is read under 10 words. E7 measures whether such notes still answer.
4. **Notes in a language written without spaces cannot be sized.** A word is what
   `str.split()` separates, so a Japanese note counts as a word or two, no halving can
   shorten it, and the run stops at the first notes trouble with the floor's note. R2
   would have stopped there too. Out of R3's scope; named so nobody is surprised.
5. **`examples/options.ipynb` shows stale labels** (`rel=v1`, `reason=v1`) since R2. R3
   does not re-run it.

### 2.4 Conductor ruling (2026-10-05, under Michael's waiver)

The Conductor accepted this design with one ruling: **`NotesTooLong` is not a subclass of
litellm's `ContextWindowExceededError`.** R2's `read_in_parts` catches
`ContextWindowExceededError` to cut a document, so a notes signal of that class can be
taken for a document's refusal wherever an `except` sits around a check, now or after a
later edit. It is r3con's own exception, in `notes.py`, raised and caught only between
r3con's stages and the loops that read notes again. `r3con.run` and the CLI see a stop
only as a plain `ContextWindowExceededError` with its note (§5.3). A direct caller of a
stage that is handed a `Budget` can see `NotesTooLong`, and the stage's Raises say so.

What follows from it here:

- **The `Budget` is the switch** (§7). A stage given one passes its notes to the splitter,
  measures before sending and hands its notes over; a stage given none passes no notes,
  so `read_in_parts` is R2's exactly, and sends as 0.2.0 does (R3.1 aside). The first
  version had `surface_relevance` build a default budget and every stage pass its notes
  always; a direct caller then met R3's behaviour without asking for it.
- **`propose_schema(budget=)`** replaces the first version's `propose_schema(splits=)`:
  the schema call needs the window only for its notes, and a `Budget` holds the run's
  `Splits` (`Budget.splits`). `check_first_turn` takes the budget too.
- **`surface_relevance` is named in the ruling, and it lets no `NotesTooLong` out**: its
  round loop catches the ones its rounds and its `final_check` raise. Its Raises name the
  stop instead (§7.6).
- **The tests that pinned the subclass** now pin the opposite (§9.3):
  `NotesTooLong` is not a `ContextWindowExceededError`, `read_in_parts` passes one raised
  by `send` through uncut, and no stop leaves `run_pipeline` or the CLI as one.
- **Everything else stands as the first version had it:** halving W rather than the
  largest W that fits; both checks, after relevance against reasoning's first turn and
  each stage's own; `MIN_NOTE_WORDS` as a module constant; `notes.json`'s fields; a
  relevance-v1 config getting R3.1 and stopping where R2 stopped, with a note naming v2;
  the cost case of §2.3 item 2 (Q2 (a) as approved); 0.3.0 as the last step; E7's 8
  workers and 90-minute job. Notes in languages written without spaces join the known
  limits (§12).

### 2.5 Divergences found in the build (Implementer, 2026-10-05)

Built test-first in §3's order, `a17196b` to `56f7389`. Where the code departs from the
sections above, the sections now say what the code does; each item names what the design
said, what the build found, what it does, and what it costs.

1. **The relevance-v1 stop names the request that did not fit** (§5.3, first row). The
   design said every stop note names the binding request, the one with the least room
   among `trouble` and `also`, while §9.2 expected `relevance-r2-d0` and 39 documents. In
   a run, `also` holds reasoning's first-turn check, whose room is smaller, so the binding
   request would be reasoning's, which never asked for a shorter prompt. The first row
   names `trouble`; the others still name the binding request. Cost: none; it is what
   §9.2 and the prototype did.
2. **A fifth stop: notes asked for 10 words that still overshoot them** (§5.3). With W
   already 10 and a model that wrote longer notes, a request over the line by estimate has
   no level left, and none of the four rows is true of it (its notes cut at 10 words would
   fit; nothing was refused). Its note names the floor: `r3con: reasoning is estimated
   over the line, and the notes of 20 documents it carries are already about 14 words
   each, with 10 the fewest a note is asked for, so reading them shorter cannot help. …`
   (`test_notes_asked_for_ten_words_that_still_overshoot_stop_with_the_floor_named`).
3. **Every W is under the last W, in the code as in the claim** (§5.2 item 3). The levels
   are kept below `start` as well as below the notes' mean: `start // 2` raised to 10 is
   `start` itself when `start` is 10, and a model that overshoots at 10 words would
   otherwise be asked for 10 again, without end. §5.2's "How it bounds itself" assumed it.
4. **A stop event keeps its round** (§5.4): the round the stop would have read again (E5:
   round 1), with `words` null.
5. **E5's room is 2,599 tokens in the suite, 2,571 in §5.2's example** (§9.4). The
   design's numbers come from the harness, which asks the live question; the suite asks
   "Who?", 28 tokens shorter across reasoning's prompt and user message. The 6,733 tokens
   the notes take at 10 words are the same.
6. **Tests the design did not name**, each pinning a line no named test reaches:
   `test_a_samples_turn_over_the_line_hands_over_its_notes_before_sending` (§4.2 step 3,
   `reason`'s own check); `test_a_failure_while_reading_again_is_recorded_by_both_stages_named_once`
   (§7.5, the run-folder note once; a provider failure inside the read again takes the
   path a stop does); a third row of `test_notes_that_do_not_fit_are_read_again_shorter`,
   `by-refusal-of-the-schema` (17,000 characters, so the schema call is the one refused;
   246 requests), since neither designed row sent the schema's hand-over through the
   pipeline. `test_the_relevant_context_is_the_notes_reasoning_saw` runs on a refused
   parse, where the pipeline's loop, not relevance's, reads the notes again. The fake
   records its refusals (`FakeLLM.refused`, for E4), and `quiet_sites` and
   `within_budget` are fixtures, as `grown_registry` is.
7. **Size** (§2.3 item 1). `src/`: +1,074 −175 lines in 12 files, docstrings included
   (`notes.py` 291); tests: +1,640 −34 in 13 files, against a forecast of about 600. The
   default suite takes about 70 s locally, from 27 s (E5 alone 14 s), against a forecast
   of about 35 s more. 478 tests, from 395: 85 new, and §9.2's renamed one.
8. **litellm 1.101 does not map `openai/gpt-6-luna`.** The lowest-bounds suite is offline
   and passes; E7 registers its windows on the model's mapped entry, so it needs the
   locked litellm (1.104), which the experiment job uses.
9. **E7's null shows at 8,192 tokens, for seed 2** (§10.1). The answer was 22 incidents
   where the same seed answered 11 on the shipped window; 16,384 was right for every seed,
   and 8,192 for seeds 1 and 3. §10's rule: R3.2 as designed ends there and goes back to
   the Conductor, with Q1's (c) or a higher floor. The run folder says more than the
   rule does: the 16-word notes kept every fact the answer needs (Northgate's "five
   equipment incidents under contractor code CT-118", Riverside's "6 equipment incidents
   under CT-118", the registry's mapping), and the parse was right; reasoning, shown one
   sample record per field (as at 16,384), sent five turns without a code block, then
   summed every CT-118 record, the parse's breakdown records (3 + 2 at Northgate, 4 + 2
   at Riverside) included. On the shipped window it saw the whole parse. So the failure
   sits where R3.1's sample view meets a schema that keeps sub-counts as records, at
   least as much as in the shorter notes; one seed in three cannot separate the two.
   Nothing was changed for it. **A second run, six seeds** (§10.1): 17 of 18 right, the
   null again at 8,192 (seed 3, 16 incidents). Across both runs' 27 cases, every wrong
   answer is a parse that kept sub-counts as records beside their totals, seen by
   reasoning as samples; the 16-word notes kept the facts in all nine runs at 8,192, and
   the shipped window was right 9 of 9. The evidence points at R3.1's sample view, not at
   the notes losing facts or at noise the shipped window shares.
10. **E7's harness recorded the reasoning loop's live message list**, so a first turn
    that went on to a second no longer had two messages when the test looked, and 8 of
    the 9 cases raised `IndexError` before printing their line. The nine runs had
    completed; their lines are recomputed from the run folders the job uploaded, with
    the test module's own helpers, and the one case that printed its own line (16,384,
    seed 1) matches its recomputed row exactly. The harness now records a copy of each
    request's messages (`tests/test_experiments.py`, `recording`); an offline dry run
    with a fake that takes two reasoning turns failed before the fix and passes after.

## 3 · Order of work

Each step is test-first, and CI is green after every step.

| Step | Item | Lands |
| --- | --- | --- |
| 1 | R3.1 | `run_codeact(on_first_turn_too_long=)` (§4.1); `Splits.model` and `Splits.over_line` (§6.3); `reason(splits=)`, the window-aware parse view and the refusal retry, with no budget yet (§4.2); `pipeline.py` passes `splits` to `reason`; E1, E2 |
| 2 | R3.2 | `notes.py`: `NotesTooLong`, `count_notes`, `Budget` (§5) |
| 3 | R3.2 | `splitting.py`: `read_in_parts(notes=)`, `measure`, `check_notes` (§6). Nothing passes notes yet, so every run is unchanged. |
| 4 | R3.2 | `prompts/relevance/v2.yaml`, `configs/default.yaml`; `relevance.py`: `max_words`, `note_texts`, `takes_word_budget`, `RelevantContext.words`, `surface_relevance(budget=, reread=, final_check=)` (§7.1, §7.2); `reasoning.check_first_turn` (§7.4); `pipeline.py` builds the run's `Budget` and passes it to relevance with the final check, and writes `max_words` (§7.5); `test_config.py`'s pins and R2's E5 test (§9.2); E5, E6. A `NotesTooLong` now arises only inside relevance, which handles it. |
| 5 | R3.2 | `budget=` on `propose_schema` (every attempt measured), `parse_documents` (notes and `measure`) and `reason` (the notes' hand-over) (§7.3, §4.2 step 3); `pipeline.py` passes the budget to them, with its stage loop and `_recorded_stage(run=)` (§7.5), in the same commit, since a stage that hands notes over needs the loop that catches them; the stopped-run test (§9.2); E3, E4 |
| 6 | R3.2 | the docstrings and README of §7.6 |
| 7 | E7 | `tests/test_experiments.py`'s E7, the `experiment` job's timeout (§10) |
| 8 | release | `version = "0.3.0"` in `pyproject.toml`, and `uv lock` (§11) |
| — | proof | one dispatch of `ci.yml` with `live`: every job green. One dispatch with `experiment` for E7, which runs R2's E6 beside it. |

## 4 · R3.1 · Reasoning shows the whole parse only when its first turn fits

**Why.** Reasoning's prompt carries the whole parse whenever it is under
`REASONING_PARSE_MAX_TOKS` (16,000 tokens), whatever the window. On an 8,192-token window
the parse alone fills it, so reasoning is refused after every other stage has run and
billed (§1.1, 60 memos). The agent already handles a sample view: it is what a huge parse
gets.

### 4.1 `runtime/codeact.py`: a first turn refused as too long

```python
def run_codeact(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    variables: dict[str, Any] | None = None,
    tools: dict[str, Callable[..., Any]] | None = None,
    max_turns: int | None = None,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    additional_authorized_imports: list[str] | None = None,
    run: StageRun | None = None,
    on_first_turn_too_long: Callable[[ContextWindowExceededError], str] | None = None,
    **llm_kwargs: Any,
) -> CodeActResult: ...
```

- **`on_first_turn_too_long`** is called when the provider refuses the first turn with
  `litellm.ContextWindowExceededError`, with that refusal. It returns the system prompt to
  send the first turn with instead, and the loop replaces `messages[0]` with it and asks
  again, for this and every later turn. It may be called again if that request is
  refused too; it ends the retries by raising. Without it, the refusal propagates, as
  today.
- **Only the first turn.** A refusal of turn 2 or later propagates as today: the
  conversation outgrowing the window is out of R3's scope.
- **The `stop` retry is kept.** `ContextWindowExceededError` is a `BadRequestError`, so it
  is caught first. The first turn is asked in a loop that ends when a request is accepted;
  a refusal of `stop` drops `stop` and asks again (today's behaviour), and a refusal as too
  long calls the hook. Either can follow the other.
- **Logs.** None here: the caller says what it changed.

### 4.2 `stages/reasoning.py`

```python
def reason(
    *,
    task: str,
    schema_code: str,
    parsed: BaseModel | dict[str, Any],
    model: str,
    prompt_version: str,
    relevance_snippets: Sequence[Snippet] | None = None,
    source_docs: dict[str, list[int]] | None = None,
    max_turns: int | None = None,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    additional_authorized_imports: list[str] | None = None,
    run: StageRun | None = None,
    splits: Splits | None = None,
    budget: Budget | None = None,
    **llm_kwargs: Any,
) -> CodeActResult: ...


def _sample_view(parse_dict: Any) -> str: ...


def _system_prompt(
    *,
    task: str,
    schema_code: str,
    relevance_snippets: Sequence[Snippet] | None,
    prompt_version: str,
    parse_json: str,
    samples_block: str,
    parse_block: str,
) -> str: ...
```

- **`_sample_view`** is the second half of today's `_render_parse_for_codeact`
  (`reasoning.py:115–126`): the "only a SAMPLE" note and one record per field. The flood
  guard keeps calling it, unchanged.
- **`_system_prompt`** is today's `load_prompt("reasoning", …)` call (`:174–184`), moved
  into a helper so that `reason` can render twice and `check_first_turn` (§7.4) renders the
  same prompt. The variables a prompt version may use do not change.
- **`splits=None`** builds `Splits([], model=model)`. The pipeline passes the run's.
- **`budget`** switches on the notes' hand-over (steps 3 and 4 below); without it, `reason`
  is R3.1 alone, and a refused samples turn propagates as today.

**The choice, in order** (`user` is the user message, `Input:\n<task>\n{task}\n</task>`):

1. `parse_block = _render_parse_for_codeact(parse_dict, parse_json)`, as today: the whole
   parse, or the sample view for a parse over `REASONING_PARSE_MAX_TOKS`.
2. **Measured** (a known window): if the whole parse is shown and
   `splits.over_line(system_prompt, user)` returns an estimate, render again with
   `_sample_view(parse_dict)`, and log INFO on `r3con.reasoning`, for example:

   ```text
   the whole parse (about 7,412 tokens) would put reasoning's first turn at about 13,596 tokens, over the 6,963-token line; showing one sample record per field
   ```

3. **The notes** (R3.2, with a `budget`): `splits.check_notes(call="reasoning",
   request=[system_prompt, user], notes=note_texts(relevance_snippets or ()))` hands the
   notes over when the samples turn is still over the line (§6.3). Nothing has been sent.
4. **Sent**, through `run_codeact` with `on_first_turn_too_long`. The hook, called with a
   refusal of the first turn:
   - while the whole parse is shown, switches to the sample view, logs WARNING on
     `r3con.reasoning` and returns the new prompt:

     ```text
     reasoning's first turn was refused as too long with the whole parse (about 7,412 tokens); sending it again with one sample record per field
     ```

   - once samples are shown, with a `budget`, calls `splits.check_notes(…,
     refusal=refusal)`, which raises `NotesTooLong` from the refusal when the turn carries
     notes; in every other case the hook re-raises the refusal.

So the whole-parse turn is sent at most once per call of `reason` (E2's null). A refusal
with a known window goes the same way: the estimate said it fits, the provider disagreed.
After the notes are read again, the pipeline calls `reason` again from the start, so the
whole parse gets a new chance with the shorter notes (§12, the cost).

**What does not change.** `REASONING_PARSE_MAX_TOKS` stays the flood guard for every
window. The full parse is bound as `parse` either way, so the agent loses reach, not data.
No prompt changes; the sample view and its note are today's text. On the default model the
first turn is under the line in UTF-8 bytes, so nothing is counted and the prompt is
0.2.0's (E6).

## 5 · `notes.py` · the notes' budget

### 5.1 Shape

```python
MIN_NOTE_WORDS = 10


class NotesTooLong(Exception):
    """A request that does not fit the model's window while the notes it carries are
    the bigger part of it: bigger than its document part, or all it carries beside its
    own prompt (the schema call, reasoning's first turn). Not a
    ``ContextWindowExceededError``, so nothing that cuts a document takes it for one."""

    model: str
    call: str
    cause: str
    estimate: int
    notes: list[str]
    notes_tokens: int
    room: int | None
    refusal: ContextWindowExceededError | None

    def __init__(
        self,
        message: str,
        *,
        model: str,
        call: str,
        cause: str,
        estimate: int,
        notes: Sequence[str],
        notes_tokens: int,
        room: int | None = None,
        refusal: ContextWindowExceededError | None = None,
    ) -> None: ...


def count_notes(notes: Sequence[str]) -> int: ...


class Budget:
    """How many words each relevance note may take in one run, and the record of every
    time the notes were read again shorter (``notes.json``)."""

    splits: Splits
    words: int | None

    def __init__(
        self,
        splits: Splits,
        *,
        can_shorten: bool = True,
        task_logger: TaskLogger | None = None,
    ) -> None: ...

    def shorten(
        self,
        trouble: NotesTooLong,
        *,
        round_idx: int,
        discarded: int = 0,
        also: Sequence[NotesTooLong] = (),
    ) -> int: ...
```

- **`NotesTooLong`'s fields.** `call` is the request that did not fit: a kind
  (`relevance-r2-d3`, `parse-d0c1`) or a stage (`schema`, `reasoning`). `cause` is
  `"estimate"` or `"refusal"`. `estimate` is the request's tokens, counted as R2 counts
  (§6). `notes` is the notes it carries, one text per document or part, without the
  empty ones (`relevance.note_texts`, §7.2). `notes_tokens` is `count_notes(notes)`.
  `room` is what the notes may take for this request to fit,
  `line - (estimate - notes_tokens)`, with a known window and an estimate; it is `None`
  after a refusal, even with a known window, because the provider counted more than the
  estimate. `refusal` is the provider's error, and `model` the run's model string, for the
  stop's error. The constructor passes `message` to `Exception`, so `str(trouble)` is the
  message, which the stop reuses. A `NotesTooLong` built after a refusal is raised `from`
  it.
- **Why not litellm's class** (§2.4). R2's `read_in_parts` cuts a document on
  `except ContextWindowExceededError`; a notes signal of that class, raised anywhere
  inside such a block now or after a later edit, would be taken for a refusal of the
  document. As r3con's own class it passes through every such `except` untouched.
- **Its message.** By estimate, R2's measured message, from the same helper:
  `r3con estimated parse-d0 at 7,158 tokens, over the 6,963-token line (the 8,192-token
  input window litellm's model map gives hosted_vllm/window-8192, less 15%); it was not
  sent.` By refusal: `schema was refused as too long, and the notes it carries (about
  5,366 tokens) are the bigger part of it.`
- **Its note**, added when it is built, for a direct caller of a stage who passed a
  `Budget`, the only place one surfaces: `r3con: the notes parse-d0 carries (about 5,366
  tokens) are the bigger part of a request that does not fit; hand this to the budget's
  shorten and read their round again, as r3con.run does.`
- **`count_notes(notes)`** is `sum(count_tokens(note + "\n\n") for note in notes)`. Every
  rendering puts a paragraph break after a note, and a note's last token merges with it
  (`.\n\n` is one cl100k token, `word\n\n` two). Counting each note with its break keeps
  the room of a request stable when its notes are cut (*run*: counted bare, 250 memos read
  round 1 twice where once was enough).
- **`Budget`** is a class because it holds state across calls: W and the record. It is
  used from the calling thread only (the stages raise from their workers; the exception
  reaches the thread that called the stage), so it takes no lock. `splits` is the run's
  `Splits`, whose window and line it records and whose `check_notes` the document-less
  stages call through it.
- **`can_shorten`** is whether the run's relevance prompt can ask for a length
  (`relevance.takes_word_budget`, §7.2). With `False`, every `shorten` stops (§2.2 item 4).
- **Exported?** No, like `Splits`: a direct caller imports `Budget` and `NotesTooLong`
  from `r3con.notes`. `r3con/__init__.py` does not change.

### 5.2 Choosing W

`shorten(trouble, *, round_idx, discarded, also)` sets `self.words` and returns it, or
stops the run (§5.3). `trouble` is the request that did not fit; `round_idx` is the round
that wrote its notes, the one to read again; `discarded` is how many of the stage's
accepted calls are thrown away; `also` holds requests the same notes, or the later rounds'
notes read under the same W, will ride in later, measured now (§7.2).

1. **The notes' length.** `mean` is the mean word count of `trouble.notes`, an integer,
   where a word is what `str.split()` separates.
2. **Where halving starts.** `start` is `self.words` when a budget is already set, else
   `mean`. So a model that overshot its W is asked for half of W, not half of what it
   wrote (the spec's "a note that overshoots its W halves W again").
3. **The levels.** `start // 2`, `start // 4`, … each raised to `MIN_NOTE_WORDS`, ending at
   the first level that is `MIN_NOTE_WORDS`, and keeping only levels below `mean` (a
   level the notes already obey cannot shorten them). From 33: 16, 10. From 28: 14, 10.
   From 12: 10. From 10: none.
4. **By estimate** (every request in `[trouble, *also]` with a `room`): keep the levels at
   which, for each such request, `_tokens_at(request.notes, W) <= request.room`, where
   `_tokens_at(notes, W)` is `sum(count_tokens(" ".join(note.split()[:W]) + "\n\n") for
   note in notes)`: each note cut at its W-th word, a stand-in for the note the model will
   write under W. The first level kept is W. So W is the fewest halvings that fit, found
   before anything is read again.
5. **By refusal** (no room): the first level is W, one halving per refusal.
6. **None left**, or `can_shorten` is false: the run stops (§5.3). Otherwise the event is
   recorded with action `"read again"`, W is logged, and W is returned.

**How it bounds itself.** Every W is below the last (levels are below `start`, and
`start` is the last W), and none is below 10. So W takes at most ⌈log₂(mean / 10)⌉ values
in a run, `mean` being the first notes': 2 for 33-word notes, 5 for 200-word notes. Each
value reads again the round that did not fit and the rounds after it.

### 5.3 The stop

The event is recorded with action `"stop"` and `"words": null`, and the run raises:

- **after a refusal**, the provider's `ContextWindowExceededError` (`trouble.refusal`);
- **before sending**, r3con's own, `ContextWindowExceededError(message=str(trouble),
  model=trouble.model, llm_provider="r3con")`.

Either is raised `from None`, so the traceback shows one error, not the `NotesTooLong` it
was caught as. This is the only way a notes trouble leaves r3con: the two loops that
catch `NotesTooLong` hand every one to `shorten`, which reads the notes again or raises
this, so `run_pipeline`, `r3con.run` and the CLI (which prints `type(e).__name__` and the
notes) see a `ContextWindowExceededError` and never a `NotesTooLong` (§2.4). The error names the request that was about to be sent (`trouble`). The note,
added with `add_note`, names the request that binds, `binding = min([trouble, *also],
key=room)`, and `_recorded_stage` adds the run-folder note after it. The first row applies
when `can_shorten` is false, the third when the binding room is 0 or less, the second
otherwise with a known window, the fourth with an unknown one. The numbers are the runs'
(the fourth row's are an example):

| Stop | Note |
| --- | --- |
| the relevance prompt asks for no length | `r3con: relevance-r2-d0 carries the notes of 39 documents (about 7,410 tokens), the bigger part of a request that does not fit, and the relevance prompt this run pins cannot ask for shorter notes (the shipped v2 can). The relevant context has outgrown the model's window.` |
| by estimate, at the floor | `r3con: even at 10 words each, the notes of 400 documents would take about 6,733 tokens, over the 2,571 that reasoning leaves them, so reading them shorter cannot help. The relevant context has outgrown the model's window.` |
| by estimate, no room at all (`room <= 0`) | `r3con: what relevance-r2-d0 sends beside its notes fills the 6,963-token line by itself, so reading them shorter cannot help. The relevant context has outgrown the model's window.` |
| by refusal, at the floor | `r3con: reasoning was refused as too long, and the notes of 120 documents it carries are already about 9 words each, with 10 the fewest a note is asked for, so reading them shorter cannot help. The relevant context has outgrown the model's window.` |

The first row names `trouble`, the request that did not fit, not the binding one
(§2.5 item 1). A fifth case has a note of its own: by estimate, with W already 10 and
notes still over it, `r3con: reasoning is estimated over the line, and the notes of 20
documents it carries are already about 14 words each, with 10 the fewest a note is asked
for, so reading them shorter cannot help. …` (§2.5 item 2).

No task ID, and the closing sentence is R2's, so a caller that matched it keeps matching.

### 5.4 `notes.json` and the logs

Written to the run folder's root with `task_logger.write_json("notes", …)` on every event.
A run that never reads notes again or stops for them has none. E3, *run*:

```json
{
  "model": "hosted_vllm/window-8192",
  "max_input_tokens": 8192,
  "margin_percent": 15,
  "line": 6963,
  "events": [
    {"call": "reasoning", "cause": "estimate", "estimate": 8330, "notes": 5366,
     "room": 3999, "sized_for": "reasoning", "discarded": 0, "action": "read again",
     "round": 2, "words": 16, "error": null}
  ]
}
```

- The window is recorded once, from the run's `Splits`, like `splits.json`.
- An event: the request that did not fit (`call`), why (`cause`), its estimate and its
  notes' tokens, the room W was sized for and the request it belongs to (`sized_for`:
  `call` itself, or reasoning's first turn, §7.2; `room` is `null` after a refusal), the
  calls discarded, the action (`"read again"` or `"stop"`), the round read again (at a
  stop, the round it would have read again) and W (`null` at a stop), and the provider's
  message after a refusal.

**Logs.** A read again logs WARNING on `r3con.notes`; the CLI shows it, and a library user
gets it on stderr, as with R2's splits:

```text
reasoning: the notes of 120 documents come to about 5,366 tokens, and the request with them to about 8,330, over the 6,963-token line; reading round 2 again with each note under 16 words
reasoning was refused as too long; the notes of 120 documents it carries (about 5,366 tokens) are the bigger part, so round 2 is read again with each note under 16 words
```

## 6 · `splitting.py` · the rule: the bigger part is halved

### 6.1 `read_in_parts(notes=)`

```python
class Splits:
    model: str
    max_input_tokens: int | None
    margin_percent: int
    line: int | None

    def read_in_parts(
        self,
        doc: int,
        *,
        call: str,
        rest: str,
        send: Callable[[str, str], T],
        notes: Sequence[str] = (),
    ) -> list[T]: ...
```

`notes` are the notes `rest` carries, one text per document or part, as rendered inside it
(`note_texts`, §7.2). A stage passes them only when it is given a `Budget` (§7). With
none, the loop is R2's exactly, which is why R2's unit tests stand (§2.2 item 5). `model` becomes public (it was `_model`), for `Budget`'s record. The
notes' tokens are counted once per call, with `count_notes`, and only where R2 already
counts.

**With a known window**, the measured step (R2's step 1) becomes:

1. If the rest alone is over the line: with notes, raise `NotesTooLong` (cause
   `"estimate"`, estimate the rest plus part 0, as R2's note does); more parts can never
   help. Without notes, R2's measured stop, unchanged.
2. Otherwise find the first part whose request is over the line (R2's byte shortcut first).
   If the notes' tokens are greater than the part's, raise `NotesTooLong` for that part's
   kind; nothing is cut. Otherwise R2: the cut floor, or halve and measure again.

**On a refusal of part `k`**, before R2's rules: if the notes' tokens are greater than the
part's, raise `NotesTooLong` (cause `"refusal"`, `room=None`) from the refusal; nothing is
cut. Otherwise R2's rules, unchanged: with an unknown window, a part shorter than the rest
stops the run; a one-character part stops it; else halve.

A tie goes to the part: cutting a document is one more call per stage, and reading notes
again is a round. The `NotesTooLong` is raised from the worker thread and reaches the
stage's caller through `parallel_map`. It is not a `ContextWindowExceededError`, so the
loop's own `except ContextWindowExceededError` around `send` never takes one raised
inside `send` for a refusal: it passes through, like any other error, and nothing is cut.

### 6.2 `measure`: a stage measures before it sends

```python
class Splits:
    def measure(
        self,
        docs: Iterable[int],
        *,
        call: str,
        rests: Callable[[int], tuple[str, Sequence[str]]],
    ) -> None: ...
```

A stage given a `Budget` calls it with a known window, before it fans out; without a
budget it does not, and R2's estimate splits stay in the workers. For each document, `rests(doc)` gives its
`(rest, notes)`, and the measured step of §6.1 runs for it, cutting the document by
estimate as `read_in_parts` would. The `NotesTooLong` of every document whose notes must
shrink is collected, and the one with the least room is raised once all are measured, so
W is sized for the tightest request and nothing has been sent. R2's stops raise at once.
With an unknown window it returns at once.

- **Why a pre-pass.** Without it, a worker whose document fits sends while another
  discovers that the notes do not fit, and the first one's call is wasted; and W would be
  sized for whichever document raised first.
- **R2's estimate splits move here** from the workers. The cuts and events are the same;
  `splits.json` lists them in document order instead of the order the workers reached
  them.
- **The cost.** `read_in_parts` measures again (its counts are per call), so near the line
  each request is counted twice. Far from it the byte shortcut decides both, and nothing
  is counted.
- `rests` is a callable so a stage renders each document's rest when it is measured:
  holding all of round 2's rests at once would hold N prompts of N notes each.

### 6.3 Requests without a document: `check_notes`, `over_line`

```python
class Splits:
    def over_line(self, *texts: str) -> int | None: ...

    def check_notes(
        self,
        *,
        call: str,
        request: Sequence[str],
        notes: Sequence[str],
        refusal: ContextWindowExceededError | None = None,
    ) -> None: ...
```

- **`over_line(*texts)`** returns the texts' tokens, `sum(count_tokens(t) for t in
  texts)`, when the window is known and they are over the line; otherwise `None`. Texts
  no larger than the line in UTF-8 bytes are not counted (R2's ruling 3).
- **`check_notes`** is for a request that carries notes and no document: the schema call
  and reasoning's first turn, called only by a stage given a `Budget`. `request` is its
  texts (system prompt, user message), counted as their sum, as R2 counts a rest and a
  part.
  - Without `refusal`: raises `NotesTooLong` (cause `"estimate"`) when the request is over
    the line (`over_line`) and carries notes; returns otherwise, and always with an
    unknown window.
  - With `refusal`: raises `NotesTooLong` (cause `"refusal"`) from it when the request
    carries notes; returns otherwise, and the caller re-raises the refusal.
  - With no notes it never raises: the request is sent, and refused, as today. That is
    the spec's "a stage's own prompt over the window by itself stops the run, as today".
- The module docstring gains a paragraph: when a request's notes are bigger than its
  document part, or it has no document, the request is handed to `r3con.notes` by
  raising `NotesTooLong` rather than cut. The unmapped-window INFO line becomes `litellm's
  model map gives no input window for %s; r3con fits a request to it only when the model
  refuses one` (R2's test matches "gives no input window", which stays).

## 7 · The stages and the pipeline

### 7.1 `prompts/relevance/v2.yaml`, `configs/default.yaml`

v2 is v1 with one change, in its Output paragraph (line 19), and v1 is not touched:

```diff
-  Reply with the task-conditioned summary of this document only — no preamble, no restating the task. If this document contributes nothing relevant to the task, it is fine to say so briefly (or reply with nothing).
+  Reply with the task-conditioned summary of this document only — no preamble, no restating the task. If this document contributes nothing relevant to the task, it is fine to say so briefly (or reply with nothing).{% if max_words %} Keep it under {{ max_words }} words: the collection is large, and every document's summary has to fit beside all the others.{% endif %}
```

Without `max_words`, or with `max_words=None`, v2 renders to v1's text byte for byte, with
or without other notes (*run*). Under a budget the sentence is 112 characters, 25 tokens,
and every request of a round read under one carries it, which the room accounts for
because it is in the rest. `configs/default.yaml` pins `relevance: v2`, so the default
label reads `prompts=(rel=v2,schema=v1,parse=v1,reason=v2)`, shortened or not. R1.12's
check finds `v2.yaml` in the package.

### 7.2 `stages/relevance.py`

```python
@dataclass
class RelevantContext:
    snippets: list[Snippet]
    rounds: list[list[Snippet]]
    words: list[int | None] = field(default_factory=list)


def note_texts(snippets: Sequence[Snippet]) -> list[str]: ...


def takes_word_budget(prompt_version: str) -> bool: ...


def relevance_snippet(
    *,
    task: str,
    document: str,
    other_snippets: list[str],
    model: str,
    prompt_version: str,
    run: StageRun | None = None,
    kind: str = "relevance",
    max_words: int | None = None,
    **llm_kwargs: Any,
) -> str: ...


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
    budget: Budget | None = None,
    reread: RelevantContext | None = None,
    final_check: Callable[[list[Snippet]], None] | None = None,
    **llm_kwargs: Any,
) -> RelevantContext: ...
```

- **`RelevantContext.words`** is the word budget each round's notes were read under,
  aligned with `rounds` (`None` for a round read without one; empty when no round was
  read). A default, so positional construction still works.
- **`note_texts(snippets)`** is the notes a rendered block carries, in order: each string
  snippet, and each part note of a list snippet, stripped, without the empty ones. It is
  the `notes` every site passes (§6). For a round-r call it is applied to the other
  documents' joined notes, the ones `_render_other_states` renders.
- **`takes_word_budget(prompt_version)`** is whether the prompt at that version renders
  differently with `max_words=10` than without (both with `task=""` and no other notes):
  true for v2, false for v1 and for an overlay that does not use `max_words`.
- **`_system_prompt(task, other_snippets, prompt_version, max_words=None)`** passes
  `max_words` to `load_prompt`; `relevance_snippet` passes its own through.
- **`budget=None`** is 0.2.0's path: no notes are passed to the splitter, nothing is
  measured before the fan-out, and no round is read again; R2's rules alone apply. With a
  `budget`, the rest of this section applies, and the round loop catches every
  `NotesTooLong` its rounds and its `final_check` raise, so none leaves
  `surface_relevance`. `reread` and `final_check` belong to a run's budget: passing
  either without one raises `ValueError` before any request.

**Reading one round** (today's inner `run_round`, `relevance.py:181–226`), round `r`
under `W = budget.words`, with a budget:

1. Its call is `relevance-r{r}`, or `relevance-r{r}-w{W}` under a budget. The round's log
   line gains `, each note under {W} words`.
2. Document `i`'s rest is `_system_prompt(task, others_i, prompt_version, W)` and its
   notes `note_texts(others_i)`, where `others_i` is today's (the previous round's notes
   of every other document, joined).
3. `splits.measure(range(n), call=…, rests=…)` (§6.2), then the fan-out, each worker
   calling `read_in_parts(i, call=…, rest=…, notes=…, send=…)` with
   `relevance_snippet(…, max_words=W)`.

**The rounds.** `read` is the list of rounds read so far (`reread.rounds[:-1]` when
`reread` is given, else empty), and round `r` is read by this loop:

- Read round `r`. On success, `read[r - 1:] = [notes]` and `words[r - 1:] = [W]`, which
  drops any later round, since it must be read again.
- On `NotesTooLong` (it is about round `r - 1`'s notes): with `final_check`, also run
  `final_check(read[r - 2])`, catching its `NotesTooLong` into `also`, because the later
  rounds are read under the same W and these notes stand in for the final ones. Then
  `budget.shorten(trouble, round_idx=r - 1, discarded=…, also=also)`, read round `r - 1`
  again by this same loop (it may read earlier rounds again in turn), and read round `r`
  again. `discarded` is the calls `run` gained since this attempt at round `r` began
  (0 without a `run`).
- Round 1 carries no notes, so it never raises `NotesTooLong`, and the recursion ends
  there.

Rounds `len(read) + 1` to `rounds` are read in order. Then, with `final_check`:
`final_check(read[-1])`; on `NotesTooLong`, `budget.shorten(trouble, round_idx=rounds)`,
read the last round again by the loop, and check again. `reread` therefore reads the last
round again under the budget's current W, keeping the earlier rounds, which is what the
pipeline asks for when a later stage's request does not fit (§7.5).

`rounds < 1` or no documents returns the empty context before any of this, as today.

### 7.3 `stages/structuring/schema.py` and `parsing.py`

```python
def propose_schema(
    *,
    task: str,
    relevance_snippets: Sequence[Snippet] | None = None,
    model: str,
    prompt_version: str,
    max_attempts: int | None = None,
    run: StageRun | None = None,
    budget: Budget | None = None,
    **llm_kwargs: Any,
) -> ProposalResult: ...


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
    budget: Budget | None = None,
    **llm_kwargs: Any,
) -> ParseResult: ...
```

- **The schema call**, with a `budget`: before every attempt, the first and each retry
  (whose user message carries the previous schema and the validator's error),
  `budget.splits.check_notes(call="schema", request=[system_prompt, user_prompt],
  notes=note_texts(relevance_snippets or ()))`; around the call,
  `except ContextWindowExceededError as refusal:` runs `check_notes(…, refusal=refusal)`
  and re-raises the refusal. Without a budget, the schema call is 0.2.0's: nothing is
  measured, and a refusal propagates after litellm's resends.
- **Parsing**, with a `budget`: `notes = note_texts(relevance_snippets or ())`; before the
  fan-out, `splits.measure(range(len(documents)), call="parse", rests=lambda _: (rest,
  notes))`; each worker passes `notes=notes` to `read_in_parts`. Without one, neither,
  and parsing is R2's. The run passes its `Splits` and its `Budget`, whose `splits` is the
  same object. `parse_one_document` does not change; a validation retry's added text is
  still not in the estimate, as in R2.

### 7.4 `stages/reasoning.py`: `check_first_turn`

```python
def check_first_turn(
    *,
    task: str,
    relevance_snippets: Sequence[Snippet],
    prompt_version: str,
    budget: Budget,
) -> None: ...
```

Reasoning's first turn as far as it is known before the schema exists:
`_system_prompt(…, schema_code="", parse_json="", samples_block="", parse_block="")` and
the user message, measured with `budget.splits.check_notes(call="reasoning", …)`. It raises
`NotesTooLong` when even that is over the line; the real turn adds the schema and the
records, so it is a lower bound, and it never shortens notes that would fit. It is
`surface_relevance`'s `final_check` in a run (§7.5). `reason` itself checks the real
first turn (§4.2).

### 7.5 `pipeline.py`

`run_pipeline`'s signature does not change. After the `Splits`:

```python
budget = Budget(
    splits,
    can_shorten=takes_word_budget(config.prompts["relevance"]),
    task_logger=task_logger,
)
```

- **Relevance.** One `functools.partial` of `surface_relevance` holds everything a read of
  the relevant context needs: `splits`, `budget`, the workers, `llm_kwargs`, and
  `final_check=lambda notes: reasoning.check_first_turn(task=task,
  relevance_snippets=notes, prompt_version=config.prompts["reasoning"], budget=budget)`. Stage 1 calls it inside `_recorded_stage("relevance")`, as today, and
  keeps that stage's record for later reads.
- **`relevance/result.json`** is built by a private `_relevance_result(relevance, rounds,
  n_docs)`: today's dict, each round's entry gaining `"max_words": W` when its `words`
  entry is not `None`.
- **Every later stage is sent through one loop**, a closure in `run_pipeline`:

  ```python
  def fitting(send: Callable[[list[Snippet]], T], run: StageRun | None) -> T: ...
  ```

  It calls `send(relevance.snippets)`. On `NotesTooLong`, it calls
  `budget.shorten(trouble, round_idx=rounds, discarded=…)`, reads the last round again
  with `surface_relevance(…, reread=relevance)` inside
  `_recorded_stage("relevance", run=<the relevance record's run>)`, rebinds `relevance`,
  and calls `send` again from the start. `discarded` is what `run` gained during the
  failed attempt. The schema, parsing and reasoning stages each run inside their own
  `_recorded_stage`, as today, with `send` a lambda over the notes passing
  `budget=budget` to every stage, and `splits=splits` to `parse_documents` and `reason`.
  `fitting` catches `NotesTooLong` only; any other exception, a stop's
  `ContextWindowExceededError` included, passes through it to `_recorded_stage`.
- **`Answer.relevant_context`** is `[join_parts(s) for s in relevance.snippets]` of the
  last read: the notes reasoning saw.
- **`_recorded_stage(…, run: StageRun | None = None)`.** With `run`, the stage's earlier
  calls are kept and this block's are added, so a round read again lands in
  `relevance/calls.json` beside the round it replaces, and `result.json` is rewritten.
  A failure adds the run-folder note only if the error does not carry it yet: a stop
  inside a read again is recorded by `relevance/` and by the stage that asked for it, each
  with `error.txt`, and carries the note once.
- The module docstring gains one sentence: notes that fill the window are read again
  shorter (`r3con.notes`).

### 7.6 Docstrings, README, `runs.py`

- `runs.py`'s layout: `notes.json` ("only when notes were read again shorter or the run
  stopped for them: the model's window and every event"); in `relevance/`, a round read
  under a budget has `max_words` and its calls are tagged `relevance-r{round}-w{W}-d{doc}`;
  a round read again that fails leaves `error.txt` beside the earlier `result.json`.
- `r3con.py`, `run`'s Raises (`:264–266`): `litellm.ContextWindowExceededError` comes when
  the prompt sent beside a document leaves it no room, or when the notes, even at 10
  words each, do not fit the model's window. `run_pipeline`'s docstring says the same.
- **The stages' Raises** (the ruling, §2.4). `propose_schema`, `parse_documents` and
  `reason`: `r3con.notes.NotesTooLong`, only when given a `budget`, when a request's notes
  are the bigger part of a request that does not fit; the caller hands it to
  `budget.shorten`, reads the round that wrote the notes again under `budget.words`, and
  calls the stage again, as `r3con.run` does. `check_first_turn`: `NotesTooLong`, its
  purpose. `surface_relevance`: no `NotesTooLong` (its round loop handles its own);
  `litellm.ContextWindowExceededError` from `Budget.shorten` when the notes cannot be read
  short enough, and `ValueError` for `reread` or `final_check` without a budget.
- `README.md`'s tree gains `├── notes.json   <- only when notes were read again shorter`.
- `splitting.py`'s module docstring (§6.3); `relevance.py`'s, one sentence on reading a
  round again under a word budget.

## 8 · What a user sees

**Now works**

- **A collection whose notes outgrow the model's window answers.** On a model litellm
  maps, or whose window the user registers, the notes are measured before anything is
  sent; on any other, on the provider's refusal. `relevant_context` still has one entry per
  document, every heading and record keeps its number, and the answer comes from shorter
  notes (§1.1).
- **Reasoning no longer fills the window with the parse.** The whole parse is shown when
  it fits, the samples when it does not (R3.1).
- **The user is told.** One WARNING per read again, and `notes.json` in the run folder.

**Fails differently**

- **When even 10-word notes cannot fit, the run stops** with
  `litellm.ContextWindowExceededError` and a note that says so (§5.3). With a known window
  that happens before anything is read again. The run folder keeps `notes.json`, the
  stage's `calls.json` and `error.txt`.
- **A config that pins relevance v1** stops where R2 stopped, with a note naming v2.
- **A direct caller of a stage** (`surface_relevance`, `parse_documents`, `propose_schema`,
  `reason`) that passes no `Budget` gets 0.2.0's behaviour, R3.1 in `reason` aside. One
  that passes a `Budget` gets R3.2: `surface_relevance` reads its own rounds again, and
  the other three raise `r3con.notes.NotesTooLong` (not a `ContextWindowExceededError`)
  for the caller to shorten and read again, as their Raises say.

**Changes in every run**

- The default label reads `rel=v2`. A run that shortens nothing sends v1's relevance text,
  and 0.2.0's requests, byte for byte (E6).
- Each stage measures its requests before sending. On the default model nothing is
  counted (the byte shortcut).

**Changes some runs**

- **A first reasoning turn estimated between 85% and 100% of a mapped window** was sent
  with the whole parse and now shows samples (§1.1, 280 memos at 32,768 tokens).
- **A document over the line only because its notes are bigger** is no longer cut in a run;
  its notes are read again shorter (Q2 (a)). A stage called without a budget still cuts
  it, as R2 does.
- **`relevance/result.json`** gains `max_words` on a round read under a budget, and
  `RelevantContext` gains `words`.

**Breaks**

- Nothing is removed from the API. R2's behaviours change where the spec says (§9.2).

## 9 · Tests, by file

They use R1's and R2's machinery: `FakeLLM` through `completion=`, `llm`,
`answering_llm`, `window`, `grown_registry`, `_isolated`, pytest-socket. "New" means the
test fails at `f4dae00`; "guard" means it passes before and after.

### 9.1 `conftest.py` additions

- **`FakeLLM.refuses_over(chars: int | None = None, *, tokens: int | None = None)`.**
  `tokens=` refuses a request whose cl100k tokens, its messages' contents and its response
  format's schema as JSON, exceed the limit: a provider that counts tokens, for E2 and E4.
  The message names the limit and the count, as `too_long` does. `refuses_over(chars)` is
  unchanged.
- **`within_budget(reply) -> Callable[[Request], str]`** wraps a relevance reply (a
  string or a callable): when the system prompt says `Keep it under (\d+) words`, it
  returns the reply's first W words. A model that obeys; the spec's "the fake honours a
  budget".
- **`quiet_sites(n) -> list[str]`**, n site memos of about 230 characters, each logging 0
  equipment incidents under CT-204 (even k) or CT-311 (odd k), with a unique site name.
  Added to the five memos, the answer cannot change. E1 to E5 and E7 use it.
- **`memo_note(request) -> str`** (in `test_pipeline.py`): a relevance reply of about 200
  characters that names the document's first line, so a run's notes are memos-shaped.

### 9.2 The R2 tests that change, and why

| Test | Change | Why |
| --- | --- | --- |
| `test_config.py`: `DEFAULT_PROMPTS`, `test_the_default_config_loads_with_its_shipped_values` | `prompts=(rel=v2,schema=v1,parse=v1,reason=v2)`; the shipped prompts are `{**PROMPTS, "relevance": "v2", "reasoning": "v2"}` | the default config pins relevance v2 (§7.1) |
| `test_pipeline.py`: `test_splitting_stops_at_the_relevant_contexts_line[by-refusal, by-estimate]` | renamed `test_notes_that_do_not_fit_stop_the_run_where_the_relevance_prompt_cannot_shorten_them`, same run and request counts (40 by estimate, 41 by refusal); the first note is §5.3's first row for `relevance-r2-d0` and 39 documents, the second the run folder; `notes.json` holds one `"stop"` event; there is no `splits.json`; no `TASK-` | its 39 notes of 1,000 characters are bigger than each 380-character report, so R2's stop no longer applies (Q2 (a)); `CONFIG` pins relevance v1, which cannot shorten, so R3 stops at the same point (§2.2 item 4) |
| `test_pipeline.py`: `test_a_stopped_run_keeps_its_calls_and_splits` | relevance notes of 1,000 characters (was 3,000) and `refuses_over(10_000)` (was 13,500); the assertions stand | with 3,000-character notes the notes are bigger than the part and the run now stops for the notes; the test is about R2's stop leaving its artifacts, so the part must be bigger than the notes (about 929 tokens against 400) and smaller than its rest (about 1,270), *run* |

No other test changes. `test_splitting.py`'s rows pass no notes and stand (§2.2 item 5).
R2's E2 to E4 and the E3 diff pass unchanged (*run*: 391 of 395 pass on the prototype; the
4 that fail are this table's).

### 9.3 New tests, by file

| File | Tests |
| --- | --- |
| `test_notes.py` (new) | `test_notes_are_halved_until_the_estimate_fits_their_room` (table: 33-word notes to 16 when 16 fits, to 10 when only 10 does) · `test_without_a_room_the_notes_are_halved_once` (28 to 14; with W = 14 set, to 10) · `test_a_budget_already_set_halves_from_itself_not_from_the_notes` (W = 16, notes that overshot at 25 words, to 10) · `test_no_note_is_asked_for_fewer_than_ten_words` (`start` 12 gives 10; 10 gives a stop) · `test_w_fits_every_later_request_it_is_given` (`also` with a smaller room picks the lower level; `sized_for` names it) · `test_ten_word_notes_that_cannot_fit_stop_before_anything_is_read_again` (r3con's error with the trouble's message, §5.3's note exactly, the event `"stop"` with `words` null, raised `from None`) · `test_a_refusal_at_the_floor_stops_with_the_providers_error` (the same object, its note) · `test_a_prompt_that_cannot_ask_for_a_length_stops_at_the_first_trouble` · `test_notes_json_is_written_at_each_event_and_only_then` · `test_notes_too_long_is_its_own_exception_naming_its_call` (not an instance of `ContextWindowExceededError` or `BadRequestError`; the fields; the message for each cause; the note for a direct caller) · `test_a_stop_is_a_plain_context_window_error` (by estimate and by refusal: `type(stop)` is litellm's class, not `NotesTooLong`) · `test_counting_notes_counts_each_with_its_paragraph_break` |
| `test_splitting.py` | `test_notes_bigger_than_the_part_are_handed_over_before_anything_is_sent` (`window`, `Provider()`: nothing sent, no cut, the room) · `test_a_part_bigger_than_its_notes_is_still_cut` (R2's estimate splits, with small notes) · `test_a_tie_between_part_and_notes_cuts_the_part` · `test_with_the_rest_alone_over_the_line_the_notes_are_handed_over` · `test_a_refused_part_smaller_than_its_notes_hands_them_over` (`QWEN`, `Provider(limit=0)`: `room` None, `__cause__` the refusal, nothing cut) · `test_measure_cuts_by_estimate_and_raises_the_least_room` (two documents: one bigger than the notes is cut, the other raises; the raised room is the smaller) · `test_measure_does_nothing_without_a_window` · `test_a_request_without_a_document_hands_over_only_notes_over_the_line` (`check_notes`, rows: over with notes raises; over without notes returns; under returns; under in bytes is not counted, with R2's `encodes` fixture) · `test_a_refused_request_without_a_document_hands_over_its_notes` (with notes raises from the refusal; without, returns) · `test_over_line_counts_only_what_its_bytes_cannot_settle` · `test_a_notes_signal_raised_by_send_passes_through_uncut` (a `send` raising `NotesTooLong`: it propagates as itself, nothing is cut, no event is recorded; the ruling's hazard, pinned) |
| `test_relevance.py` | `test_v2_is_v1_byte_for_byte_without_a_budget` (with and without other notes, and `max_words=None`) · `test_v2_asks_each_note_to_stay_under_its_budget` (the sentence, exactly, once) · `test_only_a_prompt_that_renders_the_budget_takes_one` (v2, v1, an overlay without `max_words`) · `test_note_texts_lists_each_document_or_part_note_without_the_empty_ones` · `test_notes_too_long_for_the_next_round_are_read_again_first` (`window`, `within_budget`: kinds `relevance-r1-d*`, then `relevance-r1-w{W}-d*`, then `relevance-r2-w{W}-d*`; `words == [W, W]`) · `test_a_round_refused_for_its_notes_reads_the_previous_round_again` (`QWEN`, `refuses_over`: a `"refusal"` event, `discarded` equal to the round's accepted calls) · `test_reread_reads_only_the_last_round_again` (the first round's notes are the same; only `relevance-r2-w*` kinds are sent) · `test_the_final_check_reads_the_last_round_again_until_it_passes` · `test_without_a_budget_relevance_takes_r2s_path` (the 40 reports of §9.2 on `window(6_000)`: R2's measured stop, `splits.json`, no read again) · `test_reread_or_a_final_check_without_a_budget_is_refused` |
| `test_schema.py` | with a `Budget`: `test_a_schema_request_over_the_line_hands_over_its_notes_before_sending` · `test_a_refused_schema_request_hands_over_its_notes_and_one_without_notes_is_raised` · `test_every_attempt_is_measured` (a retry over the line raises after attempt 1); and `test_without_a_budget_the_schema_call_is_sent_as_today` (sent over the line, the refusal propagates) |
| `test_parsing.py` | with a `Budget`: `test_parsing_measures_every_document_before_sending_any` (no request) · `test_a_refused_parse_whose_notes_are_bigger_hands_them_over`; and `test_without_a_budget_a_document_shorter_than_its_notes_is_cut_as_r2_does` |
| `test_reasoning.py` | `test_the_whole_parse_is_shown_only_when_the_first_turn_fits_the_line` (`window`: over the line, samples and the INFO line; under, the whole parse and today's prompt) · `test_a_first_turn_refused_with_the_whole_parse_is_sent_once_more_with_samples` (exactly two first-turn requests, the second with the note) · `test_a_first_turn_refused_with_samples_hands_over_its_notes` (with a `Budget`; without notes, or without a budget, the refusal is raised) · `test_a_later_turn_refused_is_raised_as_it_is` · `test_the_first_turn_without_a_parse_hands_over_notes_over_the_line` (`check_first_turn`). Guards: `test_a_huge_parse_is_shown_as_one_sample_per_field_and_says_so`, `test_a_small_parses_prompt_is_the_v1_template_with_the_parse_as_json` |
| `test_codeact.py` | `test_a_refused_first_turn_is_sent_again_with_the_prompt_the_hook_gives` (the rest of the loop keeps it) · `test_the_hook_can_give_up_by_raising` · `test_without_a_hook_a_refused_first_turn_is_raised` (guard) · `test_the_hook_is_never_called_for_a_later_turn` |
| `test_runs.py` | the kinds row `relevance-r2-w16-d7c1` reads `doc` 7, `chunk` 1 (guard) |
| `test_pipeline.py` | E1 to E6 (§9.4) · `test_notes_that_do_not_fit_are_read_again_shorter[by-refusal, by-estimate]`: §9.2's 40 reports with relevance v2 and `within_budget`; 40 entries; by estimate no request over the 5,100-token line and one read again (round 1 under 44 words, sized for reasoning, 162 requests, *run*); by refusal three (round 1 under 88 and 44, round 2 under 22, 247 requests, *run*) · `test_a_stage_sent_again_keeps_its_discarded_calls` (an unknown window refusing a parse whose notes are bigger: `structuring/parsing/calls.json` keeps the discarded calls, the event counts them, `relevance/calls.json` has `relevance-r2-w*` kinds, `relevance/result.json`'s round 2 has `max_words`) · `test_the_relevant_context_is_the_notes_reasoning_saw` · `test_no_stop_leaves_a_run_as_notes_too_long` (the floor by estimate, the floor by refusal, the v1 stop: each raises a `ContextWindowExceededError` that is not a `NotesTooLong`, with its notes) |
| `test_cli.py` | `test_a_notes_stop_prints_a_context_window_error_and_its_notes` (exit 1; stderr starts `r3con: ContextWindowExceededError:` and carries the notes) |

### 9.4 E1 to E6 in `test_pipeline.py`

Each runs `run_pipeline` with a `TaskLogger` on the default config (`load_config("default",
model=…)`), documents `read_documents(MEMOS) + quiet_sites(N - 5)`, and
`llm.answers(relevance=within_budget(memo_note), schema=ANSWERING_SCHEMA, parsing=…,
reasoning=ANSWERING_CODE)`. Sizes are the prototype's, *run*; seconds are its wall time.

- **E1 · `test_reasoning_shows_samples_when_the_whole_parse_would_not_fit`.** N = 60,
  `window(8192)`, parsing answering with the whole document as its row (so the parse,
  not the notes, is what fills the turn). No request over 6,963 tokens; the one first
  turn shows samples (5,512 tokens; 182 requests; 3 s).
- **E2 · `test_a_refused_first_turn_goes_again_once_with_samples`.** The same with `QWEN`
  and `refuses_over(tokens=8192)`. First turns: the whole parse, refused (9,635 tokens),
  then the samples, accepted; exactly two; the answer (183 requests; 2 s).
- **E3 · `test_notes_that_do_not_fit_are_read_again_shorter_by_estimate`.** N = 120,
  `window(8192)`. No request over the line; no `splits.json`; 120 entries in
  `relevant_context`; the reasoning prompt's summary headings are 1 to 120, in order;
  `notes.json` has exactly one event, `("reasoning", "estimate", "read again", 2, 16)`;
  482 requests (8 s).
- **E4 · `test_notes_that_do_not_fit_are_read_again_shorter_on_a_refusal`.** N = 120,
  `QWEN`, `refuses_over(tokens=8192)`. Every event's cause is `"refusal"`; no refused
  request is sent again under the same W; the run answers; 485 requests (6 s).
- **E5 · `test_shortening_stops_where_ten_word_notes_cannot_fit`.** N = 400,
  `window(8192)`. `ContextWindowExceededError`, its message `r3con estimated
  relevance-r2-d…`, its first note §5.3's floor row naming 400 documents and 10 words;
  400 requests, every one of round 1; `notes.json` has one `"stop"` event (15 s).
- **E6 · `test_an_ordinary_run_sends_what_a_v1_relevance_run_sends`.** The five memos,
  `window(1_000_000)`, `answering_llm`, `R3CON_DOC_WORKERS=1`, configs differing only in
  `relevance: v1` or `v2`. Every request's `messages` and `response_format` are equal, 17
  of them, and neither run writes `notes.json` (0.5 s).

They add about 35 s to the suite (*run*), E5 the longest.

## 10 · Experiments and the live tier

| E | Where | Null (false if) |
| --- | --- | --- |
| E1 | `test_pipeline.py` | a request estimated over the line; a first turn carrying the whole parse over the line; no answer |
| E2 | `test_pipeline.py` | the whole-parse first turn sent twice; the samples turn not sent; no answer |
| E3 | `test_pipeline.py` | a request over the line; a `splits.json`; a heading missing or renumbered; not 120 entries; more than one read again; no estimate event |
| E4 | `test_pipeline.py` | a refused request sent again under the same W; a stop while a W of 10 words or more is untried; a stage's discarded calls missing from `notes.json` |
| E5 | `test_pipeline.py` | a round-2 request or a read again sent; an error other than `ContextWindowExceededError`; a note not naming the 400 documents and 10 words; no stop event |
| E6 | `test_pipeline.py`, plus one diff against `f4dae00` | a byte of any request's `messages` or `response_format` differs; a `notes.json` |
| E7 | `test_experiments.py`, dispatched once | below |

**E6 against `f4dae00`.** The Implementer runs a script once with the suite's
`answering_llm` replies, the five memos, the default config, one worker and
`window(1_000_000)`, on `f4dae00`'s `src/` and on the head's (`PYTHONPATH` to each), dumps
every request, and diffs. The as-built document reports it. The prototype gave 17
requests on each side, identical; the labels differ in `rel=` only.

**E7 · shortened notes still answer.** `tests/test_experiments.py` gains one test, marked
like E6 (`live`, `experiment`, `enable_socket`, skipped without `OPENAI_API_KEY`):
`test_shortened_notes_still_answer[window-seed]`, over `window in (None, 16_384, 8_192)` ×
`seed in (1, 2, 3)`.

- **The corpus.** `read_documents(MEMOS) + quiet_sites(115)`: 120 documents.
- **The window.** A fixture registers it on the real model,
  `litellm.register_model({"openai/gpt-6-luna": {"max_input_tokens": tokens}})`, and at
  teardown registers the window `get_model_info` gave before. `register_model` merges
  into the shipped entry (the price and provider stay) and clears litellm's cache, so the
  run's `Splits` reads a line of 6,963 or 13,926 tokens (*run*). `None` leaves the
  shipped 922,000. The provider never refuses, so E7 tests the answer, and E4 the refusal.
- **The run.** `r3con.run(QUESTION, docs, params={"seed": seed}, logs_dir=tmp_path,
  completion=measured)` with the live tier's question and the default config, and
  `R3CON_DOC_WORKERS=8` (monkeypatched) to keep the parallel requests of a round under the
  account's rate limit, since R2.1 left litellm 2 retries. `measured` records each request's
  messages and forwards it to `litellm.completion`.
- **Preconditions.** At 8,192: `notes.json` has a `"read again"` event. At 16,384 and the
  shipped window: no `notes.json`.
- **Asserted.** The answer has "halloran" and `\b11\b`; exactly one CT-118 to Halloran
  record from Document 4 (E6's `mapping_records`); at 8,192 and 16,384, no two-message
  request over the line, counting its messages' contents, which is never more than
  r3con's estimate, and leaving out a parse validation retry, whose added error no
  estimate includes (R2).
- **Printed, one `E7 {json}` line per case before asserting:** the window, the seed, each
  event's round, W and cause, how many notes came back over their W (from
  `relevance/result.json`: notes with more words than their round's `max_words`), whether
  reasoning's first turn showed samples, calls, prompt and completion tokens, seconds, and
  the answer.

The null is a seed right at the shipped window and wrong at 16,384 or 8,192, or the
mapping record missing or doubled. A seed whose schema has no mapping record fails at
every window, and the as-built reads that as "no mapping record in this schema", as R2's
E6 did. If the null shows at 8,192 with 16,384 right, R3.2 as designed ends (the spec's
§4): it goes back to the Conductor with Q1's (c) or a higher floor.

**Cost and time.** About 480 calls and 2.3 million input tokens per run at 8,192, about
360 and 1.6 million at the others: about $1.70 for the nine runs at litellm's mapped
prices, plus R2's E6 ($0.06), estimated. With 8 workers, about 3 to 4 minutes per run at
8,192, so about 35 minutes for the job.

**The CI wiring.** The `experiment` job runs `pytest -m experiment`, so one dispatch runs
R2's E6 and E7 together; E6 under R3 re-confirms R2 with the new label. The job's
`timeout-minutes` goes from 45 to 90. Nothing else in `ci.yml` changes: the input exists,
and `release.yml` never sets it. The Conductor dispatches `ci.yml` on the branch's head
with `live: false` and `experiment: true`, once.

**The live tier is unchanged.** `tests/test_live.py` is not edited. Its label reads
`rel=v2`; it sends 0.2.0's requests, since nothing is shortened on 922,000 tokens.

### 10.1 Measured (Implementer, 2026-10-05)

**E1 to E6 in the suite** match §9.4's numbers: E1 182 requests and a 5,512-token first
turn with samples; E2 183, the whole parse refused at 9,635 tokens, then samples; E3
482, one event `("reasoning", "estimate", "read again", 2, 16)`; E4 485, every event a
refusal; E5 400 requests, all of round 1, and the floor's note (§2.5 item 5); E6 17
requests, identical between `rel=v1` and `rel=v2`. §9.2's 40 reports: 162 requests by
estimate (round 1 read again under 44 words, sized for reasoning) and 247 by refusal
(88, 44, then 22 for reasoning's refused samples turn).

**E6 against `f4dae00`** (*run* once, by a script outside the repo: the suite's
`answering_llm` replies, the five memos, the default config, one worker,
`window(1_000_000)`, each `src/` on `PYTHONPATH`): 17 requests on each side, every
`messages` and `response_format` byte-identical, the same answer. The labels differ in
`rel=` only.

**E7, live, once** (dispatched at `1d8e0a3`, run 37352518283, `experiment` job
111908028043; 24 minutes for R2's E6 and E7 together).

- *The claim and the rule, from above, before the result:* shortened notes still answer.
  The null is a seed right at the shipped window and wrong at 16,384 or 8,192, or the
  CT-118 to Halloran record from Document 4 missing or doubled; the precondition is a
  read again at 8,192 and none elsewhere. The null at 8,192 with 16,384 right ends R3.2
  as designed.
- *Conditions:* `openai/gpt-6-luna` through litellm 1.104.0 (the lock), its window
  registered in-process; the default config (`rel=v2`, `reason=v2`); the five memos and
  115 quiet site memos; seeds 1 to 3 in `params`; eight documents at a time; the live
  tier's question. About $1.81 for the nine runs at litellm's mapped prices ($0.10 and
  $0.50 a million input and output tokens), and $0.06 for R2's E6 beside them. *Noise
  floor:* R2's E6 in the same job, 9 of 9 right; the shipped window here, 3 of 3.
- *How it was read:* the test raised before printing in 8 cases (§2.5 item 10); each row
  below is recomputed from that case's run folder, and its seconds are from the run
  folders' start times, so they are approximate.

| Window (line) | Seed | Read again (round, W, cause) | Notes over W | Reasoning saw | Largest two-message request | Calls | Prompt tokens | Completion tokens | ≈ s | Mapping records | Answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shipped, 922,000 (783,700) | 1 | none | — | the whole parse | 13,817 | 363 | 1,677,501 | 34,813 | 112 | 1 | 11, Halloran ✓ |
| shipped | 2 | none | — | the whole parse | 15,435 | 363 | 1,684,086 | 36,361 | 110 | 1 | 11, Halloran ✓ |
| shipped | 3 | none | — | the whole parse | 16,250 | 363 | 1,694,140 | 41,367 | 130 | 1 | 11, Halloran ✓ |
| 16,384 (13,926) | 1 | none | — | samples | 8,424 | 362 (1.00x) | 1,647,137 (0.98x) | 38,096 | 120 | 1 | 11, Halloran ✓ |
| 16,384 | 2 | none | — | samples | 8,651 | 363 (1.00x) | 1,684,364 (1.00x) | 38,962 | 122 | 1 | 11, Halloran ✓ |
| 16,384 | 3 | none | — | samples | 8,405 | 363 (1.00x) | 1,648,642 (0.97x) | 36,422 | 115 | 1 | 11, Halloran ✓ |
| 8,192 (6,963) | 1 | (2, 16, estimate) | 34 of 120 | samples | 6,118 | 483 (1.33x) | 2,019,451 (1.20x) | 53,638 | 148 | 1 | 11, Halloran ✓ |
| 8,192 | 2 | (2, 16, estimate) | 40 of 120 | samples | 6,150 | 488 (1.34x) | 2,072,644 (1.23x) | 56,954 | 161 | 1 | **22**, Halloran ✗ |
| 8,192 | 3 | (2, 16, estimate) | 42 of 120 | samples | 6,203 | 483 (1.33x) | 2,043,985 (1.21x) | 58,398 | 156 | 1 | 11, Halloran ✓ |

Multiples are against the same seed on the shipped window. At 8,192 every event is
the one E3 predicts: after round 2, reasoning's first turn as known then leaves the notes
3,971 tokens, and round 2 is read again once, each note under 16 words. A third of the
notes came back over 16 words, and no request then went over the line (6,203 at most
against 6,963), so nothing was read a second time. Seed 2's reasoning took 7 turns, five
of them without a code block, against 2 for every other case but 16,384 seed 1 (1).

**The result:** the null shows, at 8,192 for seed 2 (§2.5 item 9). R3.2 as designed
goes back to the Conductor.

**E7 again, six seeds** (at the Conductor's call; dispatched at `b337d30` with
`e7_seeds` `1,2,3,4,5,6`; run 37358511251, `experiment` job 111927777393; 46 minutes for
R2's E6 and E7's 18 runs; about $3.64 for E7 at the mapped prices, $0.06 for R2's E6,
which was right 9 of 9). The conditions are the first run's. Before the dispatch the
harness was dry-run offline, the network blocked, through reasoning that commits at
once, looks first, or answers a turn without a code block first: 18 of 18. In the job
every case printed its line, and each row below matches it.

| Window (line) | Seeds right | Read again (round, W, cause, sized for) | Notes over W, of 120 | Reasoning saw | Largest two-message request | Calls (vs shipped, same seed) | Prompt tokens (vs shipped) | Completion tokens | Mapping records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shipped, 922,000 (783,700) | 6 of 6 | none | — | the whole parse | 15,038 to 17,653 | 362 to 363 | 1,642,210 to 1,704,603 | 34,962 to 39,812 | 1 in each |
| 16,384 (13,926) | 6 of 6 | none | — | samples | 8,463 to 8,835 | 362 to 365 (1.00 to 1.01x) | 1,653,966 to 1,698,651 (0.97 to 1.02x) | 37,770 to 39,407 | 1 in each |
| 8,192 (6,963) | **5 of 6** (seed 3: 16) | (2, 16 or 17, estimate, reasoning), once in each | 38 to 54 | samples | 6,132 to 6,343 | 482 to 484 (1.33 to 1.34x) | 2,031,706 to 2,063,052 (1.20 to 1.24x) | 52,424 to 61,867 | 1 in each |

W is 16 or 17 as the notes' first mean was 32 to 35 words. Both runs, by window: the
shipped window 9 of 9, 16,384 9 of 9, 8,192 7 of 9.

**Every wrong answer** (two in 27 runs), from its run folder:

| Run | Window, seed | Answer | Did the notes keep the facts? | Was the parse right? | Reasoning saw | What it summed |
| --- | --- | --- | --- | --- | --- | --- |
| first | 8,192, 2 | 22 | yes: Northgate "five equipment incidents under contractor code CT-118", Riverside "6 equipment incidents under CT-118", the registry's mapping | each record is true to its memo, but Northgate's total (5) sits beside its breakdown (3 conveyor stoppages, 2 HVAC trips), and Riverside's (6) beside 4 + 2, in one list | samples | every CT-118 record, 5 + 3 + 2 + 6 + 4 + 2, after five turns without a code block |
| second | 8,192, 3 | 16 | yes: "5 equipment incidents under contractor code CT-118", "six equipment incidents at a CT-118 site", the mapping | the same, Northgate's breakdown only (5, 3, 2; then 6) | samples | every Q3 CT-118 record, 5 + 3 + 2 + 6, on its first turn |

**What the evidence points at.** The parse kept a site's sub-counts as records beside
its total in 5 of the 27 runs (2 on the shipped window, 1 at 16,384, 2 at 8,192), a
property of the schema the run proposes, not of the window. Without such records, 22 of
22 runs answered 11, at every window. With them, the 2 runs shown the whole parse
answered 11 without summing; of the 3 shown samples, the one at 16,384 answered 11 from
the notes after four turns, and both at 8,192 summed every record. The shipped window
was right 9 of 9, so there is no noise there that the other windows merely share, and
the 16-word notes kept the facts in all nine runs at 8,192. So the evidence points at
R3.1's sample view, which hides from reasoning that the parse keeps sub-counts as
records of their own; not at the shorter notes losing facts. It cannot exclude that
full-length notes make reasoning less likely to sum the records: the one run shown
samples beside such records that answered right had them, one case. By §10's rule the
null shows again (8,192, seed 3, right on the shipped window).

## 11 · Changes to `src/` by module, and the version

| Module | Item |
| --- | --- |
| `notes.py` (new) | R3.2 (`MIN_NOTE_WORDS`, `NotesTooLong(Exception)`, `count_notes`, `Budget` with `splits`, `_tokens_at`) |
| `splitting.py` | R3.1 (`model`, `over_line`); R3.2 (`read_in_parts(notes=)`, the rule in the measured and refusal steps, `measure`, `check_notes`, the message helper shared with `_not_sent`, the docstring and the INFO line) |
| `runtime/codeact.py` | R3.1 (`on_first_turn_too_long`) |
| `stages/reasoning.py` | R3.1 (`splits=`, `_system_prompt`, `_sample_view`, the choice and the hook); R3.2 (`budget=`, `check_first_turn`, the notes' check, Raises) |
| `stages/relevance.py` | R3.2 (`RelevantContext.words`, `note_texts`, `takes_word_budget`, `max_words`, the round loop, `budget=`, `reread=`, `final_check=`) |
| `stages/structuring/schema.py` | R3.2 (`budget=`, every attempt measured, the refusal, Raises) |
| `stages/structuring/parsing.py` | R3.2 (`budget=`, `notes`, `measure`, Raises) |
| `prompts/relevance/v2.yaml` (new), `configs/default.yaml` | R3.2 |
| `pipeline.py` | R3.1 (`splits` to `reason`); R3.2 (`Budget`, the final check, `fitting`, `_recorded_stage(run=)`, `_relevance_result`) |
| `runs.py`, `r3con.py` | docstrings |

Outside `src/`: `tests/` (§9), `.github/workflows/ci.yml` (one timeout), `README.md` (one
line), `pyproject.toml` and `uv.lock` (the version), and this file.

**The version is 0.3.0.** r3context is at 0.x, where a minor release is the one that may
change behaviour. R3 changes what the default run is (its label reads `rel=v2`, so runs
group apart from 0.2.0's), changes two R2 behaviours that 0.2.0 shipped (its stop and its
cutting where the notes are the bigger part), and adds public names (`notes.py`,
`RelevantContext.words`, new keyword arguments). It removes nothing, so no 1.0 question
arises; it is more than a fix, so not 0.2.1. Step 8 sets `version = "0.3.0"` and runs
`uv lock`, which rewrites only the project's own entry.

## 12 · Outside r3con, and risks

**Depends on** (no collaborator's code):

- **litellm 1.101 to 1.104**, everything R2 lists, and: `register_model` merges a partial
  entry and clears the lookup cache (*run*); a provider's refusal of a reasoning turn
  reaches `run_codeact` as `ContextWindowExceededError`, as it reaches the other stages.
- **jinja2**: an undefined or `None` variable in `{% if %}` is false (v2's byte identity).
- **The model obeys a word budget, roughly.** A note over its W halves W at the next
  trouble; E7 reports how often it happens.

**Risks, and what would show them:**

- **Shorter notes answer worse.** E7's null. The as-built reports E7 per window and seed.
- **The estimate of notes at W is a stand-in.** A note cut at its W-th word is denser than
  one a model writes under W, so the estimate errs high and W comes out lower than needed,
  never higher (*run*: E3 sized at 16, and reasoning's real first turn then had about 900
  tokens to spare under the line). A model
  that writes longer notes than asked meets the backstop: the stage's own check halves W
  again, one more read.
- **With an unknown window, reasoning after a read again tries the whole parse first**,
  and is refused again when it still does not fit (§1.1's unknown 120: three refused
  first turns, 9 requests through litellm). It is R3.1's rule applied afresh to shorter
  notes.
- **Counting near the line doubles** (§6.2): each request is counted by `measure` and by
  `read_in_parts`. 400 memos take 15 s in the suite, most of it counting.
- **Reading again costs a round.** Round 1 again is cheap (each call one document); round
  2 again costs what round 2 cost (120 memos: 120 calls of about 6,300 tokens). §2.3
  item 2 is the case where cutting the document would have been cheaper.
- **A loop that catches `NotesTooLong` must hand it to `Budget.shorten`.** Today there are
  two (`surface_relevance`'s and the pipeline's). A third that swallowed one would lose the
  stop; `test_no_stop_leaves_a_run_as_notes_too_long` covers the run.

**Known limits** (the Conductor's ruling adds the first):

- **Notes in a language written without spaces cannot be sized by words.** A word is what
  `str.split()` separates, so a Japanese or Chinese note counts as a word or two; no
  halving can shorten it, and the run stops at the first notes trouble with the floor's
  note, where R2 would have stopped too (§2.3 item 4).
- **A server that truncates instead of refusing** is measured only through a registered
  window, as in R2.
- **The spec's out-of-scope stays out:** a reasoning conversation that outgrows the window
  in later turns, the cost of N notes N times, choosing which documents matter, a stage
  prompt over the window by itself.

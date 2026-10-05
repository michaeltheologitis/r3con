import json
import logging

import litellm
import pytest

from r3con.notes import MIN_NOTE_WORDS, Budget, NotesTooLong, count_notes
from r3con.runs import TaskLogger
from r3con.splitting import Splits

QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
OUTGROWN = "The relevant context has outgrown the model's window."


def notes(words: int, n: int = 20) -> list[str]:
    """``n`` notes of ``words`` words each: a note of W words with its paragraph break
    is W + 1 tokens, so 20 notes of 33 words are 680, cut at 16 words 340, at 10 220."""
    return [" ".join(["word"] * words)] * n


def trouble(
    words: int,
    *,
    room: int | None = None,
    call: str = "reasoning",
    refusal: litellm.ContextWindowExceededError | None = None,
    n: int = 20,
) -> NotesTooLong:
    """A request of 8,000 tokens carrying ``n`` notes of ``words`` words, which did not
    fit by estimate with ``room`` for its notes, or was refused with ``refusal``."""
    texts = notes(words, n)
    return NotesTooLong(
        f"{call} did not fit.",
        model=QWEN,
        call=call,
        cause="estimate" if refusal is None else "refusal",
        estimate=8_000,
        notes=texts,
        notes_tokens=count_notes(texts),
        room=room,
        refusal=refusal,
    )


def too_long() -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message="This model's maximum context length is 8192 tokens.",
        model=QWEN,
        llm_provider="hosted_vllm",
    )


@pytest.fixture
def logger(tmp_path) -> TaskLogger:
    return TaskLogger("run", root=tmp_path)


@pytest.fixture
def budget(window, logger) -> Budget:
    return Budget(Splits([], model=window(8_192)), task_logger=logger)


def events(logger: TaskLogger) -> list[dict]:
    return json.loads((logger.dir / "notes.json").read_text())["events"]


@pytest.mark.parametrize(("room", "words"), [(400, 16), (300, 10)])
def test_notes_are_halved_until_the_estimate_fits_their_room(
    budget, logger, caplog, room, words
):
    caplog.set_level(logging.WARNING, logger="r3con.notes")
    assert budget.shorten(trouble(33, room=room), round_idx=2) == words
    assert budget.words == words
    [event] = events(logger)
    assert (event["action"], event["round"], event["words"]) == ("read again", 2, words)
    assert caplog.messages == [
        (
            "reasoning: the notes of 20 documents come to about 680 tokens, and the "
            "request with them to about 8,000, over the 6,963-token line; reading "
            f"round 2 again with each note under {words} words"
        )
    ]


def test_without_a_room_the_notes_are_halved_once(budget, caplog):
    caplog.set_level(logging.WARNING, logger="r3con.notes")
    assert budget.shorten(trouble(28, refusal=too_long()), round_idx=1) == 14
    assert budget.shorten(trouble(14, refusal=too_long()), round_idx=1) == 10
    assert caplog.messages[0] == (
        "reasoning was refused as too long; the notes of 20 documents it carries "
        "(about 580 tokens) are the bigger part, so round 1 is read again with each "
        "note under 14 words"
    )


def test_a_budget_already_set_halves_from_itself_not_from_the_notes(budget):
    assert budget.shorten(trouble(33, room=400), round_idx=2) == 16
    overshot = trouble(25, room=6_000)
    assert budget.shorten(overshot, round_idx=2) == 10


def test_no_note_is_asked_for_fewer_than_ten_words(budget, window):
    assert MIN_NOTE_WORDS == 10
    assert budget.shorten(trouble(12, refusal=too_long()), round_idx=1) == 10
    with pytest.raises(litellm.ContextWindowExceededError):
        budget.shorten(trouble(12, refusal=too_long()), round_idx=1)
    fresh = Budget(Splits([], model=window(8_192)))
    with pytest.raises(litellm.ContextWindowExceededError):
        fresh.shorten(trouble(10, refusal=too_long()), round_idx=1)


def test_w_fits_every_later_request_it_is_given(budget, logger):
    round_2 = trouble(33, room=400, call="relevance-r2-d0")
    reasoning = trouble(33, room=300, call="reasoning")
    assert budget.shorten(round_2, round_idx=1, also=[reasoning]) == 10
    [event] = events(logger)
    assert (event["call"], event["room"], event["sized_for"]) == (
        "relevance-r2-d0",
        300,
        "reasoning",
    )


@pytest.mark.parametrize(
    ("room", "note"),
    [
        (
            200,
            (
                "r3con: even at 10 words each, the notes of 20 documents would take about "
                "220 tokens, over the 200 that reasoning leaves them, so reading them "
                f"shorter cannot help. {OUTGROWN}"
            ),
        ),
        (
            -5,
            (
                "r3con: what reasoning sends beside its notes fills the 6,963-token line "
                f"by itself, so reading them shorter cannot help. {OUTGROWN}"
            ),
        ),
    ],
    ids=["at-the-floor", "no-room"],
)
def test_ten_word_notes_that_cannot_fit_stop_before_anything_is_read_again(
    budget, logger, room, note
):
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        budget.shorten(trouble(33, room=room), round_idx=2)
    assert "reasoning did not fit." in str(stop.value)
    assert stop.value.llm_provider == "r3con"
    assert stop.value.__notes__ == [note]
    assert stop.value.__cause__ is None and stop.value.__suppress_context__
    [event] = events(logger)
    assert (event["action"], event["words"], event["room"]) == ("stop", None, room)
    assert budget.words is None


def test_a_refusal_at_the_floor_stops_with_the_providers_error(budget, logger):
    refused = too_long()
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        budget.shorten(trouble(9, refusal=refused), round_idx=2)
    assert stop.value is refused
    assert stop.value.__notes__ == [
        (
            "r3con: reasoning was refused as too long, and the notes of 20 documents it "
            "carries are already about 9 words each, with 10 the fewest a note is asked "
            f"for, so reading them shorter cannot help. {OUTGROWN}"
        )
    ]
    [event] = events(logger)
    assert (event["cause"], event["room"], event["error"]) == (
        "refusal",
        None,
        str(refused),
    )


def test_a_prompt_that_cannot_ask_for_a_length_stops_at_the_first_trouble(
    window, logger
):
    budget = Budget(
        Splits([], model=window(8_192)), can_shorten=False, task_logger=logger
    )
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        budget.shorten(
            trouble(33, room=400, call="relevance-r2-d0"),
            round_idx=1,
            also=[trouble(40, room=300, n=21)],
        )
    assert stop.value.__notes__ == [
        (
            "r3con: relevance-r2-d0 carries the notes of 20 documents (about 680 tokens), "
            "the bigger part of a request that does not fit, and the relevance prompt "
            "this run pins cannot ask for shorter notes (the shipped v2 can). "
            f"{OUTGROWN}"
        )
    ]
    assert [event["action"] for event in events(logger)] == ["stop"]


def test_notes_json_is_written_at_each_event_and_only_then(budget, logger):
    path = logger.dir / "notes.json"
    assert not path.exists()
    budget.shorten(trouble(33, room=400), round_idx=2, discarded=3)
    assert json.loads(path.read_text()) == {
        "model": "hosted_vllm/window-8192",
        "max_input_tokens": 8_192,
        "margin_percent": 15,
        "line": 6_963,
        "events": [
            {
                "call": "reasoning",
                "cause": "estimate",
                "estimate": 8_000,
                "notes": 680,
                "room": 400,
                "sized_for": "reasoning",
                "discarded": 3,
                "action": "read again",
                "round": 2,
                "words": 16,
                "error": None,
            }
        ],
    }
    with pytest.raises(litellm.ContextWindowExceededError):
        budget.shorten(trouble(16, room=100), round_idx=2)
    assert [event["action"] for event in events(logger)] == ["read again", "stop"]


def test_notes_too_long_is_its_own_exception_naming_its_call():
    refused = too_long()
    handed = trouble(33, call="parse-d0", refusal=refused)
    assert not isinstance(handed, litellm.BadRequestError)
    assert not isinstance(handed, litellm.ContextWindowExceededError)
    assert (handed.model, handed.call, handed.cause, handed.estimate) == (
        QWEN,
        "parse-d0",
        "refusal",
        8_000,
    )
    assert (handed.notes, handed.notes_tokens) == (notes(33), 680)
    assert (handed.room, handed.refusal) == (None, refused)
    assert str(handed) == "parse-d0 did not fit."
    assert handed.__notes__ == [
        (
            "r3con: the notes parse-d0 carries (about 680 tokens) are the bigger part of a "
            "request that does not fit; hand this to the budget's shorten and read their "
            "round again, as r3con.run does."
        )
    ]


@pytest.mark.parametrize("refused", [False, True], ids=["by-estimate", "by-refusal"])
def test_a_stop_is_a_plain_context_window_error(budget, refused):
    handed = trouble(
        9, room=None if refused else 100, refusal=too_long() if refused else None
    )
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        budget.shorten(handed, round_idx=2)
    assert type(stop.value) is litellm.ContextWindowExceededError
    assert not isinstance(stop.value, NotesTooLong)
    assert not any("hand this to the budget" in note for note in stop.value.__notes__)


def test_counting_notes_counts_each_with_its_paragraph_break():
    # ".\n\n" is one token, so a note that ends a sentence costs no more with its break.
    assert count_notes(["A note."]) == 3
    assert count_notes(["word word"]) == 3
    assert count_notes(["A note.", "word word"]) == 6
    assert count_notes([]) == 0


def test_notes_asked_for_ten_words_that_still_overshoot_stop_with_the_floor_named(
    budget,
):
    assert budget.shorten(trouble(12, refusal=too_long()), round_idx=2) == 10
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        budget.shorten(trouble(14, room=6_000), round_idx=2)
    assert stop.value.__notes__ == [
        (
            "r3con: reasoning is estimated over the line, and the notes of 20 documents it "
            "carries are already about 14 words each, with 10 the fewest a note is asked "
            f"for, so reading them shorter cannot help. {OUTGROWN}"
        )
    ]

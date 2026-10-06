import json
import logging
from pathlib import Path

import litellm
import pytest

from r3con.notes import NotesTooLong
from r3con.runs import TaskLogger
from r3con.splitting import Splits, halve

MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
REST = "Summarise the document for the task."
# 16 equal paragraphs, 3,444 characters, 720 tokens: halves and quarters fall on breaks.
PARAGRAPHS = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(16))
# Two short paragraphs, then a long one: the first cut leaves 604 and 900 characters.
LOPSIDED = "\n\n".join(["a " * 150, "b " * 150, "c " * 450])
JAPANESE = (
    "契約書の第三条に基づき、当事者は誠実に協議する。"  # 24 characters, 31 tokens
)
# A note of 33 words is 34 tokens with its paragraph break; 20 of them are 680.
NOTE = " ".join(["word"] * 33)
NOTES = [NOTE] * 20


class Provider:
    """Stands for the provider at the loop's boundary: it refuses a part longer than
    ``limit`` characters, as litellm reports a refusal, and records what it was sent."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.sent: list[tuple[str, str]] = []
        self.refusals: list[litellm.ContextWindowExceededError] = []

    def __call__(self, part: str, kind: str) -> str:
        self.sent.append((kind, part))
        if self.limit is not None and len(part) > self.limit:
            self.refusals.append(
                litellm.ContextWindowExceededError(
                    message=f"This model's maximum context length is {self.limit}.",
                    model=QWEN,
                    llm_provider="hosted_vllm",
                )
            )
            raise self.refusals[-1]
        return f"read {kind}"

    @property
    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.sent]


@pytest.fixture
def logger(tmp_path) -> TaskLogger:
    return TaskLogger("run", root=tmp_path)


def record(logger: TaskLogger) -> dict | None:
    """``splits.json``, or ``None`` when the run wrote none."""
    path = logger.dir / "splits.json"
    return json.loads(path.read_text()) if path.exists() else None


def read(document, send, *, model=QWEN, rest=REST, task_logger=None):
    """``document`` read as the only document of a run, in one ``parse`` call."""
    splits = Splits([document], model=model, task_logger=task_logger)
    return splits, splits.read_in_parts(0, call="parse", rest=rest, send=send)


def with_notes(notes: list[str], prompt: str = REST) -> str:
    """A rest that renders ``notes`` after ``prompt``, each followed by a paragraph
    break, as a stage's prompt does."""
    return prompt + "\n\n" + "".join(note + "\n\n" for note in notes)


HALVES = {
    "a paragraph break beats a nearer line break": (
        "a" * 40 + "\n\n" + "b" * 7 + "\n" + "c" * 50,
        42,
    ),
    "a paragraph break outside the middle half loses to a line break inside it": (
        "a" * 10 + "\n\n" + "b" * 40 + "\n" + "c" * 47,
        53,
    ),
    "a sentence end beats a nearer space": (
        "x" * 30 + ". " + "y" * 30 + " " + "z" * 37,
        32,
    ),
    "。 ends a sentence": ("あ" * 30 + "。" + "い" * 69, 31),
    "a space when there is no sentence end": ("x" * 45 + " " + "y" * 54, 46),
    "the exact middle for a text with no whitespace": ("x" * 101, 50),
}


@pytest.mark.parametrize(("text", "cut"), HALVES.values(), ids=HALVES.keys())
def test_a_part_is_cut_at_the_break_nearest_its_middle(text, cut):
    assert halve(text) == cut


@pytest.mark.parametrize(
    "text", [text for text, _ in HALVES.values()] + ["ab", "a b", "abc", "\n\nab"]
)
def test_halves_rejoin_to_the_text_and_neither_is_under_a_quarter(text):
    cut = halve(text)
    first, second = text[:cut], text[cut:]
    assert first + second == text
    assert min(len(first), len(second)) >= max(1, len(text) // 4)


def test_a_text_of_one_character_cannot_be_halved():
    with pytest.raises(ValueError):
        halve("a")


def test_the_line_is_litellms_window_less_the_margin(window):
    model = window(10_000)
    splits = Splits([], model=model)
    assert (splits.max_input_tokens, splits.margin_percent) == (10_000, 15)
    assert splits.line == 8_500
    assert Splits([], model=model, margin_percent=0).line == 10_000


@pytest.mark.parametrize("model", [QWEN, "my-router-alias", ""])
def test_a_model_litellm_does_not_map_has_no_window_and_no_line(model, caplog):
    caplog.set_level(logging.INFO, logger="r3con")
    splits = Splits(["text"], model=model)
    assert (splits.max_input_tokens, splits.line) == (None, None)
    assert "gives no input window" in caplog.text


@pytest.mark.parametrize(
    ("model", "window_tokens"), [("my-router-alias", None), ("mapped", 10_000)]
)
def test_looking_up_the_window_prints_nothing(
    window, capsys, monkeypatch, model, window_tokens
):
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    level = logging.getLogger("LiteLLM").level
    if window_tokens:
        model = window(window_tokens)
    assert Splits(["text"], model=model).max_input_tokens == window_tokens
    assert capsys.readouterr().out == ""
    assert litellm.suppress_debug_info is False
    assert logging.getLogger("LiteLLM").level == level


def test_a_document_that_fits_is_sent_whole_under_its_plain_kind(logger):
    provider = Provider()
    splits, results = read(PARAGRAPHS, provider, task_logger=logger)
    assert provider.kinds == ["parse-d0"]
    assert results == ["read parse-d0"]
    assert splits.parts(0) == [PARAGRAPHS]
    assert record(logger) is None


def test_a_refused_document_is_read_in_2_then_4_parts_in_order(logger):
    provider = Provider(limit=len(PARAGRAPHS) // 3)
    splits, results = read(PARAGRAPHS, provider, task_logger=logger)
    assert provider.kinds == [
        "parse-d0",
        "parse-d0c0",
        *[f"parse-d0c{k}" for k in range(4)],
    ]
    assert results == [f"read parse-d0c{k}" for k in range(4)]
    assert [part for _, part in provider.sent[-4:]] == splits.parts(0)
    assert "".join(splits.parts(0)) == PARAGRAPHS
    document = record(logger)["documents"]["0"]
    assert len(document["cuts"]) == 3
    assert [(e["cause"], e["action"], e["parts"]) for e in document["events"]] == [
        ("refusal", "split", 2),
        ("refusal", "split", 4),
    ]
    assert document["parts"] == {"parse": 4}


def test_a_refusal_after_accepted_parts_discards_them_and_rereads_the_level(logger):
    provider = Provider(limit=700)
    read(LOPSIDED, provider, task_logger=logger)
    assert provider.kinds == [
        "parse-d0",
        "parse-d0c0",
        "parse-d0c1",
        *[f"parse-d0c{k}" for k in range(4)],
    ]
    events = record(logger)["documents"]["0"]["events"]
    assert [event["discarded"] for event in events] == [0, 1]


def test_a_document_once_split_starts_the_next_call_from_its_parts(logger):
    splits, _ = read(
        PARAGRAPHS, Provider(limit=len(PARAGRAPHS) // 3), task_logger=logger
    )
    provider = Provider()
    results = splits.read_in_parts(0, call="relevance-r2", rest=REST, send=provider)
    assert provider.kinds == [f"relevance-r2-d0c{k}" for k in range(4)]
    assert len(results) == 4
    assert record(logger)["documents"]["0"]["parts"] == {"parse": 4, "relevance-r2": 4}


def test_over_the_line_a_document_is_split_before_anything_is_sent(window, logger):
    provider = Provider()
    read(PARAGRAPHS, provider, model=window(500), task_logger=logger)
    assert provider.kinds == ["parse-d0c0", "parse-d0c1"]
    events = record(logger)["documents"]["0"]["events"]
    assert [(e["cause"], e["error"]) for e in events] == [("estimate", None)]
    assert events[0]["estimate"] > 425


def test_a_refusal_splits_even_when_the_estimate_said_it_fits(window, logger):
    provider = Provider(limit=len(PARAGRAPHS) // 3)
    _, results = read(PARAGRAPHS, provider, model=window(1_000_000), task_logger=logger)
    assert len(results) == 4
    events = record(logger)["documents"]["0"]["events"]
    assert [event["cause"] for event in events] == ["refusal", "refusal"]
    assert "maximum context length" in events[0]["error"]


STOPS = {
    "a refused part shorter than the rest, with an unknown window": (
        None,
        "Context. " * 600,
        PARAGRAPHS[:300],
        ["parse-d0"],
        "the part is about 64 tokens and the prompt and notes sent with it about 1,201",
    ),
    "the rest alone over the line": (
        1_000,
        "Context. " * 600,
        PARAGRAPHS,
        [],
        (
            "the prompt and notes sent with it are about 1,201 tokens, over the "
            "850-token line by themselves"
        ),
    ),
    "a part of one character that is still refused": (
        None,
        "s",
        "abcd",
        ["parse-d0", "parse-d0c0", "parse-d0c0"],
        (
            "the part is one character, and the prompt and notes sent with it (about "
            "1 tokens) leave it no room"
        ),
    ),
}


@pytest.mark.parametrize(
    ("tokens", "rest", "document", "kinds", "clause"), STOPS.values(), ids=STOPS.keys()
)
def test_splitting_stops_where_more_parts_cannot_help(
    window, logger, tokens, rest, document, kinds, clause
):
    provider = Provider(limit=0)
    model = window(tokens) if tokens else QWEN
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        read(document, provider, model=model, rest=rest, task_logger=logger)
    assert provider.kinds == kinds
    if provider.refusals:
        assert stop.value is provider.refusals[-1]
    else:
        assert "r3con estimated parse-d0 at " in str(stop.value)
    [note] = stop.value.__notes__
    assert note == (
        "r3con: reading documents[0] (Document 1) in more parts cannot help in "
        f"{kinds[-1] if kinds else 'parse-d0'}: {clause}. The relevant context has "
        "outgrown the model's window."
    )
    assert "TASK-" not in note
    assert record(logger)["documents"]["0"]["events"][-1]["action"] == "stop"


def test_the_measured_stop_says_what_was_estimated_against_which_window(window):
    model = window(4_000)
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        read("word " * 100, Provider(), model=model, rest=" the" * 3_500)
    assert (
        "r3con estimated parse-d0 at 3,601 tokens, over the 3,400-token line (the "
        "4,000-token input window litellm's model map gives hosted_vllm/window-4000, "
        "less 15%); it was not sent."
    ) in str(stop.value)


def test_with_a_known_window_a_part_shorter_than_the_rest_is_cut_again(window, logger):
    rest = " the" * 765
    document = PARAGRAPHS[: len(PARAGRAPHS) // 3]
    provider = Provider()
    splits, results = read(
        document, provider, model=window(1_000), rest=rest, task_logger=logger
    )
    assert len(results) == len(splits.parts(0)) > 2
    assert "".join(splits.parts(0)) == document
    for _, part in provider.sent:
        assert len(litellm.encode(text=rest)) + len(litellm.encode(text=part)) <= 850
    events = record(logger)["documents"]["0"]["events"]
    assert {event["cause"] for event in events} == {"estimate"}


def test_with_a_known_window_a_refused_part_shorter_than_the_rest_is_cut_again(
    window, logger
):
    provider = Provider(limit=len(PARAGRAPHS) // 3)
    _, results = read(
        PARAGRAPHS,
        provider,
        model=window(10_000),
        rest=" the" * 2_000,
        task_logger=logger,
    )
    assert results == [f"read parse-d0c{k}" for k in range(4)]
    events = record(logger)["documents"]["0"]["events"]
    assert [event["cause"] for event in events] == ["refusal", "refusal"]


@pytest.fixture
def encodes(monkeypatch) -> list[str]:
    """The texts litellm is asked to count, recorded at litellm's boundary."""
    texts: list[str] = []
    encode = litellm.encode

    def recording(**kwargs):
        texts.append(kwargs["text"])
        return encode(**kwargs)

    monkeypatch.setattr(litellm, "encode", recording)
    return texts


def test_a_request_under_the_line_in_bytes_is_not_counted(window, encodes):
    memo = (MEMOS / "01_northgate.txt").read_text()
    provider = Provider()
    read(memo, provider, model=window(1_000_000))
    assert provider.kinds == ["parse-d0"]
    assert encodes == []


@pytest.mark.parametrize(
    ("document", "kinds"),
    [
        (" the" * 300, ["parse-d0"]),
        (JAPANESE * 30, ["parse-d0c0", "parse-d0c1"]),
    ],
    ids=["ascii-over-in-bytes-under-in-tokens", "japanese-under-in-characters"],
)
def test_a_request_over_the_line_in_bytes_is_counted(window, encodes, document, kinds):
    assert len(("s" + document).encode()) > 850
    provider = Provider()
    read(document, provider, model=window(1_000), rest="s")
    assert document in encodes
    assert provider.kinds == kinds


def test_another_error_passes_through_and_never_splits(logger):
    def bad_request(part, kind):
        raise litellm.BadRequestError(
            message="bad request", model=QWEN, llm_provider="hosted_vllm"
        )

    splits = Splits([PARAGRAPHS], model=QWEN, task_logger=logger)
    with pytest.raises(litellm.BadRequestError) as error:
        splits.read_in_parts(0, call="parse", rest=REST, send=bad_request)
    assert not isinstance(error.value, litellm.ContextWindowExceededError)
    assert splits.parts(0) == [PARAGRAPHS]
    assert record(logger) is None


def test_splits_json_is_written_as_each_split_happens(logger):
    provider = Provider(limit=len(PARAGRAPHS) // 3)
    seen: list[int] = []

    def send(part, kind):
        written = record(logger)
        seen.append(len(written["documents"]["0"]["events"]) if written else 0)
        return provider(part, kind)

    read(PARAGRAPHS, send, task_logger=logger)
    assert seen == [0, 1, 2, 2, 2, 2]


def test_over_line_counts_only_what_its_bytes_cannot_settle(window, encodes):
    splits = Splits([], model=window(1_000))
    assert splits.over_line("s" * 400, "t" * 450) is None
    assert encodes == []
    assert splits.over_line(" the" * 300) is None
    assert splits.over_line(" the" * 300, " the" * 600) == 900
    assert encodes == [" the" * 300, " the" * 300, " the" * 600]
    assert Splits([], model=QWEN).over_line(" the" * 3_000) is None


def test_notes_bigger_than_the_part_are_handed_over_before_anything_is_sent(
    window, logger
):
    document, provider = PARAGRAPHS[:900], Provider()
    splits = Splits([document], model=window(1_000), task_logger=logger)
    with pytest.raises(NotesTooLong) as handed:
        splits.read_in_parts(
            0, call="parse", rest=with_notes(NOTES), send=provider, notes=NOTES
        )
    assert provider.sent == []
    assert splits.parts(0) == [document]
    assert record(logger) is None
    trouble = handed.value
    assert (trouble.call, trouble.cause, trouble.estimate) == (
        "parse-d0",
        "estimate",
        879,
    )
    assert (trouble.notes, trouble.notes_tokens, trouble.room) == (NOTES, 680, 651)
    assert str(trouble).startswith("r3con estimated parse-d0 at 879 tokens, over the ")


def test_a_part_bigger_than_its_notes_is_still_cut(window, logger):
    notes = ["A short note on the other documents."]
    provider = Provider()
    splits = Splits([PARAGRAPHS], model=window(500), task_logger=logger)
    splits.read_in_parts(
        0, call="parse", rest=with_notes(notes), send=provider, notes=notes
    )
    assert provider.kinds == ["parse-d0c0", "parse-d0c1"]
    events = record(logger)["documents"]["0"]["events"]
    assert [(e["cause"], e["action"]) for e in events] == [("estimate", "split")]


@pytest.mark.parametrize(
    ("note_words", "kinds"),
    [(33, ["parse-d0c0", "parse-d0c1"]), (34, [])],
    ids=["a-tie", "notes-one-token-bigger"],
)
def test_a_tie_between_part_and_notes_cuts_the_part(window, note_words, kinds):
    # The document is 34 tokens; a note of 33 words is 34 with its break.
    document, notes = " ".join(["word"] * 34), [" ".join(["word"] * note_words)]
    provider = Provider()
    splits = Splits([document], model=window(1_000))
    rest = with_notes(notes, prompt=" the" * 795)
    if kinds:
        splits.read_in_parts(0, call="parse", rest=rest, send=provider, notes=notes)
    else:
        with pytest.raises(NotesTooLong):
            splits.read_in_parts(0, call="parse", rest=rest, send=provider, notes=notes)
    assert provider.kinds == kinds


def test_with_the_rest_alone_over_the_line_the_notes_are_handed_over(window, logger):
    provider = Provider()
    splits = Splits([PARAGRAPHS], model=window(1_000), task_logger=logger)
    with pytest.raises(NotesTooLong) as handed:
        splits.read_in_parts(
            0,
            call="parse",
            rest=with_notes(NOTES, prompt=" the" * 200),
            send=provider,
            notes=NOTES,
        )
    assert (handed.value.call, handed.value.cause) == ("parse-d0", "estimate")
    assert handed.value.estimate == 200 + 1 + 680 + 720
    assert provider.sent == []
    assert splits.parts(0) == [PARAGRAPHS]
    assert record(logger) is None


def test_a_refused_part_smaller_than_its_notes_hands_them_over(logger):
    document, provider = PARAGRAPHS[:300], Provider(limit=0)
    splits = Splits([document], model=QWEN, task_logger=logger)
    with pytest.raises(NotesTooLong) as handed:
        splits.read_in_parts(
            0, call="parse", rest=with_notes(NOTES), send=provider, notes=NOTES
        )
    trouble = handed.value
    assert (trouble.call, trouble.cause, trouble.room) == ("parse-d0", "refusal", None)
    assert trouble.refusal is trouble.__cause__ is provider.refusals[0]
    assert str(trouble) == (
        "parse-d0 was refused as too long, and the notes it carries (about 680 "
        "tokens) are the bigger part of it."
    )
    assert provider.kinds == ["parse-d0"]
    assert splits.parts(0) == [document]
    assert record(logger) is None


def test_measure_cuts_by_estimate_and_raises_the_least_room(window, logger):
    documents = [PARAGRAPHS, PARAGRAPHS[:900], PARAGRAPHS[:1_500]]
    rests = {0: NOTES[:5], 1: NOTES, 2: NOTES}

    def measured(docs):
        splits = Splits(documents, model=window(1_000), task_logger=logger)
        with pytest.raises(NotesTooLong) as handed:
            splits.measure(
                docs,
                call="parse",
                rests=lambda doc: (with_notes(rests[doc]), rests[doc]),
            )
        return splits, handed.value

    splits, trouble = measured(range(3))
    assert trouble.call == "parse-d2"
    assert trouble.room < measured([1])[1].room
    assert len(splits.parts(0)) == 2
    assert (splits.parts(1), splits.parts(2)) == ([documents[1]], [documents[2]])
    assert list(record(logger)["documents"]) == ["0"]


def test_measure_does_nothing_without_a_window(logger):
    asked: list[int] = []

    def rests(doc):
        asked.append(doc)
        return with_notes(NOTES), NOTES

    splits = Splits([PARAGRAPHS[:900]], model=QWEN, task_logger=logger)
    splits.measure(range(1), call="parse", rests=rests)
    assert asked == []
    assert splits.parts(0) == [PARAGRAPHS[:900]]


OVER = [with_notes(NOTES, prompt=" the" * 200), "Input:"]
UNDER = [with_notes(NOTES), "Input:"]


@pytest.mark.parametrize(
    ("tokens", "request_texts", "notes", "handed", "counted"),
    [
        (1_000, OVER, NOTES, True, True),
        (1_000, OVER, [], False, False),
        (1_000, UNDER, NOTES, False, True),
        (1_000_000, UNDER, NOTES, False, False),
    ],
    ids=["over-with-notes", "over-without-notes", "under", "under-in-bytes"],
)
def test_a_request_without_a_document_hands_over_only_notes_over_the_line(
    window, encodes, tokens, request_texts, notes, handed, counted
):
    splits = Splits([], model=window(tokens))
    if not handed:
        splits.check_notes(call="schema", request=request_texts, notes=notes)
    else:
        with pytest.raises(NotesTooLong) as trouble:
            splits.check_notes(call="schema", request=request_texts, notes=notes)
        assert (trouble.value.call, trouble.value.cause) == ("schema", "estimate")
        assert (trouble.value.estimate, trouble.value.room) == (883, 647)
        assert str(trouble.value) == (
            "r3con estimated schema at 883 tokens, over the 850-token line (the "
            "1,000-token input window litellm's model map gives "
            "hosted_vllm/window-1000, less 15%); it was not sent."
        )
    assert bool(encodes) is counted


def test_a_refused_request_without_a_document_hands_over_its_notes():
    splits = Splits([], model=QWEN)
    refused = Provider(limit=0)
    with pytest.raises(litellm.ContextWindowExceededError):
        refused("x", "schema")
    [refusal] = refused.refusals
    with pytest.raises(NotesTooLong) as handed:
        splits.check_notes(
            call="schema", request=[with_notes(NOTES)], notes=NOTES, refusal=refusal
        )
    assert handed.value.__cause__ is handed.value.refusal is refusal
    assert (handed.value.cause, handed.value.room) == ("refusal", None)
    assert str(handed.value) == (
        "schema was refused as too long, and the notes it carries (about 680 tokens) "
        "are the bigger part of it."
    )
    splits.check_notes(call="schema", request=["Input:"], notes=[], refusal=refusal)


def test_a_notes_signal_raised_by_send_passes_through_uncut(logger):
    signal = NotesTooLong(
        "parse-d0 did not fit.",
        model=QWEN,
        call="parse-d0",
        cause="estimate",
        estimate=900,
        notes=NOTES,
        notes_tokens=680,
        room=100,
    )

    def send(part, kind):
        raise signal

    splits = Splits([PARAGRAPHS], model=QWEN, task_logger=logger)
    with pytest.raises(NotesTooLong) as raised:
        splits.read_in_parts(0, call="parse", rest=REST, send=send)
    assert raised.value is signal
    assert splits.parts(0) == [PARAGRAPHS]
    assert record(logger) is None

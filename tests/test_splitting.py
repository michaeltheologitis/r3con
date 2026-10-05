import json
import logging
from pathlib import Path

import litellm
import pytest

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
    if window_tokens:
        model = window(window_tokens)
    assert Splits(["text"], model=model).max_input_tokens == window_tokens
    assert capsys.readouterr().out == ""
    assert litellm.suppress_debug_info is False


@pytest.mark.parametrize("margin", [-1, 100])
def test_a_margin_outside_0_to_99_is_refused(margin):
    message = f"margin_percent must be from 0 to 99, got {margin}."
    with pytest.raises(ValueError, match=f"^{message}$"):
        Splits(["text"], model=QWEN, margin_percent=margin)


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

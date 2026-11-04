import unicodedata

import pytest

from r3con.quotes import FUZZY_MIN_SCORE, MIN_ELIDED_FRAGMENT_CHARS, locate_quotes

DECOMPOSED = unicodedata.normalize("NFD", "Café au lait")
HANGUL = unicodedata.normalize("NFD", "한국어 문서")
SOFT_HYPHEN = chr(0xAD)
ZERO_WIDTH_SPACE = chr(0x200B)
TEXT = (
    "Transit stands at $80 million. I propose a “leaner” budget – cut by $12 million.\n"
    "Parks   stay\tfunded — not from transit. Ｑ３ figures follow. "
    f"{DECOMPOSED}. A trans{SOFT_HYPHEN}port hub. {HANGUL}."
)
SHORT = "zz"
EARNINGS = (
    "Intro. Our first quarter revenue was $4.07 billion and earnings per share were "
    "$1.28. Revenue grew 11%."
)
MARCH = ["March (r)", "1,355", "948", "S", "393", "148", "66", "223", "129", "696"]
MARCH_ROW = [*MARCH, "530", "288", "223"]
TABLE = (
    "<table><tr><td>February (p)</td><td>1,412</td></tr><tr>"
    + "".join(f"<td>{cell}</td>" for cell in MARCH_ROW)
    + "</tr><tr><td>April (r)</td><td>1,392</td></tr></table>"
)


@pytest.mark.parametrize(
    ("quote", "located", "match"),
    [
        ("Transit stands at $80 million.", "Transit stands at $80 million.", "exact"),
        ('I propose a "leaner" budget', "I propose a “leaner” budget", "normalized"),
        ("budget - cut by $12 million", "budget – cut by $12 million", "normalized"),
        ("funded - not from transit", "funded — not from transit", "normalized"),
        (
            "TRANSIT STANDS at $80 million.",
            "Transit stands at $80 million.",
            "normalized",
        ),
        ("Parks stay funded", "Parks   stay\tfunded", "normalized"),
        ("cut by $12 million. Parks", "cut by $12 million.\nParks", "normalized"),
        ("Q3 figures follow.", "Ｑ３ figures follow.", "normalized"),
        ("café au lait", DECOMPOSED, "normalized"),
        ("한국어 문서", HANGUL, "normalized"),
        ("A transport hub.", f"A trans{SOFT_HYPHEN}port hub.", "normalized"),
        (f"Parks stay{ZERO_WIDTH_SPACE} funded", "Parks   stay\tfunded", "normalized"),
        (
            '"Transit stands at $80 million."',
            "Transit stands at $80 million.",
            "normalized",
        ),
        ("“Parks stay funded”", "Parks   stay\tfunded", "normalized"),
        (
            "Transit stands at $80 million... cut by $12 million.",
            TEXT[: TEXT.index("\n")],
            "elided",
        ),
        (
            "I propose a “leaner” budget … not from transit.",
            TEXT[TEXT.index("I propose") : TEXT.index(" Ｑ３")],
            "elided",
        ),
        (
            "Transit stands at $80 million [...] cut by $12 million.",
            TEXT[: TEXT.index("\n")],
            "elided",
        ),
        (
            "Transit stands at $80 million […] cut by $12 million.",
            TEXT[: TEXT.index("\n")],
            "elided",
        ),
        (
            "Transit stands at $80 million (...) cut by $12 million.",
            TEXT[: TEXT.index("\n")],
            "elided",
        ),
        (
            "Transit stands at $80 million . . . cut by $12 million.",
            TEXT[: TEXT.index("\n")],
            "elided",
        ),
        (
            'i propose a "leaner" budget ... parks stay funded',
            TEXT[TEXT.index("I propose") : TEXT.index(" —")],
            "elided",
        ),
        (
            f"Transit stands at $80 million … {SHORT} … Parks   stay",
            TEXT[: TEXT.index("   stay") + len("   stay")],
            "elided",
        ),
        ("Roads get $5 million.", None, None),
        ("Parks   stay\tfunded … Transit stands at $80 million", None, None),
        ("Transit … $80", None, None),
        ("Transit stands at $80 million … Roads get $5 million", None, None),
    ],
    ids=[
        "exact",
        "curly-quotes",
        "en-dash",
        "em-dash",
        "case",
        "whitespace-run",
        "line-break",
        "nfkd",
        "decomposed-letter",
        "decomposed-hangul",
        "soft-hyphen-in-the-text",
        "zero-width-space-in-the-quote",
        "straight-quotation-marks-around-it",
        "curly-quotation-marks-around-it",
        "elided-dots",
        "elided-ellipsis",
        "elided-bracketed-dots",
        "elided-bracketed-ellipsis",
        "elided-parenthesized-dots",
        "elided-spaced-dots",
        "elided-normalized-fragment",
        "elided-short-fragment-ignored",
        "not-found",
        "elided-out-of-order",
        "elided-only-short-fragments",
        "elided-fragment-not-found",
    ],
)
def test_a_quote_is_placed_in_the_original_text_by_the_first_tier_that_finds_it(
    quote, located, match
):
    [found] = locate_quotes([quote], TEXT)
    start = None if located is None else TEXT.index(located)
    end = None if start is None else start + len(located)
    assert found == {
        "text": quote,
        "start": start,
        "end": end,
        "match": match,
        "score": None,
    }


@pytest.mark.parametrize(
    ("text", "quote", "located"),
    [
        (
            "<tr><td>enrolled in credit courses</td><td>2</td></tr>",
            "enrolled in credit courses 2",
            "enrolled in credit courses</td><td>2",
        ),
        (
            "The ranks are $[34,38,55]$ in order.",
            "The ranks are [34,38,55] in order.",
            "The ranks are $[34,38,55]$ in order.",
        ),
        ("| Name | Score |\n|---|---|\n| Ann | 3 |", "Ann 3", "Ann | 3"),
        (
            "Total **revenue**: `42` today",
            "Total revenue: 42 today",
            "Total **revenue**: `42` today",
        ),
        ("Total revenue 42", "<b>Total</b> revenue | 42", "Total revenue 42"),
        ("x < 5 and y > 3", "X < 5 AND y > 3", "x < 5 and y > 3"),
    ],
    ids=[
        "html-table-row",
        "latex",
        "markdown-pipe-table",
        "markdown-emphasis-and-code",
        "markup-in-the-quote",
        "a-lone-angle-bracket-is-no-tag",
    ],
)
def test_markup_on_either_side_is_ignored_once_normalized(text, quote, located):
    [found] = locate_quotes([quote], text)
    start = text.index(located)
    assert found == {
        "text": quote,
        "start": start,
        "end": start + len(located),
        "match": "normalized",
        "score": None,
    }


@pytest.mark.parametrize(
    ("text", "quote", "located", "match", "score"),
    [
        (
            EARNINGS,
            (
                "Our first quarter revenue was $4.07 billion and earnings per share "
                "was $1.28."
            ),
            (
                "Our first quarter revenue was $4.07 billion and earnings per share "
                "were $1.28"
            ),
            "fuzzy",
            96.0,
        ),
        (
            TABLE,
            " ".join(cell for cell in MARCH_ROW if cell != "948"),
            TABLE[
                TABLE.index("March") : TABLE.index("</td><td>223</td></tr><tr><td>A")
            ],
            "fuzzy",
            92.3,
        ),
        (
            "<td>Total undergraduates</td><td>2,108</td>",
            "Total undergraduates: 4,818",
            None,
            None,
            None,
        ),
        (
            EARNINGS,
            "Our first quarter revenue was $4.07 billion",
            "Our first quarter revenue was $4.07 billion",
            "exact",
            None,
        ),
    ],
    ids=[
        "a-word-changed",
        "a-table-cell-left-out",
        "only-loosely-like-it",
        "exact-though-it-would-score-100",
    ],
)
def test_a_quote_the_model_altered_is_located_as_fuzzy_only_at_the_minimum_score(
    text, quote, located, match, score
):
    [found] = locate_quotes([quote], text)
    start = None if located is None else text.index(located)
    end = None if start is None else start + len(located)
    assert found == {
        "text": quote,
        "start": start,
        "end": end,
        "match": match,
        "score": score,
    }
    assert score is None or score >= FUZZY_MIN_SCORE


@pytest.mark.parametrize(
    ("text", "quote", "located"),
    [
        (
            (
                "The council approved the transit budget in March. The vote was then "
                "challenged in court. The council approved the transit budget again "
                "in May, after the audit."
            ),
            "The council approved the transit budget … after the audit.",
            "The council approved the transit budget again in May, after the audit.",
        ),
        (
            "Budget – cut by $12 million. Parks stay funded. Budget - cut by $12 million.",
            "Budget - cut by $12 million … Parks stay funded.",
            "Budget – cut by $12 million. Parks stay funded.",
        ),
    ],
    ids=["a-repeated-first-fragment", "a-later-exact-copy"],
)
def test_an_elided_quotes_fragments_are_placed_ending_as_early_as_they_can(
    text, quote, located
):
    [found] = locate_quotes([quote], text)
    start = text.rindex(located)
    assert (found["start"], found["end"], found["match"]) == (
        start,
        start + len(located),
        "elided",
    )


def test_a_fragment_as_long_as_the_minimum_must_be_found():
    assert len(SHORT) < MIN_ELIDED_FRAGMENT_CHARS
    long_enough = "Q" * MIN_ELIDED_FRAGMENT_CHARS
    [found] = locate_quotes([f"Transit stands … {long_enough}"], TEXT)
    assert found["match"] is None


def test_quotes_in_a_part_are_placed_in_its_whole_document():
    document = "An opening part of the document. " + TEXT
    start = len("An opening part of the document. ")
    found = locate_quotes(
        ["Transit stands", 'a "leaner" budget', "Roads get $5 million."],
        document[start:],
        start,
    )
    assert [document[q["start"] : q["end"]] for q in found[:2]] == [
        "Transit stands",
        "a “leaner” budget",
    ]
    assert [q["match"] for q in found] == ["exact", "normalized", None]


NOT_FOUND = {"start": None, "end": None, "match": None, "score": None}


@pytest.mark.parametrize(
    ("quote", "located"),
    [("", []), (" \n", []), ("<b></b>", [{"text": "<b></b>", **NOT_FOUND}])],
    ids=["empty", "blank", "markup-only"],
)
def test_a_quote_with_nothing_to_find_is_dropped_or_not_found(quote, located):
    assert locate_quotes([quote], TEXT) == located


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        ("<br>", "a quote longer than the text it is looked for in"),
        ("<tr><td></td></tr>", "a quote longer than the text it is looked for in"),
        ("| | |", "x y z"),
        ("$**$", "a quote of words, in a text of markup"),
        (
            "The contractor is Acme.",
            "The contractor is Acme. Payment was due in March and was paid late.",
        ),
    ],
    ids=[
        "a-tag-only",
        "an-empty-table-row",
        "pipes-only",
        "markup-characters-only",
        "a-quote-longer-than-its-text-that-holds-it-all",
    ],
)
def test_a_text_with_nothing_like_the_quote_gives_no_fuzzy_match(text, quote):
    assert locate_quotes([quote], text) == [{"text": quote, **NOT_FOUND}]

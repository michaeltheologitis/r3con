import logging

from r3con.runtime.llm import schema_as_sent
from r3con.stages.citing import CitingRequest, citing_schema, read_citing

MODEL = "openai/gpt-6-luna"
SHOWN = ["a1", "b2", "c3"]


def read(request, completion):
    return read_citing(
        request,
        "The document.",
        rewrite="Read it again",
        kind="test",
        max_attempts=1,
        model=MODEL,
        completion=completion,
    )


def test_the_payload_comes_first_in_order_then_the_ids_it_cites():
    reply = citing_schema("Reply", SHOWN, note=(str, ...), page=(int, ...))
    schema = schema_as_sent(reply)
    assert list(schema["properties"]) == ["note", "page", "provenance_ids"]
    assert schema["properties"]["provenance_ids"]["items"]["enum"] == SHOWN


def test_unshown_and_repeated_ids_are_dropped_and_the_rest_kept_in_order(llm, caplog):
    request = CitingRequest(
        "Prompt.", citing_schema("Reply", SHOWN, note=(str, ...)), SHOWN
    )
    reply = {"note": "A note.", "provenance_ids": ["c3", "zz", "a1", "c3", "zz"]}
    with caplog.at_level(logging.WARNING, logger="r3con.citing"):
        read_reply, provenance = read(request, llm.replies(reply))
    assert (read_reply.note, provenance) == ("A note.", ["c3", "a1"])
    assert caplog.messages == [
        "test: dropped provenance id(s) the call did not show: ['zz', 'zz']"
    ]


def test_a_reply_with_nothing_to_cite_has_no_ids_and_cites_nothing(llm):
    request = CitingRequest(
        "Prompt.", citing_schema("Reply", None, note=(str, ...)), []
    )
    read_reply, provenance = read(request, llm.replies({"note": "A note."}))
    assert (read_reply.note, provenance) == ("A note.", [])
    sent = llm.requests[0]["response_format"]["json_schema"]["schema"]
    assert list(sent["properties"]) == ["note"]

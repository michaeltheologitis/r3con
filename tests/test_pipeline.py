import ast
import datetime
import json
import logging
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

import litellm
import pytest
import yaml
from pydantic import ValidationError

from r3con import read_documents, settings
from r3con.config import (
    PROMPT_STAGES,
    REASONING_PATHS,
    RunConfig,
    load_config,
    normalize_model_name,
)
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger
from r3con.stages.structuring.parsing import SchemaError

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
DOCS = ["Halloran memo", "Merrow memo"]
LATER_STAGES = ("schema", "parsing", "reasoning")
NEVER_COMMITS = "<code>\nprint(1)\n</code>"
COMMITS_THE_PARSE = "<code>\nfinal_answer(parse)\n</code>"
# A commit as the coding v2 prompt asks for it, citing documents in a flat list.
COMMITS_IN_THE_V2_STYLE = (
    '<code>\nfinal_answer("; ".join(r["who"] for r in parse["rows"]), '
    'provenance=[r["document"] for r in parse["rows"]])\n</code>'
)
STAGE_FOLDERS = {
    "relevance": "relevance",
    "schema": "structuring/schema",
    "parsing": "structuring/parsing",
    "reasoning": "reasoning",
}
CONFIG = {
    "name": "test",
    "model": MODEL,
    "relevance_rounds": 2,
    "prompts": {
        **dict.fromkeys(PROMPT_STAGES, "v1"),
        "relevance/later-rounds": "v2",
        "structuring/parsing": "v3",
        "reasoning/direct": "v3",
        "reasoning/coding": "v3",
    },
}
READ_IN_PARTS = "Some documents were too long to read whole and were read in parts"
REGISTRY_IN_4_PARTS = ["1", "2", "3", "4.1", "4.2", "4.3", "4.4", "5"]
ROUTINE = "The site logged its routine checks and found nothing out of the ordinary. "
SAMPLE = "only a SAMPLE"
LINE_8192 = 6_963
SUMMARY_ID_LINE = re.compile(r"^### Summary ID: \S+$", re.MULTILINE)
DOCUMENT_ID_LINE = re.compile(r"^(### )?Document ID: \S+$", re.MULTILINE)


def answer(
    llm,
    task_logger=None,
    *,
    documents=DOCS,
    max_reasoning_turns=None,
    on_event=None,
    **fields,
):
    """A whole run over ``documents`` on CONFIG with ``fields`` changed, ``llm``
    answering."""
    return run_pipeline(
        task="Who?",
        documents=documents,
        config=RunConfig(**{**CONFIG, **fields}),
        max_reasoning_turns=max_reasoning_turns,
        task_logger=task_logger,
        completion=llm,
        on_event=on_event,
    )


def provider_down() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(
        "provider down", llm_provider="openai", model=MODEL
    )


def read_json(logger, name: str):
    return json.loads((logger.dir / f"{name}.json").read_text())


def node(event) -> tuple:
    """What an event starts or finishes: its stage, and the round and document it
    names, if any."""
    stage = event["type"].rpartition("_")[0]
    return stage, event.get("round"), event.get("doc_id")


def snippet_id(event) -> str:
    """The id of the snippet a ``relevance_finished`` event reports."""
    part = event["part"]
    return event["doc_id"] if part is None else f"{event['doc_id']}.{part}"


def notes_by_round():
    reads = Counter()

    def note(request) -> str:
        document = request["messages"][1]["content"]
        reads[document] += 1
        return f"Round {reads[document]} notes on {document}."

    return note


def system_prompts(llm, stage) -> list[str]:
    return [request["messages"][0]["content"] for request in llm.requests_for(stage)]


def memos_with(registry: str) -> list[str]:
    """The five memos, with the contractor registry (documents[3]) replaced."""
    documents = read_documents(MEMOS)
    documents[3] = registry
    return documents


def citable_lists(request) -> dict:
    """The JSON schema of each list of ids a direct reasoning reply may cite."""
    schema = request["response_format"]["json_schema"]["schema"]
    return schema["$defs"]["ProvenanceIds"]["properties"]


def summary_headings(llm) -> list[str]:
    """The ids heading the reasoning prompt's summaries, without its examples'."""
    [prompt, *_] = system_prompts(llm, "reasoning")
    section = prompt.split("## Document summaries", 1)[1].split("## The parsed", 1)[0]
    return re.findall(r"^### Document ID: (\S+)$", section, re.MULTILINE)


def as_ids(ids: list[str], labels: list[str]) -> list[str]:
    """``labels`` such as ``"4.1"``, document 4's part 1, with each document's number
    replaced by its id."""
    return [re.sub(r"^\d+", lambda n: ids[int(n[0]) - 1], label) for label in labels]


def stamped(ids: list[str]) -> dict:
    """DOCS' parse, each row stamped with its document's id."""
    return {
        "rows": [
            {"document": id_, "who": who} for id_, who in zip(ids, DOCS, strict=True)
        ]
    }


def chars(request) -> int:
    return sum(len(message["content"]) for message in request["messages"])


def tokens(request) -> int:
    return sum(len(litellm.encode(text=m["content"])) for m in request["messages"])


def users(llm) -> list[str]:
    return [request["messages"][1]["content"] for request in llm.requests]


def long_notes_for(registry: str, size: int):
    """A relevance reply: ``size`` characters for a part of ``registry``, and the
    first line for any other document."""
    note = "A long note on one part of the contractor registry. " * (size // 50 + 1)

    def reply(request) -> str:
        document = request["messages"][1]["content"]
        return note[:size] if document in registry else f"Notes on {document[:20]}."

    return reply


def memo_note(request) -> str:
    """A relevance reply of about 200 characters, 33 words, that names the document by
    its first line, as a note on one of the memos does."""
    first = request["messages"][1]["content"].splitlines()[0]
    note = (
        f"{first}: an operations document; it reports what each site logged and the "
        "contractor code its maintenance runs under, which the registry maps to a name. "
    ) * 4
    return note[:200].rsplit(" ", 1)[0] + "."


def parsed(request) -> str:
    """The document, or part, a parse request sends below its ``Document ID`` line."""
    return request["messages"][1]["content"].split("\n\n", 1)[1]


def whole_document_row(request) -> dict:
    """A parse reply whose one row is the whole document, so the parse fills the
    reasoning prompt."""
    return {"rows": [{"who": parsed(request)}]}


def memos_run(llm, documents, model, tmp_path, **overrides):
    """A run of the default config over ``documents`` on ``model`` with ``overrides``,
    ``llm`` answering."""
    logger = TaskLogger("run", root=tmp_path)
    result = run_pipeline(
        task="Who?",
        documents=documents,
        config=load_config("default", model=model, **overrides),
        task_logger=logger,
        completion=llm,
    )
    return logger, result


@pytest.fixture
def priced_model():
    """MODEL at round test rates: 10 tokens in and 20 out cost $10 a call. litellm
    caches each model's price by name, so the cache is cleared once the rates are set
    and again once they are undone."""
    name = normalize_model_name(MODEL)
    rates = {"input_cost_per_token": 0.5, "output_cost_per_token": 0.25}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            litellm.model_cost, name, {**litellm.model_cost.get(name, {}), **rates}
        )
        litellm.get_model_info.cache_clear()
        yield
    litellm.get_model_info.cache_clear()


def counting_ids(monkeypatch) -> None:
    """Document ids drawn as 000000, 000001, and so on, the same in every run."""
    monkeypatch.setattr(random, "sample", lambda population, k: list(population[:k]))


def citing_every_shown_id(shown_ids):
    """A relevance reply: a note naming the document, and, from round 2 on, every id
    the call shows, last first."""

    def reply(request):
        note = f"Notes on {request['messages'][1]['content'][:20]}."
        if "response_format" not in request:
            return note
        return {"summary": note, "provenance_ids": shown_ids(request)[::-1]}

    return reply


def citing_its_own_note(request) -> dict:
    """A parse reply: the document as its one row, citing the id heading its user
    message, its own."""
    user = request["messages"][1]["content"]
    own = re.match(r"Document ID: (\w+)\n", user)[1]
    return {"parse": {"rows": [{"who": parsed(request)}]}, "provenance_ids": [own]}


def first_turns(llm) -> list[dict]:
    return [r for r in llm.requests_for("reasoning") if len(r["messages"]) == 2]


@pytest.fixture
def shows_the_notes_and_the_stamped_parse(reasoning_sections, parse_shown):
    """Checks a system message is ``path``'s reasoning prompt over DOCS' notes, each
    under its document's id, and their whole parse, each row stamped with its
    document's id."""

    def check(message: dict, path: str, ids: list[str]) -> None:
        assert message["role"] == "system"
        shown = reasoning_sections(message["content"], path)
        assert READ_IN_PARTS not in shown["## Document summaries"]
        assert shown["## Document summaries"].endswith(
            f"### Document ID: {ids[0]}\nNotes on Halloran memo.\n\n"
            f"### Document ID: {ids[1]}\nNotes on Merrow memo."
        )
        assert parse_shown(shown) == stamped(ids)

    return check


def test_a_run_answers_and_leaves_every_stages_artifacts(
    answering_llm, priced_model, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    result = answer(answering_llm, logger)
    assert str(result) == result.answer == "Halloran memo; Merrow memo"
    assert result.relevant_context == [
        "Notes on Halloran memo.",
        "Notes on Merrow memo.",
    ]
    assert result.structured_context == stamped(result.document_ids)
    assert "class Parse(BaseModel)" in result.schema_code
    assert result.source_docs == {"rows": [0, 1]}
    assert result.run_dir == logger.dir
    written = {str(p.relative_to(logger.dir)) for p in logger.dir.rglob("*.*")}
    assert written == {
        "manifest.json",
        "ids.json",
        "events.jsonl",
        "relevance/calls.json",
        "relevance/result.json",
        "structuring/schema/calls.json",
        "structuring/schema/result.json",
        "structuring/schema/transcript.yaml",
        "structuring/parsing/calls.json",
        "structuring/parsing/result.json",
        "reasoning/calls.json",
        "reasoning/result.json",
        "reasoning/transcript.yaml",
    }
    relevance = json.loads((logger.dir / "relevance/result.json").read_text())
    assert (relevance["n_rounds"], relevance["n_docs"]) == (2, 2)
    assert [r["round"] for r in relevance["rounds"]] == [1, 2]
    assert relevance["totals"] == {
        "n_steps": 4,
        "tokens": {"prompt": 40, "completion": 80, "total": 120},
        "usd": 40.0,
    }


def test_the_run_folder_and_the_answer_record_which_summaries_each_one_used(
    answering_llm, shown_ids, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    documents = [*DOCS, "Calder memo"]
    llm = answering_llm.answers(relevance=citing_every_shown_id(shown_ids))
    result = answer(llm, logger, documents=documents)
    ids = result.document_ids
    assert read_json(logger, "ids") == {ids[0]: 0, ids[1]: 1, ids[2]: 2}
    rounds = read_json(logger, "relevance/result")["rounds"]
    notes = [f"Notes on {document}." for document in documents]
    assert rounds == [
        {
            "round": 1,
            "snippets": [
                {"id": id_, "summary": note, "provenance": []}
                for id_, note in zip(ids, notes, strict=True)
            ],
        },
        {
            "round": 2,
            "snippets": [
                {"id": ids[0], "summary": notes[0], "provenance": [ids[2], ids[1]]},
                {"id": ids[1], "summary": notes[1], "provenance": [ids[2], ids[0]]},
                {"id": ids[2], "summary": notes[2], "provenance": [ids[1], ids[0]]},
            ],
        },
    ]
    assert result.relevance_provenance == [
        [ids[2], ids[1]],
        [ids[2], ids[0]],
        [ids[1], ids[0]],
    ]


def test_the_run_folder_and_the_answer_record_which_notes_each_parse_used(
    answering_llm, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    documents = [*DOCS, "Calder memo"]
    llm = answering_llm.answers(parsing=citing_its_own_note)
    result = answer(llm, logger, documents=documents)
    ids = result.document_ids
    assert result.parsing_provenance == [[ids[0]], [ids[1]], [ids[2]]]
    parsing = read_json(logger, "structuring/parsing/result")
    assert parsing["provenance"] == result.parsing_provenance
    assert [row["who"] for row in parsing["parsed"]["rows"]] == documents


@pytest.mark.parametrize(
    ("path", "reply", "cites_the_parse"),
    [("direct", None, True), ("coding", COMMITS_IN_THE_V2_STYLE, False)],
)
def test_a_config_that_pins_reasoning_v2_still_answers(
    answering_llm, tmp_path, path, reply, cites_the_parse
):
    logger = TaskLogger("run", root=tmp_path)
    prompts = {**CONFIG["prompts"], "reasoning/direct": "v2", "reasoning/coding": "v2"}
    llm = answering_llm.answers(reasoning=reply) if reply else answering_llm
    result = answer(llm, logger, prompts=prompts, reasoning=path)
    assert result.answer == "Halloran memo; Merrow memo"
    [prompt] = system_prompts(llm, "reasoning")
    assert "- `provenance`: The IDs of the documents above" in prompt
    parse_ids = result.document_ids if cites_the_parse else []
    assert result.provenance == {"summary_ids": [], "parse_ids": parse_ids}
    assert read_json(logger, "reasoning/result")["provenance"] == result.provenance


def test_a_config_that_pins_parsing_v1_still_runs(answering_llm, tmp_path):
    logger = TaskLogger("run", root=tmp_path)
    prompts = {**CONFIG["prompts"], "structuring/parsing": "v1"}
    result = answer(answering_llm, logger, prompts=prompts)
    assert result.answer == "Halloran memo; Merrow memo"
    assert result.parsing_provenance == [[], []]
    for prompt in system_prompts(answering_llm, "parsing"):
        assert "## Output" not in prompt
        assert "provenance" not in prompt
    manifest = read_json(logger, "manifest")["config"]
    assert manifest["prompts"]["structuring/parsing"] == "v1"


def test_a_config_that_pins_the_prompts_before_provenance_ids_still_records_them(
    answering_llm, shown_ids, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    prompts = {
        **CONFIG["prompts"],
        "relevance/later-rounds": "v1",
        "structuring/parsing": "v2",
    }
    llm = answering_llm.answers(
        relevance=citing_every_shown_id(shown_ids), parsing=citing_its_own_note
    )
    result = answer(llm, logger, prompts=prompts)
    ids = result.document_ids
    assert result.relevance_provenance == [[ids[1]], [ids[0]]]
    assert result.parsing_provenance == [[ids[0]], [ids[1]]]
    assert "`provenance`:" in system_prompts(llm, "parsing")[0]
    manifest = read_json(logger, "manifest")["config"]
    assert manifest["prompts"]["relevance/later-rounds"] == "v1"


def test_every_later_stage_reads_the_final_rounds_notes_only(answering_llm):
    answer(answering_llm.answers(relevance=notes_by_round()))
    for stage in LATER_STAGES:
        for prompt in system_prompts(answering_llm, stage):
            assert "Round 2 notes on Halloran memo." in prompt
            assert "Round 2 notes on Merrow memo." in prompt
            assert "Round 1 notes" not in prompt


def test_the_default_config_answers_in_one_call_over_the_whole_stamped_parse(
    answering_llm, shows_the_notes_and_the_stamped_parse, tmp_path
):
    def cites_the_first_summary_and_the_second_parse(request):
        lists = citable_lists(request)
        return {
            "answer": "Halloran and Merrow.",
            "provenance_ids": {
                "summary_ids": lists["summary_ids"]["items"]["enum"][:1],
                "parse_ids": lists["parse_ids"]["items"]["enum"][1:],
            },
        }

    llm = answering_llm.answers(reasoning=cites_the_first_summary_and_the_second_parse)
    logger, result = memos_run(llm, DOCS, MODEL, tmp_path)
    ids = result.document_ids
    [request] = llm.requests_for("reasoning")
    system, user = request["messages"]
    shows_the_notes_and_the_stamped_parse(system, "direct", ids)
    assert user == {"role": "user", "content": "Input:\n<task>\nWho?\n</task>"}
    cited = {"summary_ids": [ids[0]], "parse_ids": [ids[1]]}
    assert (result.answer, result.provenance) == ("Halloran and Merrow.", cited)
    reasoning = read_json(logger, "reasoning/result")
    assert (reasoning["answer"], reasoning["provenance"]) == (
        "Halloran and Merrow.",
        cited,
    )
    assert (reasoning["path"], reasoning["fallback"]) == ("direct", None)
    assert reasoning["totals"]["n_steps"] == 1
    transcript = yaml.safe_load((logger.dir / "reasoning/transcript.yaml").read_text())
    roles = [message["role"] for message in transcript["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_the_coding_path_sends_the_coding_prompt_over_the_whole_stamped_parse(
    answering_llm, shows_the_notes_and_the_stamped_parse, tmp_path
):
    llm = answering_llm.answers(reasoning=COMMITS_THE_PARSE)
    logger = TaskLogger("run", root=tmp_path)
    result = answer(llm, logger, reasoning="coding")
    ids = result.document_ids
    [request] = llm.requests_for("reasoning")
    system, user = request["messages"]
    shows_the_notes_and_the_stamped_parse(system, "coding", ids)
    assert user == {"role": "user", "content": "Input:\n<task>\nWho?\n</task>"}
    assert ast.literal_eval(result.answer) == stamped(ids)
    assert result.provenance == {"summary_ids": [], "parse_ids": []}
    reasoning = read_json(logger, "reasoning/result")
    assert (reasoning["path"], reasoning["fallback"]) == ("coding", None)
    assert (reasoning["terminated_by"], reasoning["n_turns"]) == ("final_answer", 1)
    assert (logger.dir / "reasoning" / "transcript.yaml").is_file()


def test_a_direct_prompt_the_provider_refuses_reasons_with_code_and_the_folder_says_why(
    answering_llm, tmp_path, refusal, shows_the_notes_and_the_stamped_parse
):
    refused = refusal()
    llm = answering_llm.answers(reasoning=[refused, COMMITS_THE_PARSE])
    logger = TaskLogger("run", root=tmp_path)
    result = answer(llm, logger)
    ids = result.document_ids
    assert ast.literal_eval(result.answer) == stamped(ids)
    direct, coding = (r["messages"][0] for r in llm.requests_for("reasoning"))
    shows_the_notes_and_the_stamped_parse(direct, "direct", ids)
    assert SAMPLE in coding["content"] and "final_answer(" in coding["content"]
    reasoning = read_json(logger, "reasoning/result")
    assert reasoning["path"] == "coding"
    assert reasoning["fallback"] == {"cause": "refusal", "error": str(refused)}
    assert (logger.dir / "reasoning" / "transcript.yaml").is_file()


def test_direct_replies_that_never_fit_reason_with_code_and_the_folder_says_why(
    answering_llm, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "REASONING_MAX_ATTEMPTS", 2)
    no_answer = {"provenance_ids": {"summary_ids": [], "parse_ids": []}}
    llm = answering_llm.answers(reasoning=[no_answer, no_answer, COMMITS_THE_PARSE])
    logger = TaskLogger("run", root=tmp_path)
    result = answer(llm, logger)
    assert ast.literal_eval(result.answer) == stamped(result.document_ids)
    reasoning = read_json(logger, "reasoning/result")
    assert reasoning["path"] == "coding"
    assert reasoning["fallback"] == {
        "cause": "invalid",
        "error": "answer: Field required",
    }
    kinds = [call["kind"] for call in read_json(logger, "reasoning/calls")]
    assert kinds == ["direct", "direct-retry-1", "turn-1"]
    assert read_json(logger, "manifest")["settings"]["reasoning_max_attempts"] == 2


def test_a_fallback_that_fails_after_a_turn_leaves_that_turn_in_its_transcript(
    answering_llm, tmp_path, refusal
):
    llm = answering_llm.answers(reasoning=[refusal(), NEVER_COMMITS, provider_down()])
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(litellm.APIConnectionError):
        answer(llm, logger)
    calls = read_json(logger, "reasoning/calls")
    assert [(call["kind"], call["error"] is None) for call in calls] == [
        ("direct", False),
        ("turn-1", True),
        ("turn-2", False),
    ]
    assert (logger.dir / "reasoning" / "error.txt").is_file()
    transcript = yaml.safe_load((logger.dir / "reasoning/transcript.yaml").read_text())
    assert transcript["messages"][-1]["content"] == NEVER_COMMITS


def test_document_ids_reach_relevance_and_parsing_only_as_headings_never_the_schema(
    answering_llm, shown_ids, monkeypatch
):
    counting_ids(monkeypatch)
    result = answer(answering_llm, documents=[*DOCS, "Calder memo"])
    ids = result.document_ids

    def ids_in(text: str) -> set[str]:
        return {id_ for id_ in ids if id_ in text}

    relevance = answering_llm.requests_for("relevance")
    later_rounds = [request for request in relevance if "response_format" in request]
    assert [len(shown_ids(request)) for request in later_rounds] == [2, 2, 2]
    for request in relevance:
        text = "\n".join(message["content"] for message in request["messages"])
        assert ids_in(SUMMARY_ID_LINE.sub("", text)) == set()
    for request in later_rounds:
        schema = request["response_format"]["json_schema"]["schema"]
        enum = schema["properties"]["provenance_ids"]["items"]["enum"]
        assert enum == shown_ids(request)
    for request in answering_llm.requests_for("parsing"):
        text = "\n".join(message["content"] for message in request["messages"])
        shown = text.split("## Your job", 1)[0]
        assert re.findall(r"^### Document ID: (\S+)$", shown, re.MULTILINE) == ids
        assert ids_in(DOCUMENT_ID_LINE.sub("", text)) == set()
        schema = request["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["provenance_ids"]["items"]["enum"] == ids
    for request in answering_llm.requests_for("schema"):
        assert ids_in(json.dumps(request, default=str)) == set()


def test_the_schema_names_each_document_by_number_and_reasoning_by_id(
    answering_llm,
):
    result = answer(answering_llm)
    ids = result.document_ids
    for prompt in system_prompts(answering_llm, "schema"):
        assert "### Document 1\nNotes on Halloran memo." in prompt
        assert "### Document 2\nNotes on Merrow memo." in prompt
        assert "Document ID" not in prompt
    [request] = answering_llm.requests_for("reasoning")
    prompt = request["messages"][0]["content"]
    assert summary_headings(answering_llm) == ids
    assert f"### Document ID: {ids[0]}\nNotes on Halloran memo." in prompt
    assert f"### Document ID: {ids[1]}\nNotes on Merrow memo." in prompt
    assert re.search(r"Document \d|\"document\": \d", prompt) is None
    lists = citable_lists(request)
    assert [lists[key]["items"]["enum"] for key in lists] == [ids, ids]


def test_every_record_carries_its_documents_id_a_split_documents_parts_included(
    answering_llm, grown_registry, tmp_path
):
    documents = memos_with(grown_registry(40_000))

    def rows(request):
        document = parsed(request)
        return {"rows": [] if document == documents[1] else [{"who": document[:20]}]}

    logger = TaskLogger("run", root=tmp_path)
    llm = answering_llm.answers(parsing=rows).refuses_over(18_000)
    result = answer(llm, logger, documents=documents, model=QWEN)
    ids = result.document_ids
    assert result.source_docs == {"rows": [0, 2, 3, 3, 3, 3, 4]}
    assert result.source_ids == {"rows": [ids[0], ids[2], *[ids[3]] * 4, ids[4]]}
    stamps = [row["document"] for row in result.structured_context["rows"]]
    assert stamps == result.source_ids["rows"]
    parsing = read_json(logger, "structuring/parsing/result")
    assert parsing["source_docs"] == result.source_docs
    assert parsing["source_ids"] == result.source_ids


@pytest.mark.parametrize("rounds", [1, 3])
def test_each_document_is_read_once_per_configured_round(answering_llm, rounds):
    answer(answering_llm, relevance_rounds=rounds)
    assert len(answering_llm.requests_for("relevance")) == rounds * len(DOCS)


def test_the_configs_params_are_in_every_request(answering_llm):
    params = {"temperature": 0.7, "extra_body": {"top_k": 20}}
    answer(answering_llm, model="openai/gpt-4o", params=params)
    for request in answering_llm.requests:
        assert {key: request[key] for key in params} == params


def test_no_request_carries_a_seed_unless_params_sets_one(answering_llm):
    answer(answering_llm)
    assert not any("seed" in request for request in answering_llm.requests)
    answering_llm.requests.clear()
    answer(answering_llm, model="openai/gpt-4o", params={"seed": 7})
    assert [request["seed"] for request in answering_llm.requests] == [7] * 8


@pytest.mark.parametrize("reasoning", REASONING_PATHS)
def test_each_stage_renders_the_prompt_version_its_config_pins(
    answering_llm, tmp_path, reasoning
):
    versions = dict(
        zip(PROMPT_STAGES, ["vA", "vB", "vC", "vD", "vE", "vF"], strict=True)
    )
    for stage, version in versions.items():
        prompt = yaml.safe_load((settings.PROMPTS_DIR / stage / "v1.yaml").read_text())
        prompt["instructions"] += f"\nMARKER-{version}"
        path = tmp_path / "prompts" / stage / f"{version}.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(prompt))
    answer(answering_llm, prompts=versions, reasoning=reasoning)
    relevance = system_prompts(answering_llm, "relevance")
    later = ["vC", "vD", versions[f"reasoning/{reasoning}"]]
    sent = {
        "vA": relevance[: len(DOCS)],
        "vB": relevance[len(DOCS) :],
        **{
            version: system_prompts(answering_llm, stage)
            for version, stage in zip(later, LATER_STAGES, strict=True)
        },
    }
    for version, prompts in sent.items():
        assert prompts
        for prompt in prompts:
            assert set(re.findall(r"MARKER-v\w", prompt)) == {f"MARKER-{version}"}


@pytest.mark.parametrize(
    ("prompts", "error"),
    [
        ({**CONFIG["prompts"], "reasoning/coding": "v9"}, FileNotFoundError),
        ({"reasoning/direct": "v1"}, ValueError),
    ],
    ids=["missing-file", "missing-stage"],
)
def test_a_config_that_cannot_render_its_prompts_is_refused_before_any_request(
    answering_llm, tmp_path, prompts, error
):
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(error):
        answer(answering_llm, logger, prompts=prompts)
    assert answering_llm.requests == []
    assert not (logger.dir / "manifest.json").exists()


@pytest.mark.parametrize("stage", list(STAGE_FOLDERS))
def test_a_failing_stage_leaves_its_calls_and_traceback_and_names_the_run_folder(
    answering_llm, tmp_path, stage
):
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(litellm.APIConnectionError, match="provider down") as failure:
        answer(answering_llm.answers(**{stage: provider_down()}), logger)
    assert failure.value.__notes__ == [f"r3con: partial artifacts in {logger.dir}"]
    folder = logger.dir / STAGE_FOLDERS[stage]
    error = (folder / "error.txt").read_text()
    assert "APIConnectionError: litellm.APIConnectionError: provider down" in error
    assert (folder / "calls.json").is_file()
    assert not (folder / "result.json").exists()
    earlier = list(STAGE_FOLDERS.values())[: list(STAGE_FOLDERS).index(stage)]
    assert all((logger.dir / name / "result.json").is_file() for name in earlier)


def test_a_stage_that_fails_midway_keeps_the_calls_made_before_it(
    answering_llm, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    down = provider_down()
    with pytest.raises(litellm.APIConnectionError):
        answer(answering_llm.answers(relevance=["one", "two", down, down]), logger)
    calls = read_json(logger, "relevance/calls")
    down_in_brief = "APIConnectionError: litellm.APIConnectionError: provider down"
    assert sorted((call["kind"], call["error"]) for call in calls) == [
        ("relevance-r1-d0", None),
        ("relevance-r1-d1", None),
        ("relevance-r2-d0", down_in_brief),
        ("relevance-r2-d1", down_in_brief),
    ]


def test_an_interrupted_stage_is_recorded_like_a_failure(answering_llm, tmp_path):
    logger = TaskLogger("run", root=tmp_path)
    answering_llm.answers(relevance=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        answer(answering_llm, logger, documents=DOCS[:1])
    assert "KeyboardInterrupt" in (logger.dir / "relevance" / "error.txt").read_text()


def test_without_a_logger_a_failure_carries_no_note(answering_llm, tmp_path):
    with pytest.raises(litellm.APIConnectionError) as failure:
        answer(answering_llm.answers(schema=provider_down()))
    assert not hasattr(failure.value, "__notes__")
    assert list(tmp_path.iterdir()) == []


def test_without_a_logger_nothing_is_written(answering_llm, tmp_path):
    result = answer(answering_llm)
    assert result.answer == "Halloran memo; Merrow memo"
    assert result.run_dir is None
    assert list(tmp_path.iterdir()) == []


def test_every_request_of_a_run_goes_through_the_callers_completion(
    answering_llm, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    answer(answering_llm, logger, model=QWEN)
    assert len(answering_llm.requests) == 8
    assert {r["model"] for r in answering_llm.requests} == {QWEN}
    assert "completion" not in (logger.dir / "manifest.json").read_text()


@pytest.mark.parametrize(
    ("setting", "explicit"), [(2, None), (5, 2)], ids=["in-settings", "explicit"]
)
def test_a_turn_cap_bounds_the_run_and_is_recorded(
    answering_llm, tmp_path, monkeypatch, setting, explicit
):
    monkeypatch.setattr(settings, "REASONING_MAX_TURNS", setting)
    logger = TaskLogger("run", root=tmp_path)
    answering_llm.answers(reasoning=NEVER_COMMITS)
    answer(answering_llm, logger, max_reasoning_turns=explicit, reasoning="coding")
    assert len(answering_llm.requests_for("reasoning")) == 3
    assert read_json(logger, "reasoning/result")["n_turns"] == 2
    assert read_json(logger, "manifest")["settings"]["reasoning_max_turns"] == 2


@pytest.mark.parametrize(
    ("cap", "stage", "reply", "error"),
    [
        (
            "SCHEMA_MAX_ATTEMPTS",
            "schema",
            "class Foo(BaseModel):\n    x: int",
            pytest.raises(SchemaError, match="exhausted 2 attempts"),
        ),
        (
            "PARSING_MAX_ATTEMPTS",
            "parsing",
            {"rows": [{"who": 7}]},
            pytest.raises(ValidationError, match="who"),
        ),
    ],
)
def test_an_attempt_cap_set_in_settings_bounds_its_stage(
    answering_llm, monkeypatch, cap, stage, reply, error
):
    monkeypatch.setattr(settings, cap, 2)
    with error:
        answer(answering_llm.answers(**{stage: reply}), documents=DOCS[:1])
    assert len(answering_llm.requests_for(stage)) == 2


def test_the_manifest_records_the_margin(answering_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WINDOW_MARGIN_PERCENT", 20)
    logger = TaskLogger("run", root=tmp_path)
    answer(answering_llm, logger)
    assert read_json(logger, "manifest")["settings"]["window_margin_percent"] == 20


def test_a_cap_below_its_floor_sends_and_writes_nothing(llm, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REASONING_MAX_TURNS", 0)
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(ValueError, match="REASONING_MAX_TURNS must be >= 1, got 0."):
        answer(llm, logger)
    assert llm.requests == []
    assert not (logger.dir / "manifest.json").exists()


def test_a_refused_document_is_read_in_parts_and_keeps_its_number(
    answering_llm, grown_registry, tmp_path
):
    registry = grown_registry(40_000)
    logger = TaskLogger("run", root=tmp_path)
    llm = answering_llm.refuses_over(18_000)
    result = answer(llm, logger, documents=memos_with(registry), model=QWEN)
    assert len(result.relevant_context) == 5
    assert summary_headings(llm) == as_ids(result.document_ids, REGISTRY_IN_4_PARTS)
    assert READ_IN_PARTS in system_prompts(llm, "reasoning")[0]
    assert result.source_docs == {"rows": [0, 1, 2, 3, 3, 3, 3, 4]}
    stamps = [row["document"] for row in result.structured_context["rows"]]
    assert stamps == as_ids(
        result.document_ids, ["1", "2", "3", "4", "4", "4", "4", "5"]
    )
    snippets = read_json(logger, "relevance/result")["rounds"][0]["snippets"]
    part_notes = [s["summary"] for s in snippets[3:7]]
    assert [s["id"] for s in snippets[3:7]] == [
        f"{result.document_ids[3]}.{k}" for k in range(1, 5)
    ]
    for request in llm.requests_for("relevance"):
        part = request["messages"][1]["content"]
        if part in registry and part != registry:
            system = request["messages"][0]["content"]
            assert not any(note in system for note in part_notes)
    over = [request for request in llm.requests if chars(request) > 18_000]
    assert len(over) == len({json.dumps(r["messages"]) for r in over}) == 2
    record = read_json(logger, "splits")["documents"]["3"]
    assert len(record["cuts"]) == 3
    assert [(e["cause"], e["action"]) for e in record["events"]] == [
        ("refusal", "split"),
        ("refusal", "split"),
    ]
    assert record["parts"] == {"relevance-r1": 4, "relevance-r2": 4, "parse": 4}


def test_identical_documents_get_distinct_ids_and_each_sees_the_others(
    answering_llm, shown_ids, monkeypatch
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    result = answer(answering_llm, documents=["Same memo"] * 3)
    ids = result.document_ids
    assert len(set(ids)) == 3
    round_2 = [
        r for r in answering_llm.requests_for("relevance") if "response_format" in r
    ]
    assert [set(shown_ids(request)) for request in round_2] == [
        set(ids) - {id_} for id_ in ids
    ]


def test_a_relevance_attempt_cap_set_in_settings_bounds_its_resends(
    answering_llm, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "RELEVANCE_MAX_ATTEMPTS", 2)
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")

    def reply(request):
        return {"provenance_ids": []} if "response_format" in request else "A snippet."

    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(ValidationError, match="summary"):
        answer(answering_llm.answers(relevance=reply), logger)
    assert len(answering_llm.requests_for("relevance")) == len(DOCS) + 2
    assert read_json(logger, "manifest")["settings"]["relevance_max_attempts"] == 2


def test_a_split_documents_provenance_joins_its_parts_without_repeats(
    answering_llm, grown_registry, shown_ids
):
    llm = answering_llm.answers(relevance=citing_every_shown_id(shown_ids))
    documents = memos_with(grown_registry(40_000))
    result = answer(llm.refuses_over(16_000), documents=documents, model=QWEN)
    ids = result.document_ids
    registry_parts = [f"{ids[3]}.{k}" for k in (4, 3, 2, 1)]
    assert result.relevance_provenance == [
        [ids[4], *registry_parts, ids[2], ids[1]],
        [ids[4], *registry_parts, ids[2], ids[0]],
        [ids[4], *registry_parts, ids[1], ids[0]],
        [ids[4], ids[2], ids[1], ids[0]],
        [*registry_parts, ids[2], ids[1], ids[0]],
    ]


def test_a_document_over_the_line_is_split_before_sending(
    answering_llm, grown_registry, window, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    documents = memos_with(grown_registry(40_000))
    result = answer(answering_llm, logger, documents=documents, model=window(5_000))
    ids = result.document_ids
    assert summary_headings(answering_llm) == as_ids(ids, REGISTRY_IN_4_PARTS)
    assert max(tokens(request) for request in answering_llm.requests) <= 4_250
    events = read_json(logger, "splits")["documents"]["3"]["events"]
    assert {event["cause"] for event in events} == {"estimate"}


def test_the_reasoning_path_changes_only_what_reasoning_sends(
    answering_llm, window, tmp_path, monkeypatch
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    documents, model = read_documents(MEMOS), window(1_000_000)
    sent = []
    for path in REASONING_PATHS:
        counting_ids(monkeypatch)
        answering_llm.requests.clear()
        logger = TaskLogger(path, root=tmp_path)
        answer(answering_llm, logger, documents=documents, model=model, reasoning=path)
        sent.append(
            [
                (r["messages"], r.get("response_format"))
                for stage in ("relevance", "schema", "parsing")
                for r in answering_llm.requests_for(stage)
            ]
        )
        assert len(answering_llm.requests_for("reasoning")) == 1
        assert not (logger.dir / "splits.json").exists()
    first, *others = sent
    assert len(first) == 16
    assert all(other == first for other in others)


@pytest.mark.parametrize("measured", [False, True], ids=["by-refusal", "by-estimate"])
def test_a_later_stage_splits_further_without_renumbering(
    answering_llm, grown_registry, window, tmp_path, measured
):
    registry = grown_registry(40_000)
    llm = answering_llm.answers(relevance=long_notes_for(registry, 2_000))
    if not measured:
        llm.refuses_over(25_000)
    logger = TaskLogger("run", root=tmp_path)
    model = window(6_000) if measured else QWEN
    result = answer(llm, logger, documents=memos_with(registry), model=model)
    record = read_json(logger, "splits")["documents"]["3"]
    assert record["parts"] == {"relevance-r1": 2, "relevance-r2": 2, "parse": 4}
    assert summary_headings(llm) == as_ids(
        result.document_ids, ["1", "2", "3", "4.1", "4.2", "5"]
    )
    calls = read_json(logger, "structuring/parsing/calls")
    replied = [call["kind"] for call in calls if call["error"] is None]
    assert sorted(kind for kind in replied if "-d3" in kind) == [
        f"parse-d3c{k}" for k in range(4)
    ]
    assert len(llm.requests_for("relevance")) == (12 if measured else 13)
    assert result.source_docs == {"rows": [0, 1, 2, 3, 3, 3, 3, 4]}


@pytest.mark.parametrize(
    ("measured", "clause"),
    [
        (False, "the part is about "),
        (True, "over the 5,100-token line by themselves"),
    ],
    ids=["by-refusal", "by-estimate"],
)
def test_splitting_stops_at_the_relevant_contexts_line(
    answering_llm, window, tmp_path, monkeypatch, measured, clause
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    reports = [f"Report {i:02d}. " + ROUTINE * 5 for i in range(40)]
    llm = answering_llm.answers(relevance=(ROUTINE * 14)[:1_000])
    if not measured:
        llm.refuses_over(20_000)
    logger = TaskLogger("run", root=tmp_path)
    model = window(6_000) if measured else QWEN
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        answer(llm, logger, documents=reports, model=model)
    note, folder = stop.value.__notes__
    assert note.startswith(
        "r3con: reading documents[0] (Document 1) in more parts cannot help in "
        "relevance-r2-d0: "
    )
    assert clause in note
    assert note.endswith("The relevant context has outgrown the model's window.")
    assert folder == f"r3con: partial artifacts in {logger.dir}"
    assert "TASK-" not in note
    assert [user for user in users(llm) if user in reports[0]] == [reports[0]] * (
        1 if measured else 2
    )
    assert len(llm.requests) == (40 if measured else 41)
    [(doc, record)] = read_json(logger, "splits")["documents"].items()
    assert (doc, record["cuts"]) == ("0", [])
    assert [event["action"] for event in record["events"]] == ["stop"]


def test_a_stopped_run_keeps_its_calls_and_splits(answering_llm, tmp_path):
    # Relevance and the schema fit under the limit; a parse call, which carries every
    # note and a whole document, does not, and its document is shorter than its rest.
    documents = [f"{name} memo. " + ROUTINE * 66 for name in ("Halloran", "Merrow")]
    llm = answering_llm.answers(relevance=(ROUTINE * 41)[:3_000]).refuses_over(13_500)
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        answer(llm, logger, documents=documents, model=QWEN)
    assert "in more parts cannot help in parse-d0" in stop.value.__notes__[0]
    written = {str(p.relative_to(logger.dir)) for p in logger.dir.rglob("*.*")}
    assert {
        "relevance/result.json",
        "structuring/schema/result.json",
        "structuring/parsing/calls.json",
        "structuring/parsing/error.txt",
        "splits.json",
    } <= written
    assert "structuring/parsing/result.json" not in written
    events = read_json(logger, "splits")["documents"]["0"]["events"]
    assert [(e["call"], e["action"]) for e in events] == [("parse-d0", "stop")]


def test_a_parse_too_big_for_the_direct_prompt_is_read_with_code_from_samples(
    answering_llm, quiet_sites, window, tmp_path, monkeypatch
):
    counting_ids(monkeypatch)
    documents = read_documents(MEMOS) + quiet_sites(55)
    llm = answering_llm.answers(relevance=memo_note, parsing=whole_document_row)
    logger, result = memos_run(llm, documents, window(8_192), tmp_path)
    assert result.answer == "; ".join(documents)
    assert len(llm.requests) == 182
    assert max(tokens(request) for request in llm.requests) <= LINE_8192
    [turn] = first_turns(llm)
    assert SAMPLE in turn["messages"][0]["content"]
    reasoning = read_json(logger, "reasoning/result")
    assert reasoning["path"] == "coding"
    assert reasoning["fallback"] == {
        "cause": "estimate",
        "estimate": 10_698,
        "line": LINE_8192,
    }
    assert (logger.dir / "reasoning" / "transcript.yaml").is_file()


def test_a_refused_first_turn_goes_again_once_with_samples(
    answering_llm, quiet_sites, tmp_path
):
    documents = read_documents(MEMOS) + quiet_sites(55)
    llm = answering_llm.answers(
        relevance=memo_note, parsing=whole_document_row
    ).refuses_over(tokens=8_192)
    _, result = memos_run(llm, documents, QWEN, tmp_path, reasoning="coding")
    assert result.answer == "; ".join(documents)
    assert len(llm.requests) == 183
    whole, samples = first_turns(llm)
    assert SAMPLE not in whole["messages"][0]["content"]
    assert tokens(whole) > 8_192
    assert SAMPLE in samples["messages"][0]["content"]


def test_a_run_reports_each_stage_and_document_from_its_start_to_its_finish(
    answering_llm,
):
    events = []
    result = answer(
        answering_llm, documents=[*DOCS, "Calder memo"], on_event=events.append
    )
    assert events[0] == {
        "type": "run_started",
        "time": events[0]["time"],
        "question": "Who?",
        "model": MODEL,
        "document_ids": result.document_ids,
        "rounds": 2,
        "run_dir": None,
    }
    types = [event["type"] for event in events]
    assert types.count("run_finished") == 1
    assert types[-1] == "run_finished"
    for i, event in enumerate(events):
        if event["type"].endswith("_finished"):
            earlier = [node(e) for e in events[:i] if e["type"].endswith("_started")]
            assert node(event) in earlier
    started = [node(e) for e in events if e["type"].endswith("_started")]
    finished = [node(e) for e in events if e["type"].endswith("_finished")]
    assert Counter(started) == Counter(finished)
    assert Counter(stage for stage, *_ in started) == {
        "run": 1,
        "relevance": 6,
        "schema": 1,
        "parsing": 3,
        "reasoning": 1,
    }
    last = {t.partition("_")[0]: i for i, t in enumerate(types)}
    assert last["relevance"] < types.index("schema_started")
    assert last["schema"] < types.index("parsing_started")
    assert last["parsing"] < types.index("reasoning_started")


def test_each_finished_event_carries_what_the_run_folder_records(
    answering_llm, shown_ids, priced_model, tmp_path, monkeypatch
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    logger = TaskLogger("run", root=tmp_path)
    events = []
    llm = answering_llm.answers(relevance=citing_every_shown_id(shown_ids))
    result = answer(
        llm, logger, documents=[*DOCS, "Calder memo"], on_event=events.append
    )
    by_type = {}
    for event in events:
        by_type.setdefault(event.pop("type"), []).append(event)
        del event["time"]
    rounds = read_json(logger, "relevance/result")["rounds"]
    assert by_type["relevance_finished"] == [
        {
            "round": r["round"],
            "doc_id": s["id"],
            "part": None,
            "summary": s["summary"],
            "provenance": s["provenance"],
        }
        for r in rounds
        for s in r["snippets"]
    ]
    assert any(e["provenance"] for e in by_type["relevance_finished"])
    schema = read_json(logger, "structuring/schema/result")
    assert by_type["schema_finished"] == [
        {"schema_code": schema["schema_code"], "thought": schema["thought"]}
    ]
    parsing = read_json(logger, "structuring/parsing/result")
    assert by_type["parsing_finished"] == [
        {"doc_id": id_, "n_records": 1, "provenance": parsing["provenance"][i]}
        for i, id_ in enumerate(result.document_ids)
    ]
    reasoning = read_json(logger, "reasoning/result")
    assert by_type["reasoning_finished"] == [
        {f: reasoning[f] for f in ("answer", "provenance", "path", "fallback")}
    ]
    assert by_type["run_finished"] == [
        {
            "run_dir": str(logger.dir),
            "tokens": {"prompt": 110, "completion": 220, "total": 330},
            "usd": 110.0,
        }
    ]


def test_run_started_names_the_run_folder_already_holding_its_manifest(
    answering_llm, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    at_start = []

    def on_event(event):
        if event["type"] == "run_started":
            folder = event["run_dir"]
            at_start.append((folder, (Path(folder) / "manifest.json").is_file()))

    answer(answering_llm, logger, on_event=on_event)
    assert at_start == [(str(logger.dir), True)]


@pytest.mark.parametrize("cause", ["estimate", "refusal"])
def test_a_split_document_starts_once_a_round_and_finishes_each_part_it_ends_up_in(
    answering_llm, grown_registry, window, tmp_path, cause
):
    logger = TaskLogger("run", root=tmp_path)
    events = []
    llm, model = {
        "estimate": (answering_llm, window(5_000)),
        "refusal": (answering_llm.refuses_over(18_000), QWEN),
    }[cause]
    documents = memos_with(grown_registry(40_000))
    result = answer(
        llm, logger, documents=documents, model=model, on_event=events.append
    )
    record = read_json(logger, "splits")["documents"]["3"]
    assert {event["cause"] for event in record["events"]} == {cause}
    starts = [node(e) for e in events if e["type"] == "relevance_started"]
    assert sorted(starts) == sorted(
        ("relevance", r, id_) for r in (1, 2) for id_ in result.document_ids
    )
    for i, event in enumerate(events):
        if event["type"] == "relevance_finished":
            assert node(event) in [node(e) for e in events[:i]]
    reported = [
        (e["round"], snippet_id(e), e["summary"], e["provenance"])
        for e in events
        if e["type"] == "relevance_finished"
    ]
    recorded = [
        (r["round"], s["id"], s["summary"], s["provenance"])
        for r in read_json(logger, "relevance/result")["rounds"]
        for s in r["snippets"]
    ]
    assert sorted(reported) == sorted(recorded)
    assert len(recorded) == 2 * len(REGISTRY_IN_4_PARTS)
    records = {
        e["doc_id"]: e["n_records"] for e in events if e["type"] == "parsing_finished"
    }
    assert records == {id_: 1 for id_ in result.document_ids} | {
        result.document_ids[3]: 4
    }


def test_events_jsonl_holds_each_event_the_callback_saw_in_time_order(
    answering_llm, events_in, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    events = []
    answer(answering_llm, logger, on_event=events.append)
    assert events_in(logger.dir) == events
    times = [datetime.datetime.fromisoformat(event["time"]) for event in events]
    assert times == sorted(times)
    assert all(t.tzinfo == datetime.UTC for t in times)


def test_a_callback_that_changes_its_events_changes_nothing_in_the_run(
    answering_llm, events_in, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)

    def clears_every_list_and_dict(event):
        for value in event.values():
            if isinstance(value, (list, dict)):
                value.clear()

    result = answer(answering_llm, logger, on_event=clears_every_list_and_dict)
    assert result.answer == "Halloran memo; Merrow memo"
    assert len(result.document_ids) == 2
    assert result.provenance["parse_ids"] == result.document_ids
    assert events_in(logger.dir)[0]["document_ids"] == result.document_ids


def test_the_callback_is_never_entered_twice_at_once(answering_llm):
    inside = threading.Lock()
    overlapping = []

    def on_event(event):
        if not inside.acquire(blocking=False):
            overlapping.append(event["type"])
            return
        time.sleep(0.002)
        inside.release()

    def slow_note(request):
        time.sleep(0.01)
        return f"Notes on {request['messages'][1]['content']}."

    documents = [f"Memo {k}" for k in range(8)]
    llm = answering_llm.answers(relevance=slow_note)
    answer(llm, documents=documents, on_event=on_event)
    assert llm.peak_in_flight > 1
    assert overlapping == []


def test_a_raising_callback_is_logged_once_per_kind_of_error_and_the_run_goes_on(
    answering_llm, events_in, tmp_path, caplog
):
    caplog.set_level(logging.DEBUG, logger="r3con.runs")
    logger = TaskLogger("run", root=tmp_path)

    def on_event(event):
        if event["type"] == "run_started":
            raise RuntimeError("the panel broke")
        raise ValueError("the panel broke again")

    result = answer(answering_llm, logger, on_event=on_event)
    assert result.answer == "Halloran memo; Merrow memo"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [w.getMessage() for w in warnings] == [
        "on_event raised RuntimeError on a run_started event; the run goes on",
        "on_event raised ValueError on a relevance_started event; the run goes on",
    ]
    assert all(w.exc_info for w in warnings)
    assert len(caplog.records) == len(events_in(logger.dir))
    assert events_in(logger.dir)[-1]["type"] == "run_finished"


def test_a_run_whose_events_file_cannot_be_written_still_finishes(
    answering_llm, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    logger = TaskLogger("run", root=tmp_path)
    (logger.dir / "events.jsonl").mkdir()
    events = []
    result = answer(answering_llm, logger, on_event=events.append)
    assert result.answer == "Halloran memo; Merrow memo"
    assert (events[0]["type"], events[-1]["type"]) == ("run_started", "run_finished")
    [warning] = caplog.records
    assert warning.getMessage() == (
        f"writing {logger.dir / 'events.jsonl'} raised IsADirectoryError on a "
        "run_started event; the run goes on"
    )


@pytest.mark.parametrize("stage", list(STAGE_FOLDERS))
def test_a_failing_stage_reports_run_failed_with_its_name_and_no_finish(
    answering_llm, events_in, tmp_path, stage
):
    logger = TaskLogger("run", root=tmp_path)
    events = []
    with pytest.raises(litellm.APIConnectionError):
        answer(
            answering_llm.answers(**{stage: provider_down()}),
            logger,
            on_event=events.append,
        )
    *_, failed = events
    assert failed == {
        "type": "run_failed",
        "time": failed["time"],
        "stage": stage,
        "error": "APIConnectionError: litellm.APIConnectionError: provider down",
    }
    assert "run_finished" not in [event["type"] for event in events]
    assert events_in(logger.dir) == events

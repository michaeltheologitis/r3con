import os

# Before litellm is imported: otherwise it fetches its price map from GitHub, and
# collection runs outside pytest-socket's reach.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import json
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import litellm
import pytest

# Its import-time IPv6 probe opens a socket: here, before sockets are blocked.
import urllib3  # noqa: F401
import yaml

from r3con.config import normalize_model_name
from tests.fakes import Content, FakeLLM, Request, too_long

litellm.suppress_debug_info = True

MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
MODEL = "openai/gpt-6-luna"
R3CON_VARIABLES = (
    "R3CON_LOGS_DIR",
    "R3CON_DOC_WORKERS",
    "R3CON_PROMPTS_DIR",
    "R3CON_CONFIGS_DIR",
    "R3CON_LOG_LEVEL",
)
ROUTER_CALLBACKS = (
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
)
# The line, and the blank one after it, that heads a parse request's user message.
PARSE_HEADING = re.compile(r"\ADocument ID: \w+\n\n")
ANSWERING_SCHEMA = """Thought: one row per person.
<schema>
class Row(BaseModel):
    who: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
ANSWERING_CODE = (
    '<code>\nfinal_answer("; ".join(r["who"] for r in parse["rows"]), '
    'provenance_ids={"summary_ids": [], '
    '"parse_ids": [r["document"] for r in parse["rows"]]})\n</code>'
)


def grow_registry(chars: int, at: float = 0.6) -> str:
    """The contractor registry memo grown to about ``chars`` characters by filler
    paragraphs, with its heading first and its code lines and footer at fraction
    ``at``. The filler names no contractor and counts nothing."""
    heading, body = (MEMOS / "04_contractor_registry.txt").read_text().split("\n\n", 1)
    room = chars - len(heading) - len(body)
    before = _filler(0, round(room * at))
    after = _filler(len(before), room - _size(before))
    return _PARAGRAPH.join([heading, *before, body.strip(), *after])


def _filler(start: int, chars: int) -> list[str]:
    """Paragraphs of procurement and insurance boilerplate, each tagged uniquely from
    ``start`` on, about ``chars`` characters with their breaks."""
    paragraphs: list[str] = []
    while _size(paragraphs) < chars:
        i = start + len(paragraphs)
        tag = "".join(chr(ord("a") + int(digit)) for digit in str(i))
        paragraphs.append(_BOILERPLATE[i % len(_BOILERPLATE)].format(tag=tag))
    return paragraphs


def _size(paragraphs: list[str]) -> int:
    return sum(len(paragraph) + len(_PARAGRAPH) for paragraph in paragraphs)


_PARAGRAPH = "\n\n"
_BOILERPLATE = (
    (
        "Procurement note, {tag}. Purchase orders above the delegated limit need a second "
        "signature from the regional office before a contractor is engaged. Quotations are "
        "kept on file for the period the finance policy sets, and a contractor that "
        "declines to quote is recorded as such rather than left out."
    ),
    (
        "Insurance note, {tag}. Each approved contractor holds public liability and "
        "employer's liability cover at the levels the framework agreement sets. "
        "Certificates are renewed every year and checked by the compliance team; a lapsed "
        "certificate suspends new work orders until a current one arrives."
    ),
    (
        "Onboarding note, {tag}. A new contractor completes the site induction, the "
        "permit-to-work briefing and the lone-working assessment before a first visit. "
        "Induction records are held by the site office and are not reproduced in this "
        "extract."
    ),
    (
        "Payment note, {tag}. Invoices are matched against the work order and the "
        "completion sheet signed on site. A disputed line is held, not rejected, and the "
        "contractor is told in writing which line is held and why."
    ),
    (
        "Review note, {tag}. The facilities board reviews this registry every quarter. "
        "Changes to scope, rates or contact details take effect from the first day of the "
        "following month and are sent to every site manager."
    ),
    (
        "Records note, {tag}. This extract leaves out rates, bank details and named "
        "contacts, which are held in the procurement system. Requests for the full record "
        "go to the facilities administrator."
    ),
)


_SITES = (
    "Ashby",
    "Brook",
    "Calder",
    "Dunmore",
    "Elsworth",
    "Fairlie",
    "Glenholm",
    "Harrow",
    "Ingle",
    "Jarrow",
    "Kelso",
    "Linton",
)


def quiet_site_memos(n: int) -> list[str]:
    """``n`` site memos of about 230 characters, each under its own site name, that log
    no equipment incident, under CT-204 (even k) or CT-311 (odd k): added to the five
    memos, they cannot change the answer."""
    return [
        f"MEMO — {_SITES[k % 12]} {k:03d} site, Q3 operations review\n\n"
        "Equipment incidents logged this quarter: 0.\nNo faults were reported.\n"
        "Planned maintenance at this site is carried out under contractor code "
        f"{'CT-204' if k % 2 == 0 else 'CT-311'}.\n"
        f"Site manager: {_SITES[(k * 7) % 12]} Hale."
        for k in range(n)
    ]


def summary_ids_shown(request: Request) -> list[str]:
    """The ids a later-rounds relevance request shows, in order, without its
    examples'."""
    shown = request["messages"][0]["content"].split("\n## Output\n")[0]
    return re.findall(r"^### Summary ID: (\S+)$", shown, re.MULTILINE)


SUMMARIES = "## Document summaries"
PARSE = "## The parsed data (`parse`)"
REASONING_SECTIONS = {
    "direct": [
        "## The task and your two views",
        SUMMARIES,
        PARSE,
        "## Rules",
        "## Output",
    ],
    "coding": [
        "## The task and your two views",
        "## Execution environment",
        SUMMARIES,
        PARSE,
        "## Rules",
        "## How to work",
    ],
}
# v2 shares v3's headings: only a line like this tells the versions apart.
REASONING_V3 = "`summary_ids`: The IDs of the summaries above that you used in writing"


def prompt_sections(prompt: str) -> dict[str, str]:
    """A prompt's ``## `` sections, in order, each heading to its body, stripped,
    without its examples'. Fails on a prompt with template text left unrendered
    (``{{``, ``{%`` or ``None``) or a heading that repeats."""
    instructions = prompt.split("\n\nExamples:\n\n", 1)[0]
    for unrendered in ("{{", "{%", "None"):
        assert unrendered not in instructions
    _, *parts = re.split(r"^(## .+)$", instructions, flags=re.MULTILINE)
    headings = parts[::2]
    assert len(set(headings)) == len(headings), f"a section repeats: {headings}"
    return {heading: body.strip() for heading, body in zip(headings, parts[1::2])}


def reasoning_prompt_sections(
    prompt: str, path: str, *, summaries: bool = True
) -> dict[str, str]:
    """``prompt``'s sections (:func:`prompt_sections`), checked to be those of
    ``path``'s v3 reasoning prompt, its summaries section only with ``summaries``."""
    shown = prompt_sections(prompt)
    expected = [h for h in REASONING_SECTIONS[path] if summaries or h != SUMMARIES]
    assert list(shown) == expected
    assert REASONING_V3 in prompt
    return shown


def shown_parse(sections: dict[str, str]) -> Any:
    """The parse a reasoning prompt's ``sections`` print, read back from its JSON."""
    parse = sections[PARSE]
    return json.loads(parse[parse.index("{") :])


def _first_line(request: Request) -> str:
    """The first line of the document a request sends, below a parse request's
    ``Document ID`` line."""
    return re.sub(PARSE_HEADING, "", request["messages"][1]["content"]).splitlines()[0]


def _joined_rows(request: Request) -> Content:
    """A reasoning reply that joins the parse's rows and cites their documents for
    their records: on the coding path, code that does; on the direct path, the rows of
    the parse its prompt shows, joined, and their documents' ids."""
    if "response_format" not in request:
        return ANSWERING_CODE
    system = request["messages"][0]["content"]
    shown = system.split("## The parsed data (`parse`)", 1)[1].split("\n## Rules")[0]
    rows = json.loads(shown[shown.index("{") : shown.rindex("}") + 1])["rows"]
    return {
        "answer": "; ".join(row["who"] for row in rows),
        "provenance_ids": {
            "summary_ids": [],
            "parse_ids": [row["document"] for row in rows],
        },
    }


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def answering_llm(llm: FakeLLM) -> FakeLLM:
    """``llm``, answering every stage so that any run completes: each document's
    first line is its note and its one parsed row, and the answer joins the rows, on
    either reasoning path."""
    return llm.answers(
        relevance=lambda request: f"Notes on {_first_line(request)}.",
        schema=ANSWERING_SCHEMA,
        parsing=lambda request: {"rows": [{"who": _first_line(request)}]},
        reasoning=_joined_rows,
    )


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], str]:
    """Maps a model with an input window of ``tokens`` in litellm's model map and
    returns its name. litellm caches its lookup per name, so the name carries the
    window: one name always means one window, across tests."""

    def register(tokens: int) -> str:
        model = f"hosted_vllm/window-{tokens}"
        entry = {"max_input_tokens": tokens, "litellm_provider": "hosted_vllm"}
        monkeypatch.setitem(litellm.model_cost, model, {**entry, "mode": "chat"})
        return model

    return register


@pytest.fixture
def refusal() -> Callable[[], litellm.ContextWindowExceededError]:
    """Makes a new error of the kind litellm raises for a request refused as too long."""
    return lambda: too_long(4_000, 5_000, MODEL)


@pytest.fixture
def priced_model() -> Iterator[dict[str, float]]:
    """MODEL at round test rates, which it yields: 10 tokens in and 20 out cost $10 a
    call. litellm caches each model's price by name, so the cache is cleared once the
    rates are set and again once they are undone."""
    name = normalize_model_name(MODEL)
    rates = {"input_cost_per_token": 0.5, "output_cost_per_token": 0.25}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            litellm.model_cost, name, {**litellm.model_cost.get(name, {}), **rates}
        )
        litellm.get_model_info.cache_clear()
        yield rates
    litellm.get_model_info.cache_clear()


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> Callable[..., litellm.Router]:
    """Builds a ``litellm.Router`` over deployments given as their ``litellm_params``,
    each named by its model, and undoes what building one leaves in litellm's globals:
    its callbacks, and its deployments in the price map, which litellm 1.103 prices at
    0 for every later call to the same model."""
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for name in ROUTER_CALLBACKS:
        monkeypatch.setattr(litellm, name, list(getattr(litellm, name)))

    def build(*deployments: dict[str, Any]) -> litellm.Router:
        model_list = [
            {"model_name": d["model"], "litellm_params": d} for d in deployments
        ]
        return litellm.Router(model_list=model_list, num_retries=0)

    return build


@pytest.fixture
def grown_registry() -> Callable[..., str]:
    return grow_registry


@pytest.fixture
def quiet_sites() -> Callable[[int], list[str]]:
    return quiet_site_memos


@pytest.fixture
def shown_ids() -> Callable[[Request], list[str]]:
    return summary_ids_shown


@pytest.fixture
def sections() -> Callable[[str], dict[str, str]]:
    return prompt_sections


@pytest.fixture
def reasoning_sections() -> Callable[..., dict[str, str]]:
    return reasoning_prompt_sections


@pytest.fixture
def parse_shown() -> Callable[[dict[str, str]], Any]:
    return shown_parse


def recorded_events(folder: Path) -> list[dict[str, Any]]:
    """The events in ``folder``'s ``events.jsonl``, in order."""
    lines = (folder / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def events_in() -> Callable[[Path], list[dict[str, Any]]]:
    return recorded_events


@pytest.fixture
def configs(tmp_path):
    """Writes ``name -> fields`` into the working directory's config overlay."""
    root = tmp_path / "configs"
    root.mkdir()

    def write(**by_name: dict) -> None:
        for name, fields in by_name.items():
            (root / f"{name}.yaml").write_text(yaml.safe_dump(fields))

    return write


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The seconds r3con waits before each resend, recorded instead of waited."""
    recorded: list[float] = []
    monkeypatch.setattr(
        "r3con.runtime.llm._sleep", lambda seconds, cancel: recorded.append(seconds)
    )
    return recorded


@pytest.fixture
def isolated_r3con(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No ``R3CON_*`` variable, no key registered for masking, the test's own working
    directory, and the ``r3con`` logger restored afterwards."""
    for name in R3CON_VARIABLES:
        # Set, then delete, so the undo also removes a value the code under test writes.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    # Each test starts with no key registered for masking by an earlier one's calls.
    monkeypatch.setattr("r3con.masking._registered", set())
    logger = logging.getLogger("r3con")
    level, propagate, handlers = logger.level, logger.propagate, list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.propagate = propagate
    logger.handlers[:] = handlers


@pytest.fixture(autouse=True)
def _isolated(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    isolated_r3con: None,
) -> None:
    if not any(request.node.get_closest_marker(m) for m in ("live", "allow_hosts")):
        monkeypatch.setattr(litellm, "completion", FakeLLM())
    if not request.node.get_closest_marker("live"):
        request.getfixturevalue("waits")


@pytest.fixture(scope="session")
def httpserver_listen_address() -> tuple[str, int]:
    """Serve pytest-httpserver on 127.0.0.1, the host the tests' ``allow_hosts`` names.

    Its default, ``localhost``, resolves to ::1 first on GitHub's runners, and
    pytest-socket refuses that connection.
    """
    return ("127.0.0.1", 0)

import os

# Before litellm is imported: otherwise it fetches its price map from GitHub, and
# collection runs outside pytest-socket's reach.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import json
import logging
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn

import keyring
import litellm
import pytest
import pytest_socket

# Its import-time IPv6 probe opens a socket: here, before sockets are blocked.
import urllib3  # noqa: F401
import yaml
from litellm.llms.chatgpt.authenticator import Authenticator as ChatGPTAuthenticator
from litellm.llms.github_copilot.authenticator import (
    Authenticator as CopilotAuthenticator,
)

from r3con.config import normalize_model_name
from tests.fakes import (
    ANSWERING,
    FakeKeyring,
    FakeLLM,
    Request,
    too_long,
)

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
TOKEN_DIR_VARIABLES = ("GITHUB_COPILOT_TOKEN_DIR", "CHATGPT_TOKEN_DIR")
# keyring's own choice of store, read in each process it starts in.
KEYRING_BACKEND_VARIABLE = "PYTHON_KEYRING_BACKEND"
NULL_KEYRING = "keyring.backends.null.Keyring"
TOKEN_REQUESTS = (
    (CopilotAuthenticator, "get_api_key", "Copilot"),
    (CopilotAuthenticator, "get_access_token", "Copilot"),
    (ChatGPTAuthenticator, "get_access_token", "ChatGPT"),
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


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def answering_llm(llm: FakeLLM) -> FakeLLM:
    """``llm``, answering every stage so that any run completes: each document's
    first line is its note and its one parsed row, and the answer joins the rows, on
    either reasoning path."""
    return llm.answers(**ANSWERING)


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
def own_trace(request: pytest.FixtureRequest) -> str:
    """A ``trace`` for the metadata of this test's calls, its own: the test's id."""
    return request.node.nodeid


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch, own_trace: str) -> threading.Semaphore:
    """Released each time litellm, on its own thread, is done logging a call that
    answered with ``own_trace`` as its metadata's ``trace``: by then it has written that
    call's ``hidden_params`` there. litellm reads its callbacks when it logs, so an
    earlier test's call, logged late, calls this one too, but releases nothing."""
    done = threading.Semaphore(0)

    def release_for_this_test(kwargs: dict[str, Any], *_: Any) -> None:
        if kwargs["litellm_params"]["metadata"].get("trace") == own_trace:
            done.release()

    monkeypatch.setattr(litellm, "success_callback", [release_for_this_test])
    return done


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
    """The events in ``folder``'s ``events.jsonl``, in order: one a line, a line ended
    by "\n" alone, as an event's text may hold other line separators."""
    lines = (folder / "events.jsonl").read_text().split("\n")
    return [json.loads(line) for line in lines if line]


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
def credential_store(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    """An empty credential store in memory, in keyring's place for the whole test, so
    that no test reads or writes the system's. keyring finds the system's only when
    first asked for a store, which this answers first; a process the test starts,
    such as ``r3con app``, gets keyring's null store, which holds nothing."""
    monkeypatch.setenv(KEYRING_BACKEND_VARIABLE, NULL_KEYRING)
    store = FakeKeyring()
    keyring.set_keyring(store)
    return store


@pytest.fixture(autouse=True)
def _isolated(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    isolated_r3con: None,
) -> None:
    if not any(request.node.get_closest_marker(m) for m in ("live", "allow_hosts")):
        monkeypatch.setattr(litellm, "completion", FakeLLM())
    # Each test is a process of its own to a provider that signs in on its first call.
    monkeypatch.setattr("r3con.runtime.llm._signed_in", set())
    if not request.node.get_closest_marker("live"):
        request.getfixturevalue("waits")


tokens_asked_for: list[str] = []


def _refused(provider: str) -> Callable[[object], NoReturn]:
    """An authenticator method that records a request for a ``provider`` token in
    ``tokens_asked_for``, then refuses it."""
    message = f"a test asked litellm for a {provider} token, starting its sign-in"

    def refuse(self: object) -> NoReturn:
        tokens_asked_for.append(message)
        raise RuntimeError(message)

    return refuse


# Never undone: a worker thread a test leaves running can still ask while a later test
# allows sockets.
for authenticator, method, provider in TOKEN_REQUESTS:
    setattr(authenticator, method, _refused(provider))


@pytest.fixture(autouse=True)
def _no_litellm_sign_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Gives litellm the test's own token folder, and fails a test that asks it for a
    Copilot or ChatGPT token."""
    for variable in TOKEN_DIR_VARIABLES:
        monkeypatch.setenv(variable, str(tmp_path))
    tokens_asked_for.clear()
    yield
    # litellm turns the refusal into a failed lookup, which code under test may pass.
    assert not tokens_asked_for, sorted(set(tokens_asked_for))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup() -> None:
    """Lifts the last test's pytest-socket guard, as pytest-socket's setup expects."""
    pytest_socket._remove_restrictions()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown() -> Iterator[None]:
    """Keeps the test's pytest-socket guard up through its fixtures' teardown, until
    the next test's setup lifts it."""
    # pytest-socket's own teardown hook lifts the guard before pytest's tears them down.
    lift_guard = pytest_socket._remove_restrictions
    pytest_socket._remove_restrictions = lambda: None
    try:
        yield
    finally:
        pytest_socket._remove_restrictions = lift_guard


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    """Puts the run's guard back for good before pytest's own session-end work: a
    thread a test left running may outlive the session."""
    config = session.config
    blocked = config.getoption("--disable-socket", default=False)
    if blocked and not config.getoption("--force-enable-socket"):
        pytest_socket.disable_socket(config.getoption("--allow-unix-socket"))


@pytest.fixture(scope="session")
def httpserver_listen_address() -> tuple[str, int]:
    """Serve pytest-httpserver on 127.0.0.1, the host the tests' ``allow_hosts`` names.

    Its default, ``localhost``, resolves to ::1 first on GitHub's runners, and
    pytest-socket refuses that connection.
    """
    return ("127.0.0.1", 0)

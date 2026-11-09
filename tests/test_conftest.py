import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError

REPO = Path(__file__).resolve().parents[1]
LOCAL_HOST = "127.0.0.1"
# Loopback too, so a connection the guard failed to refuse would not leave the machine.
OTHER_LOCAL_HOST = "127.0.0.2"
SESSION_TIMEOUT_SECONDS = 120
# A conftest.py for a session of its own. Outside any test, it tries to open a network
# socket, as a thread a test left running might, and logs each try to outcomes.txt.
OUTSIDE_TESTS = """\
import socket
from pathlib import Path

from pytest_socket import SocketBlockedError

OUTCOMES = Path(__file__).with_name("outcomes.txt")


def try_a_network_socket(moment):
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except SocketBlockedError:
        outcome = "refused"
    else:
        outcome = "opened"
    with OUTCOMES.open("a") as outcomes:
        outcomes.write(f"{moment}: {outcome}\\n")


def pytest_runtest_logfinish(nodeid):
    if nodeid.endswith("::test_blocked"):
        try_a_network_socket("between tests")


def pytest_sessionfinish():
    # Where pytest does its own session-end work, such as deleting old temp folders.
    try_a_network_socket("as the session finishes")


def pytest_unconfigure():
    try_a_network_socket("once the session is over")
"""
# Its tests: a blocked one, then a last one that allows sockets.
TESTS = """\
import pytest


def test_blocked():
    pass


@pytest.mark.enable_socket
def test_allows_sockets():
    pass
"""

# pytest-socket warns as it refuses a socket.
pytestmark = pytest.mark.filterwarnings("ignore:A test tried to use socket")


@pytest.fixture
def network_socket_at_teardown() -> Iterator[None]:
    """Once its test is done, opens a network socket, as a thread the test left running
    might."""
    yield
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.fixture
def local_connections_at_teardown() -> Iterator[None]:
    """Serves on ``LOCAL_HOST``; once its test is done, connects there, then to the
    same port on ``OTHER_LOCAL_HOST``."""
    with socket.create_server((LOCAL_HOST, 0)) as server:
        port = server.getsockname()[1]
        yield
        socket.create_connection((LOCAL_HOST, port)).close()
        with pytest.raises(SocketConnectBlockedError):
            socket.create_connection((OTHER_LOCAL_HOST, port))


@pytest.mark.usefixtures("network_socket_at_teardown")
def test_a_fixtures_teardown_cannot_open_a_network_socket():
    pass


@pytest.mark.allow_hosts([LOCAL_HOST])
@pytest.mark.usefixtures("local_connections_at_teardown")
def test_a_fixtures_teardown_reaches_only_the_hosts_its_test_allows():
    pass


def test_sockets_stay_blocked_between_tests_and_once_the_session_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "conftest.py").write_text(OUTSIDE_TESTS)
    module = tmp_path / "test_session.py"
    module.write_text(TESTS)
    monkeypatch.setenv("PYTHONPATH", str(REPO), prepend=os.pathsep)
    session = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "--disable-socket",
            "--allow-unix-socket",
            f"--basetemp={tmp_path / 'basetemp'}",
            str(module),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=SESSION_TIMEOUT_SECONDS,
        check=False,
    )
    assert session.returncode == 0, session.stdout
    assert (tmp_path / "outcomes.txt").read_text().splitlines() == [
        "between tests: refused",
        "as the session finishes: refused",
        "once the session is over: refused",
    ]

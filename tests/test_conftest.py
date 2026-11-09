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
# A test module for a session of its own, which writes to outcome.txt beside it whether
# the thread its test left running could open a network socket once that session ended.
LEFT_RUNNING = """\
import socket
import threading
from pathlib import Path

from pytest_socket import SocketBlockedError

OUTCOME = Path(__file__).with_name("outcome.txt")


def open_a_network_socket(session_over):
    session_over.wait()
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except SocketBlockedError:
        OUTCOME.write_text("refused")
    else:
        OUTCOME.write_text("opened")


def test_leaves_a_thread_running(pytestconfig):
    session_over = threading.Event()
    threading.Thread(target=open_a_network_socket, args=(session_over,)).start()
    pytestconfig.add_cleanup(session_over.set)
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


def test_a_thread_left_running_cannot_open_a_network_socket_once_the_session_is_over(
    tmp_path: Path,
):
    module = tmp_path / "test_left_running.py"
    module.write_text(LEFT_RUNNING)
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
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=SESSION_TIMEOUT_SECONDS,
        check=False,
    )
    assert session.returncode == 0, session.stdout
    assert (tmp_path / "outcome.txt").read_text() == "refused"

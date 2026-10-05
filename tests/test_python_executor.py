"""The vendored smolagents executor: the sandboxing r3con relies on to run
model-written Python in-process.

If one of these fails after a vendor update, decide deliberately: the new behaviour
is fine for r3con, or it broke a property r3con relies on.
"""

import textwrap
import time

import pytest

from r3con.runtime.python_executor import (
    ExecutionTimeoutError,
    InterpreterError,
    LocalPythonExecutor,
)


@pytest.fixture
def executor() -> LocalPythonExecutor:
    return LocalPythonExecutor(additional_authorized_imports=[])


@pytest.mark.parametrize(
    ("code", "output"),
    [
        ("1 + 2", 3),
        ("a = 10\nb = 20\nc = a + b\nc * 2", 60),
        ("import math\nmath.floor(3.7)", 3),
        ("from datetime import date\n(date(2023, 3, 20) - date(2023, 2, 11)).days", 37),
    ],
)
def test_the_last_expressions_value_is_the_output(executor, code, output):
    assert executor(code).output == output


def test_prints_are_captured_into_the_logs(executor, capsys):
    out = executor("print('hello')\nprint('world')\n42")
    assert out.logs == "hello\nworld\n"
    assert out.output == 42
    assert capsys.readouterr().out == ""


def test_sent_variables_are_visible_to_the_code(executor):
    executor.send_variables({"parse": {"items": [{"n": 1}, {"n": 2}, {"n": 3}]}})
    assert executor("sum(it['n'] for it in parse['items'])").output == 6


def test_state_persists_from_one_call_to_the_next(executor):
    executor("x = 100")
    assert executor("x + 1").output == 101


def test_an_extra_authorized_import_is_let_through():
    executor = LocalPythonExecutor(additional_authorized_imports=["json"])
    assert executor("import json\njson.loads('{\"a\": 1}')").output == {"a": 1}


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("import os", "Import of os is not allowed"),
        ("(1).__class__", "__class__"),
        ("open('/etc/passwd')", "open"),
        ("exec('x = 1')", "exec"),
        ("def f(:", "SyntaxError"),
        ("raise RuntimeError('boom')", "boom"),
        ("does_not_exist + 1", "does_not_exist"),
    ],
)
def test_what_the_sandbox_forbids_or_cannot_run_raises_an_interpreter_error(
    executor, code, error
):
    with pytest.raises(InterpreterError, match=error):
        executor(code)


def test_a_runaway_loop_is_stopped():
    executor = LocalPythonExecutor(additional_authorized_imports=[], timeout_seconds=1)
    started = time.perf_counter()
    with pytest.raises((InterpreterError, ExecutionTimeoutError)):
        executor("while True:\n    pass")
    # The worker thread cannot be killed; the loop ends at the iteration cap, which
    # takes seconds under load. The bound catches a hang, not a slow machine.
    assert time.perf_counter() - started < 20.0


@pytest.mark.parametrize(
    ("code", "printed"),
    [
        (
            """
            def f():
                try:
                    return 42
                except:
                    return -1
            print(f())
            """,
            "42\n",
        ),
        (
            """
            def f():
                try:
                    return "success"
                except Exception:
                    return "oops"
            print(f())
            """,
            "success\n",
        ),
        (
            """
            result = None
            for i in range(10):
                try:
                    if i == 3:
                        break
                except:
                    pass
                result = i
            print(result)
            """,
            "2\n",
        ),
        (
            """
            out = []
            for i in range(5):
                try:
                    if i == 2:
                        continue
                    out.append(i)
                except:
                    pass
            print(out)
            """,
            "[0, 1, 3, 4]\n",
        ),
        (
            """
            def f():
                try:
                    raise ValueError("boom")
                except Exception as e:
                    return f"caught: {e}"
            print(f())
            """,
            "caught: boom\n",
        ),
    ],
)
def test_return_break_and_continue_pass_through_except_as_in_python(
    executor, code, printed
):
    assert executor(textwrap.dedent(code)).logs == printed

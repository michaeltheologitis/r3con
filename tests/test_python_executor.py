"""The vendored smolagents executor: the sandboxing r3con relies on to run
model-written Python in-process.

If one of these fails after a vendor update, decide deliberately: the new behaviour
is fine for r3con, or it broke a property r3con relies on.
"""

import re
import textwrap
import time

import pytest

from r3con.runtime.python_executor import (
    BASE_BUILTIN_MODULES,
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
        (
            "import os",
            re.escape(
                "Import of os is not allowed. Authorized imports are: "
                f"{sorted(BASE_BUILTIN_MODULES)}"
            ),
        ),
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


TIME_LIMIT_S = 0.5
# Each loop ends on its own after LOOP_S, so a test fails rather than hangs.
LOOP_S = 3
STOPS_WITHIN_S = 2.0
LOOP = f"""
import time
end = time.monotonic() + {LOOP_S}
while time.monotonic() < end:
    pass
"""
LOOP_INSIDE_TRY = f"""
import time
end = time.monotonic() + {LOOP_S}
for i in range(10**9):
    try:
        if time.monotonic() > end:
            break
    except Exception:
        pass
"""
LOOP_INSIDE_BARE_EXCEPT = LOOP_INSIDE_TRY.replace("except Exception:", "except:")


@pytest.fixture
def limited_executor() -> LocalPythonExecutor:
    return LocalPythonExecutor(
        additional_authorized_imports=[], timeout_seconds=TIME_LIMIT_S
    )


def seconds_until_timed_out(executor: LocalPythonExecutor, code: str) -> float:
    started = time.perf_counter()
    with pytest.raises(ExecutionTimeoutError):
        executor(code)
    return time.perf_counter() - started


@pytest.mark.parametrize(
    "code",
    [LOOP, LOOP_INSIDE_TRY, LOOP_INSIDE_BARE_EXCEPT],
    ids=["plain", "except Exception", "bare except"],
)
def test_a_python_loop_stops_at_the_time_limit(limited_executor, code):
    assert seconds_until_timed_out(limited_executor, code) < STOPS_WITHIN_S


def test_variables_from_earlier_calls_survive_a_call_that_timed_out(
    limited_executor,
):
    limited_executor("rows = ['a', 'b']")
    seconds_until_timed_out(limited_executor, LOOP + "rows = []")
    assert limited_executor("rows").output == ["a", "b"]


def test_a_closure_from_an_earlier_call_outlives_that_calls_time_limit(
    limited_executor,
):
    limited_executor("def adder(n):\n    return lambda x: x + n\nadd_two = adder(2)")
    time.sleep(2 * TIME_LIMIT_S)
    assert limited_executor("add_two(1)").output == 3


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

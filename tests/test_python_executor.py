"""Tests for `r3con.python_executor`.

This module is vendored from smolagents.local_python_executor (Apache-2.0).
These tests pin the contract we depend on — the sandboxing guarantees that
let us run LLM-emitted Python in-process without a subprocess.

If a test here fails after vendor updates, decide deliberately: either
update the test (the new behavior is fine for us) or hold the line
(the change broke a security property we relied on).

Run with:  uv run python tests/test_python_executor.py
"""

from __future__ import annotations

import time

from r3con.runtime.python_executor import (
    BASE_PYTHON_TOOLS,
    ExecutionTimeoutError,
    InterpreterError,
    LocalPythonExecutor,
    evaluate_python_code,
)


# ---------- core eval ----------

def test_last_expression_value_is_returned() -> None:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex("1 + 2")
    assert out.output == 3


def test_print_is_captured_into_logs_not_host_stdout() -> None:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex("print('hello')\nprint('world')\n42")
    assert "hello" in out.logs
    assert "world" in out.logs
    assert out.output == 42


def test_multistatement_program_runs() -> None:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex("a = 10\nb = 20\nc = a + b\nprint(c)\nc * 2")
    assert out.output == 60
    assert "30" in out.logs


def test_send_variables_injects_into_state() -> None:
    """Injected variables are visible to the executed code."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    ex.send_variables({"parse": {"items": [{"n": 1}, {"n": 2}, {"n": 3}]}})
    out = ex("sum(it['n'] for it in parse['items'])")
    assert out.output == 6


def test_state_persists_across_calls() -> None:
    """Calls share state — useful for incremental sessions but irrelevant to one-shot use."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    ex("x = 100")
    out = ex("x + 1")
    assert out.output == 101


# ---------- authorized imports ----------

def test_default_builtin_module_is_importable() -> None:
    """`math` is in BASE_BUILTIN_MODULES so it should import without being listed explicitly."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex("import math\nmath.floor(3.7)")
    assert out.output == 3


def test_unauthorized_import_raises() -> None:
    """`os` is NOT in BASE_BUILTIN_MODULES — must be blocked unless explicitly allowed."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("import os")
    except InterpreterError as e:
        assert "os" in str(e).lower() or "import" in str(e).lower()
    else:
        raise AssertionError("expected InterpreterError on unauthorized `import os`")


def test_additional_authorized_imports_unblocks_module() -> None:
    """An additional_authorized_imports entry should let that module through."""
    ex = LocalPythonExecutor(additional_authorized_imports=["json"])
    out = ex('import json\njson.loads(\'{"a": 1}\')')
    assert out.output == {"a": 1}


# ---------- sandboxing ----------

def test_dunder_attribute_access_is_blocked() -> None:
    """`__class__`/`__subclasses__` style escapes must be rejected."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("(1).__class__")
    except InterpreterError as e:
        assert "dunder" in str(e).lower() or "__class__" in str(e)
    else:
        raise AssertionError("expected dunder access to be blocked")


def test_open_is_not_in_default_builtins() -> None:
    """`open` is deliberately absent from BASE_PYTHON_TOOLS — code cannot touch the filesystem."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("open('/etc/passwd')")
    except InterpreterError:
        pass  # any rejection is fine; we just don't want it to succeed
    else:
        raise AssertionError("expected `open` to be unavailable")


def test_exec_is_not_in_default_builtins() -> None:
    """`exec` would let the LLM bypass the AST walker entirely."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("exec('x = 1')")
    except InterpreterError:
        pass
    else:
        raise AssertionError("expected `exec` to be unavailable")


# ---------- failure modes ----------

def test_syntax_error_raises_interpreter_error() -> None:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("def f(:")
    except InterpreterError as e:
        assert "SyntaxError" in str(e) or "syntax" in str(e).lower()
    else:
        raise AssertionError("expected InterpreterError on syntax error")


def test_runtime_error_raises_interpreter_error_with_message() -> None:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("raise RuntimeError('boom')")
    except InterpreterError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("expected InterpreterError on raise")


def test_runaway_loop_is_killed_by_timeout() -> None:
    """Wall-clock timeout caps execution — pathological loops must not hang the host.

    Note: the upstream contract raises ``ExecutionTimeoutError`` (not
    ``InterpreterError``) on the wall-clock cap. The analyst wrapper must
    catch both.
    """
    ex = LocalPythonExecutor(additional_authorized_imports=[], timeout_seconds=1)
    t0 = time.perf_counter()
    try:
        ex("while True:\n    pass")
    except (InterpreterError, ExecutionTimeoutError):
        pass
    else:
        raise AssertionError("expected timeout to terminate the loop")
    elapsed = time.perf_counter() - t0
    # Generous ceiling: this asserts the loop TERMINATES (doesn't hang the host), not that it's
    # fast. The worker thread can't be force-killed, so the call blocks until the loop self-ends at
    # the MAX_WHILE_ITERATIONS cap — an AST-interpreted grind (~5s) that stretches under CPU load.
    # Bound it loosely to catch a true hang rather than pin a load-sensitive speed (was 5.0s, which
    # flaked on a busy machine).
    assert elapsed < 20.0, f"runaway loop was not terminated ({elapsed:.2f}s)"


def test_undefined_name_raises_interpreter_error() -> None:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    try:
        ex("does_not_exist + 1")
    except InterpreterError as e:
        assert "does_not_exist" in str(e) or "not defined" in str(e).lower()
    else:
        raise AssertionError("expected InterpreterError on NameError")


# ---------- module-level convenience function ----------

def test_evaluate_python_code_function_works_standalone() -> None:
    """The free function is the lower-level entry point — needs static_tools wired in by the caller."""
    output, _ = evaluate_python_code(
        "print('x')\n7",
        static_tools=BASE_PYTHON_TOOLS,
        state={},
        authorized_imports=[],
    )
    assert output == 7


# ---------- control-flow tunnels through except (Python semantics) ----------
#
# Regression tests for the upstream-divergent fix where BreakException,
# ContinueException, and ReturnException now inherit from BaseException so
# user-emitted `try: ... except:` / `except Exception:` clauses don't swallow
# return / break / continue. See python_executor.py header for context.


def test_return_inside_try_except_returns_the_value() -> None:
    """`return X` inside a try-with-bare-except returns X, not None.

    Pre-fix: ReturnException inherited from Exception, so `except:` caught
    it and the function returned via the except branch (or fell through
    to None). Post-fix: ReturnException inherits from BaseException so a
    bare `except:` doesn't see it.
    """
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex(
        "def f():\n"
        "    try:\n"
        "        return 42\n"
        "    except:\n"
        "        return -1\n"
        "result = f()\n"
        "print(result)"
    )
    assert "42" in out.logs


def test_return_inside_try_except_exception_returns_the_value() -> None:
    """Same as above but with `except Exception:` (the common idiom)."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex(
        "def f():\n"
        "    try:\n"
        "        return 'success'\n"
        "    except Exception:\n"
        "        return 'oops'\n"
        "print(f())"
    )
    assert "success" in out.logs


def test_break_inside_try_except_exits_the_loop() -> None:
    """`break` inside a try-with-bare-except exits the enclosing loop."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex(
        "result = None\n"
        "for i in range(10):\n"
        "    try:\n"
        "        if i == 3:\n"
        "            break\n"
        "    except:\n"
        "        pass\n"
        "    result = i\n"
        "print(result)"
    )
    # If break was caught, result would be 9 (loop ran to end).
    # Correct behavior: result is the i just before break fired = 2.
    assert "2" in out.logs


def test_continue_inside_try_except_skips_iteration() -> None:
    """`continue` inside a try-with-bare-except advances the loop."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex(
        "out = []\n"
        "for i in range(5):\n"
        "    try:\n"
        "        if i == 2:\n"
        "            continue\n"
        "        out.append(i)\n"
        "    except:\n"
        "        pass\n"
        "print(out)"
    )
    # Pre-fix: continue was caught, all 5 got appended → [0,1,2,3,4].
    # Post-fix: 2 is skipped → [0,1,3,4].
    assert "[0, 1, 3, 4]" in out.logs


def test_real_exception_inside_try_is_still_caught() -> None:
    """Sanity: making control-flow exceptions BaseException doesn't break
    catching of actual user-raised exceptions."""
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    out = ex(
        "def f():\n"
        "    try:\n"
        "        raise ValueError('boom')\n"
        "    except Exception as e:\n"
        "        return f'caught: {e}'\n"
        "print(f())"
    )
    assert "caught: boom" in out.logs


if __name__ == "__main__":
    tests = [
        test_last_expression_value_is_returned,
        test_print_is_captured_into_logs_not_host_stdout,
        test_multistatement_program_runs,
        test_send_variables_injects_into_state,
        test_state_persists_across_calls,
        test_default_builtin_module_is_importable,
        test_unauthorized_import_raises,
        test_additional_authorized_imports_unblocks_module,
        test_dunder_attribute_access_is_blocked,
        test_open_is_not_in_default_builtins,
        test_exec_is_not_in_default_builtins,
        test_syntax_error_raises_interpreter_error,
        test_runtime_error_raises_interpreter_error_with_message,
        test_runaway_loop_is_killed_by_timeout,
        test_undefined_name_raises_interpreter_error,
        test_evaluate_python_code_function_works_standalone,
        test_return_inside_try_except_returns_the_value,
        test_return_inside_try_except_exception_returns_the_value,
        test_break_inside_try_except_exits_the_loop,
        test_continue_inside_try_except_skips_iteration,
        test_real_exception_inside_try_is_still_caught,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")

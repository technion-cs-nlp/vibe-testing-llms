"""
Provides a simple and safe way to execute generated code against test cases.
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .utils import extract_all_code_blocks

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """
    Structured result returned by the sandbox executor.
    """

    passed: bool
    message: str
    stdout: str = ""
    stderr: str = ""
    exception: Optional[str] = None
    status: str = "ok"


def _execute_script(full_script: str, result_queue: "mp.Queue") -> None:
    """
    Run a provided Python script and push execution details into a queue.
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    old_stdin = sys.stdin
    sys.stdin = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            exec(full_script, {})
        result_queue.put(
            {
                "passed": True,
                "message": "All tests passed.",
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "exception": None,
                "status": "ok",
            }
        )
    except AssertionError as exc:
        logger.debug("AssertionError during execution: %s", exc)
        result_queue.put(
            {
                "passed": False,
                "message": f"Test failed: {exc}",
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "exception": repr(exc),
                "status": "assertion_error",
            }
        )
    except EOFError as exc:
        logger.warning(
            "EOFError during execution: Code tried to read from stdin.\nCode:\n%s",
            full_script,
        )
        result_queue.put(
            {
                "passed": False,
                "message": "Execution error: Code tried to read from stdin.",
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "exception": repr(exc),
                "status": "stdin_error",
            }
        )
    except Exception as exc:
        logger.warning(
            f"Exception during execution: {exc}\nCode tail:\n{full_script[-20:]}"
        )
        result_queue.put(
            {
                "passed": False,
                "message": f"Execution error: {type(exc).__name__}: {exc}",
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "exception": repr(exc),
                "status": "error",
            }
        )
    finally:
        sys.stdin = old_stdin


def _get_sandbox_context() -> "mp.context.BaseContext":
    """
    Pick a multiprocessing context for sandbox execution.

    Why this exists:
    - On Linux, `fork` avoids re-importing the whole parent process (and its heavy
      dependencies) in the child. This prevents false timeouts when the main
      evaluation script imports large libraries (e.g., torch/transformers).
    - `spawn` is safer in some environments, but is sensitive to import-time
      side effects and requires proper `if __name__ == "__main__":` usage.

    Override:
        Set env var `VIBE_TESTING_MP_START_METHOD` to one of {"fork", "spawn", "forkserver"}.
    """
    override = os.environ.get("VIBE_TESTING_MP_START_METHOD")
    if override:
        try:
            return mp.get_context(override)
        except ValueError as exc:
            logger.warning(
                "Invalid VIBE_TESTING_MP_START_METHOD=%r; falling back to default. Error: %s",
                override,
                exc,
            )

    # Prefer fork on Linux to avoid re-import overhead in sandbox children.
    if sys.platform.startswith("linux"):
        try:
            return mp.get_context("fork")
        except ValueError:
            # If fork isn't available for some reason, fall back to spawn.
            pass

    return mp.get_context("spawn")


class SandboxExecutor:
    """
    Executes generated code together with assertion-based tests.
    """

    def __init__(self, timeout: int = 15):
        """
        Args:
            timeout (int): Timeout in seconds for sandboxed execution.
                          Defaults to 30 seconds to allow moderately long
                          code solutions while still preventing hangs.
        """
        self.timeout = timeout

    def run(self, generated_code: str, tests: List[str]) -> SandboxResult:
        """
        Execute code plus test assertions within an isolated namespace.
        """
        if not generated_code or not isinstance(generated_code, str):
            return SandboxResult(False, "Evaluation failed: No code generated.")
        if not tests or not isinstance(tests, list):
            return SandboxResult(True, "No tests provided.")

        code_to_run = (
            generated_code  # extract_all_code_blocks(generated_code) or generated_code
        )
        full_script = code_to_run + "\n\n" + "\n".join(tests)

        ctx = _get_sandbox_context()
        result_queue: "mp.Queue" = ctx.Queue()
        process = ctx.Process(target=_execute_script, args=(full_script, result_queue))
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            logger.warning(
                "Execution timed out after %s seconds; terminating sandbox process.",
                self.timeout,
            )
            process.terminate()
            process.join()
            result_queue.close()
            result_queue.join_thread()
            return SandboxResult(
                passed=False,
                message=f"Execution timed out after {self.timeout} seconds.",
                stdout="",
                stderr="",
                exception="TimeoutError",
                status="timeout",
            )

        if result_queue.empty():
            logger.error(
                "Sandbox process exited without returning a result (exitcode=%s).",
                process.exitcode,
            )
            result_queue.close()
            result_queue.join_thread()
            return SandboxResult(
                passed=False,
                message=(
                    "Execution error: No result returned from sandbox process "
                    f"(exitcode={process.exitcode})."
                ),
                stdout="",
                stderr="",
                exception="RuntimeError('empty result')",
                status="error",
            )

        payload = result_queue.get()
        result_queue.close()
        result_queue.join_thread()
        return SandboxResult(**payload)


class CodeExecutor:
    """Backward-compatible wrapper used by existing scripts."""

    def __init__(self, timeout: int = 30):
        self._sandbox = SandboxExecutor(timeout=timeout)

    def evaluate_code(
        self, generated_code: str, tests: List[str], timeout: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Executes generated code against tests and returns pass/fail with message.
        """
        if timeout is not None and timeout != self._sandbox.timeout:
            self._sandbox.timeout = timeout
        result = self._sandbox.run(generated_code, tests)
        return result.passed, result.message

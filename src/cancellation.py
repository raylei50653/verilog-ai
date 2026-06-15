from __future__ import annotations

import subprocess
import threading
import time
from typing import Callable


class PipelineCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self._kill_callbacks: list[Callable[[], None]] = []

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            try:
                process.kill()
            except Exception:
                pass

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise PipelineCancelled("cancelled by user")

    def register_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)
        if self.is_cancelled:
            try:
                process.kill()
            except Exception:
                pass

    def register_kill_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked by kill_processes() alongside process killing.

        Used by Docker sim to also send 'docker kill <name>' when Kill SIM fires.
        """
        with self._lock:
            self._kill_callbacks.append(callback)

    def unregister_kill_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            try:
                self._kill_callbacks.remove(callback)
            except ValueError:
                pass

    def kill_processes(self) -> None:
        """Kill registered subprocesses WITHOUT triggering pipeline cancellation.

        Used by Kill SIM: the subprocess dies (sim fails) but the pipeline keeps
        running to the next step.  Sends SIGKILL (not SIGTERM) so stuck vvp /
        Docker containers die immediately.
        """
        with self._lock:
            processes = list(self._processes)
            callbacks = list(self._kill_callbacks)
        for process in processes:
            try:
                process.kill()
            except Exception:
                pass
        for cb in callbacks:
            try:
                cb()
            except Exception:
                pass

    def unregister_process(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)


def run_cancelable_command(
    cmd: list[str],
    *,
    timeout: float,
    cancel_token: CancellationToken | None = None,
    text: bool = True,
    cwd: str | None = None,
    on_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess:
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    start = time.monotonic()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        cwd=cwd,
        bufsize=1,
    )
    if cancel_token is not None:
        cancel_token.register_process(process)

    try:
        if on_line is not None:
            # Streaming mode: reader threads feed a queue so timeout/cancel check
            # fires every 0.1 s even when the process produces no output.
            import queue as _queue

            stdout_q: _queue.Queue[str | None] = _queue.Queue()
            stderr_lines: list[str] = []

            def _read_stdout() -> None:
                assert process.stdout is not None
                for l in process.stdout:
                    stdout_q.put(l)
                stdout_q.put(None)

            def _read_stderr() -> None:
                assert process.stderr is not None
                for l in process.stderr:
                    stderr_lines.append(l)

            threading.Thread(target=_read_stdout, daemon=True).start()
            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            stdout_lines: list[str] = []
            while True:
                if cancel_token is not None and cancel_token.is_cancelled:
                    process.kill()
                    raise PipelineCancelled("cancelled by user")
                if (time.monotonic() - start) >= timeout:
                    process.kill()
                    stderr_thread.join(timeout=1)
                    raise subprocess.TimeoutExpired(cmd, timeout)
                try:
                    raw_line = stdout_q.get(timeout=0.1)
                except _queue.Empty:
                    continue
                if raw_line is None:
                    break
                stdout_lines.append(raw_line)
                on_line(raw_line.rstrip())

            process.wait()
            stderr_thread.join(timeout=2)
            return subprocess.CompletedProcess(
                cmd, process.returncode,
                "".join(stdout_lines),
                "".join(stderr_lines),
            )

        # Polling mode
        while True:
            if cancel_token is not None and cancel_token.is_cancelled:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise PipelineCancelled("cancelled by user")

            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if (time.monotonic() - start) >= timeout:
                    process.kill()
                    stdout, stderr = process.communicate()
                    raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)

        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    finally:
        if cancel_token is not None:
            cancel_token.unregister_process(process)

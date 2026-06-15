import subprocess
from unittest.mock import patch

import pytest

from src.cancellation import CancellationToken, PipelineCancelled, run_cancelable_command


class _FakeProcess:
    def __init__(self, token: CancellationToken):
        self.token = token
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.communicate_calls = 0

    def communicate(self, timeout=None):
        self.communicate_calls += 1
        if self.communicate_calls == 1:
            self.token._event.set()
        if self.terminated or self.killed:
            self.returncode = -15
            return ("", "")
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = -15 if self.terminated else -9
        return self.returncode


def test_run_cancelable_command_terminates_process_on_cancel():
    token = CancellationToken()
    process = _FakeProcess(token)

    with patch("src.cancellation.subprocess.Popen", return_value=process):
        with pytest.raises(PipelineCancelled):
            run_cancelable_command(["fake"], timeout=5, cancel_token=token)

    assert process.terminated

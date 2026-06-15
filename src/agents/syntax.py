import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src.agents.base import SubAgent, AgentResult
from src.cancellation import CancellationToken, run_cancelable_command


class SyntaxAgent(SubAgent):
    def __init__(self, iverilog_bin: str = "iverilog"):
        self.iverilog_bin = iverilog_bin

    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> AgentResult:
        config = config or {}
        timeout = config.get("timeout", 30)
        cancel_token: CancellationToken | None = config.get("cancel_token")

        start = time.monotonic()
        if shutil.which(self.iverilog_bin) is None:
            return AgentResult(
                pass_=False,
                errors=[{"line": 0, "message": f"Syntax checker '{self.iverilog_bin}' not found on system"}],
                is_infra_failure=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            sources: list[str] = []
            for name, content in rtl_files.items():
                filepath = tmppath / name
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)
                sources.append(str(filepath))

            try:
                result = run_cancelable_command(
                    [self.iverilog_bin, "-g2012", "-o", str(tmppath / "a.out"), *sources],
                    timeout=timeout,
                    cancel_token=cancel_token,
                )
            except subprocess.TimeoutExpired:
                return AgentResult(
                    pass_=False,
                    errors=[{"line": 0, "message": f"Compilation timed out after {timeout}s"}],
                    duration_ms=(time.monotonic() - start) * 1000,
                )

        elapsed = (time.monotonic() - start) * 1000
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        errors = self._parse_iverilog_errors(stderr) if stderr else []
        warnings = self._parse_iverilog_warnings(stderr) if stderr else []

        return AgentResult(
            pass_=result.returncode == 0,
            errors=errors,
            warnings=warnings,
            raw_output=f"{stdout}\n{stderr}",
            duration_ms=elapsed,
        )

    def _parse_iverilog_errors(self, stderr: str) -> list[dict]:
        errors: list[dict] = []
        pattern = re.compile(
            r"(?P<file>[^:]+):(?P<line>\d+):\s*(?P<severity>\w+):\s*(?P<message>.*)"
        )
        for match in pattern.finditer(stderr):
            errors.append({
                "file": match.group("file"),
                "line": int(match.group("line")),
                "severity": match.group("severity"),
                "message": match.group("message").strip(),
            })
        if not errors:
            errors.append({"line": 0, "message": stderr})
        return errors

    def _parse_iverilog_warnings(self, stderr: str) -> list[dict]:
        warnings: list[dict] = []
        pattern = re.compile(r"warning", re.IGNORECASE)
        for line in stderr.splitlines():
            if pattern.search(line):
                warnings.append({"message": line.strip()})
        return warnings

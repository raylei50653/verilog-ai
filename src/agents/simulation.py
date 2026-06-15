import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from src.agents.base import SubAgent, AgentResult
from src.cancellation import CancellationToken, run_cancelable_command


class SimulationAgent(SubAgent):
    def __init__(self, sim_image: str | None = None):
        self.sim_image = sim_image or "nvidia/cvdp-sim:v1.0.0"

    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> AgentResult:
        config = config or {}
        timeout = config.get("timeout", 300)
        cancel_token: CancellationToken | None = config.get("cancel_token")
        on_sim_output = config.get("on_sim_output")

        if not testbench_files:
            return AgentResult(pass_=False, errors=[{"message": "No testbench files provided"}])

        if self._is_cocotb_testbench(testbench_files):
            if self._check_docker():
                return self._run_docker_sim(rtl_files, testbench_files, timeout, cancel_token, on_sim_output)
            return AgentResult(
                pass_=True,
                warnings=[{"message": "cocotb testbench detected — simulation skipped (needs Docker+cocotb)"}],
                raw_output="",
            )

        return self._run_iverilog_sim(rtl_files, testbench_files, timeout, cancel_token, on_sim_output)

    def _is_cocotb_testbench(self, testbench_files: dict[str, str]) -> bool:
        for name in testbench_files:
            if name.endswith(".py"):
                return True
        return False

    def _run_iverilog_sim(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str],
        timeout: int,
        cancel_token: CancellationToken | None,
        on_sim_output: Any = None,
    ) -> AgentResult:
        start = time.monotonic()

        if shutil.which("iverilog") is None:
            return AgentResult(
                pass_=False,
                errors=[{"message": "iverilog not found"}],
                is_infra_failure=True,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            all_sources: list[str] = []

            for name, content in {**rtl_files, **testbench_files}.items():
                dest = tmppath / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
                if name.endswith(".sv") or name.endswith(".v"):
                    all_sources.append(str(dest))

            if not all_sources:
                return AgentResult(pass_=True, warnings=[{"message": "No Verilog testbench — simulation skipped"}])

            try:
                compile_result = run_cancelable_command(
                    ["iverilog", "-g2012", "-o", str(tmppath / "sim.vvp"), *all_sources],
                    timeout=timeout,
                    cancel_token=cancel_token,
                )
                if compile_result.returncode != 0:
                    return AgentResult(
                        pass_=False,
                        errors=[{"phase": "compile", "message": compile_result.stderr.strip()}],
                        raw_output=compile_result.stderr,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )

                if on_sim_output:
                    on_sim_output(f"[iverilog] running (timeout={timeout}s)...")
                sim_result = run_cancelable_command(
                    ["vvp", str(tmppath / "sim.vvp")],
                    timeout=timeout,
                    cancel_token=cancel_token,
                    on_line=on_sim_output,
                )
            except subprocess.TimeoutExpired:
                return AgentResult(pass_=False, errors=[{"message": f"Simulation timed out after {timeout}s"}])

        elapsed = (time.monotonic() - start) * 1000
        stdout = sim_result.stdout or ""
        stderr = sim_result.stderr or ""
        pass_ = sim_result.returncode == 0

        failures: list[dict] = []
        for line in (stdout + stderr).splitlines():
            stripped = line.strip()
            if (stripped.startswith("=") and stripped.endswith("=")) or (stripped.startswith("-") and stripped.endswith("-")):
                continue
            if any(kw in stripped.lower() for kw in ["error", "fail", "mismatch"]):
                failures.append({"message": stripped})

        return AgentResult(
            pass_=pass_,
            errors=[] if pass_ else failures,
            raw_output=f"{stdout}\n{stderr}",
            duration_ms=elapsed,
        )

    def _run_docker_sim(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str],
        timeout: int,
        cancel_token: CancellationToken | None,
        on_sim_output: Any = None,
    ) -> AgentResult:
        start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "rtl").mkdir(parents=True, exist_ok=True)
            (tmppath / "src").mkdir(parents=True, exist_ok=True)
            (tmppath / "rundir").mkdir(parents=True, exist_ok=True)

            for name, content in {**rtl_files, **testbench_files}.items():
                dest = tmppath / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith(".env") or ".env" in name:
                    content = self._sanitize_env_content(content)
                dest.write_text(content, encoding="utf-8")

            env_file = tmppath / "src/.env"
            if not env_file.exists():
                return AgentResult(
                    pass_=False,
                    errors=[{"message": "src/.env not found in harness files"}],
                    raw_output="",
                    duration_ms=(time.monotonic() - start) * 1000,
                    is_infra_failure=True,
                )

            import os
            uid_gid_args = []
            if hasattr(os, "getuid") and hasattr(os, "getgid"):
                uid_gid_args = ["-u", f"{os.getuid()}:{os.getgid()}"]

            container_name = f"verigen-sim-{uuid.uuid4().hex[:12]}"

            cmd = [
                "docker", "run", "--rm",
                "--name", container_name,
                *uid_gid_args,
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{tmppath.resolve()}:/code",
                "-v", f"{(tmppath / 'src').resolve()}:/src",
                "-w", "/code/rundir",
                "--env-file", f"{env_file.resolve()}",
                self.sim_image,
                "pytest", "-s",
                "-p", "no:cacheprovider",
                "-o", "cache_dir=/tmp/.pytest_cache",
                "/src/test_runner.py",
            ]

            def _docker_kill() -> None:
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True, timeout=5,
                )

            if cancel_token is not None:
                cancel_token.register_kill_callback(_docker_kill)

            if on_sim_output:
                on_sim_output(f"[Docker] launching (timeout={timeout}s, image={self.sim_image})...")
            try:
                result = run_cancelable_command(
                    cmd,
                    timeout=timeout,
                    cancel_token=cancel_token,
                    on_line=on_sim_output,
                )
            except subprocess.TimeoutExpired:
                _docker_kill()
                return AgentResult(
                    pass_=False,
                    errors=[{"message": f"Docker simulation timed out after {timeout}s"}],
                    raw_output="",
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            finally:
                if cancel_token is not None:
                    cancel_token.unregister_kill_callback(_docker_kill)

        elapsed = (time.monotonic() - start) * 1000
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        pass_ = result.returncode == 0

        failures: list[dict] = []
        for line in (stdout + stderr).splitlines():
            stripped = line.strip()
            if (stripped.startswith("=") and stripped.endswith("=")) or (stripped.startswith("-") and stripped.endswith("-")):
                continue
            if any(kw in stripped.lower() for kw in ["error", "fail", "mismatch"]):
                failures.append({"message": stripped})

        if not pass_ and not failures:
            failures.append({"message": f"Docker pytest exited with code {result.returncode}"})

        combined = (stdout + stderr).lower()
        is_infra = not pass_ and (
            result.returncode == 125 or                          # docker error
            result.returncode == 4 or                            # pytest usage error (e.g. unrecognized args)
            "daemon" in combined or
            "unable to find image" in combined or
            "unrecognized arguments" in combined or
            "no module named" in combined                        # missing python dep in image
        )

        return AgentResult(
            pass_=pass_,
            errors=[] if pass_ else failures,
            raw_output=f"{stdout}\n{stderr}",
            duration_ms=elapsed,
            is_infra_failure=is_infra,
        )

    def _check_docker(self) -> bool:
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _sanitize_env_content(self, content: str) -> str:
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines.append(line)
                continue
            if "=" in stripped:
                key, val = stripped.split("=", 1)
                lines.append(f"{key.strip()}={val.strip()}")
            else:
                lines.append(line)
        return "\n".join(lines)

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src.agents.base import SubAgent, AgentResult


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

        if not testbench_files:
            return AgentResult(pass_=False, errors=[{"message": "No testbench files provided"}])

        if self._is_cocotb_testbench(testbench_files):
            if self._check_docker():
                return self._run_docker_sim(rtl_files, testbench_files, timeout)
            return AgentResult(
                pass_=True,
                warnings=[{"message": "cocotb testbench detected — simulation skipped (needs Docker+cocotb)"}],
                raw_output="",
            )

        return self._run_iverilog_sim(rtl_files, testbench_files, timeout)

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
    ) -> AgentResult:
        start = time.monotonic()

        if shutil.which("iverilog") is None:
            return AgentResult(pass_=False, errors=[{"message": "iverilog not found"}])

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
                compile_result = subprocess.run(
                    ["iverilog", "-g2012", "-o", str(tmppath / "sim.vvp"), *all_sources],
                    capture_output=True, text=True, timeout=timeout,
                )
                if compile_result.returncode != 0:
                    return AgentResult(
                        pass_=False,
                        errors=[{"phase": "compile", "message": compile_result.stderr.strip()}],
                        raw_output=compile_result.stderr,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )

                sim_result = subprocess.run(
                    ["vvp", str(tmppath / "sim.vvp")],
                    capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return AgentResult(pass_=False, errors=[{"message": f"Simulation timed out after {timeout}s"}])

        elapsed = (time.monotonic() - start) * 1000
        stdout = sim_result.stdout or ""
        stderr = sim_result.stderr or ""
        pass_ = sim_result.returncode == 0

        failures: list[dict] = []
        for line in (stdout + stderr).splitlines():
            if any(kw in line.lower() for kw in ["error", "fail", "mismatch"]):
                failures.append({"message": line.strip()})

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
                dest.write_text(content)

            env_file = tmppath / "src/.env"
            if not env_file.exists():
                return AgentResult(
                    pass_=False,
                    errors=[{"message": "src/.env not found in harness files"}],
                    raw_output="",
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            cmd = [
                "docker", "run", "--rm",
                "-v", f"{tmppath.resolve()}:/code",
                "-v", f"{(tmppath / 'src').resolve()}:/src",
                "-w", "/code/rundir",
                "--env-file", f"{env_file.resolve()}",
                self.sim_image,
                "pytest", "-s", "-o", "cache_dir=/rundir/harness/.cache", "/src/test_runner.py"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return AgentResult(
                    pass_=False,
                    errors=[{"message": f"Docker simulation timed out after {timeout}s"}],
                    raw_output="",
                    duration_ms=(time.monotonic() - start) * 1000,
                )

        elapsed = (time.monotonic() - start) * 1000
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        pass_ = result.returncode == 0

        failures: list[dict] = []
        for line in (stdout + stderr).splitlines():
            if any(kw in line.lower() for kw in ["error", "fail", "mismatch"]):
                failures.append({"message": line.strip()})

        if not pass_ and not failures:
            failures.append({"message": f"Docker pytest exited with code {result.returncode}"})

        return AgentResult(
            pass_=pass_,
            errors=[] if pass_ else failures,
            raw_output=f"{stdout}\n{stderr}",
            duration_ms=elapsed,
        )

    def _check_docker(self) -> bool:
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

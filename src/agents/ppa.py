import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src.agents.base import SubAgent, AgentResult


class PPAAgent(SubAgent):
    def __init__(self, yosys_bin: str = "yosys"):
        self.yosys_bin = yosys_bin

    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> AgentResult:
        config = config or {}
        timeout = config.get("timeout", 120)
        top_module = config.get("top_module", "top")
        target_metrics: dict[str, Any] = config.get("target_metrics", {})

        start = time.monotonic()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            sources: list[str] = []
            for name, content in rtl_files.items():
                filepath = tmppath / name
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)
                sources.append(str(filepath))

            yosys_script = tmppath / "synth.ys"
            read_cmds = "\n".join(f"read_verilog -sv {s}" for s in sources)
            yosys_script.write_text(
                f"""\
{read_cmds}
hierarchy -top {top_module}
proc
opt
fsm
opt
memory
opt
techmap
opt
abc
opt
clean
stat -json {tmppath / "stat.json"}
"""
            )

            try:
                result = subprocess.run(
                    [self.yosys_bin, "-s", str(yosys_script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(tmppath),
                )
            except subprocess.TimeoutExpired:
                return AgentResult(
                    pass_=False,
                    errors=[{"message": f"Synthesis timed out after {timeout}s"}],
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            except FileNotFoundError:
                return AgentResult(
                    pass_=False,
                    errors=[{"message": "yosys not found in PATH"}],
                    duration_ms=(time.monotonic() - start) * 1000,
                )

        elapsed = (time.monotonic() - start) * 1000

        if result.returncode != 0:
            return AgentResult(
                pass_=False,
                errors=[{"message": result.stderr.strip()}],
                raw_output=result.stderr,
                duration_ms=elapsed,
            )

        stat_file = tmppath / "stat.json"
        metrics = self._parse_stat_json(stat_file) if stat_file.exists() else {}

        target_met: dict[str, bool] = {}
        for key, target in target_metrics.items():
            if key in metrics and isinstance(metrics[key], (int, float)):
                target_met[key] = metrics[key] <= target

        return AgentResult(
            pass_=True,
            metrics=metrics,
            raw_output=result.stdout,
            warnings=[],
            duration_ms=elapsed,
        )

    def _parse_stat_json(self, stat_path: Path) -> dict[str, Any]:
        import json

        try:
            data = json.loads(stat_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

        modules = data.get("modules", {})
        design = data.get("design", {})

        area = 0
        num_cells = 0
        for mod_name, mod_data in modules.items():
            if mod_data.get("num_cells_by_type"):
                for cell_type, count in mod_data["num_cells_by_type"].items():
                    num_cells += count

        return {
            "area": num_cells,
            "area_unit": "cells",
            "num_modules": len(modules),
            "num_wires": design.get("num_wires", 0),
            "num_cells": num_cells,
        }

    @staticmethod
    def _parse_stat_text(text: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        cells_match = re.search(r"Number of cells:\s*(\d+)", text)
        if cells_match:
            metrics["num_cells"] = int(cells_match.group(1))
            metrics["area"] = int(cells_match.group(1))

        wires_match = re.search(r"Number of wires:\s*(\d+)", text)
        if wires_match:
            metrics["num_wires"] = int(wires_match.group(1))

        return metrics

import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from src.agents.base import SubAgent, AgentResult


_UTIL_PATTERNS: list[tuple[str, str]] = [
    ("luts",         r"\|\s*Slice LUTs\s*\|\s*(\d+)"),
    ("luts_logic",   r"\|\s*LUT as Logic\s*\|\s*(\d+)"),
    ("luts_memory",  r"\|\s*LUT as Memory\s*\|\s*(\d+)"),
    ("registers",    r"\|\s*Slice Registers\s*\|\s*(\d+)"),
    ("dsps",         r"\|\s*DSPs\s*\|\s*(\d+)"),
    ("brams",        r"\|\s*Block RAM Tile\s*\|\s*(\d+)"),
]

_TIMING_RE = re.compile(
    r"WNS\(ns\).*?\n\s*[-]+.*?\n\s*([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+(\d+)",
    re.DOTALL,
)
_POWER_RE = re.compile(r"Total On-Chip Power \(W\)\s*\|\s*([\d.]+)")


def _parse_utilization(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in _UTIL_PATTERNS:
        m = re.search(pattern, text)
        if m:
            out[key] = int(m.group(1))
    return out


def _parse_timing(text: str) -> dict[str, Any]:
    m = _TIMING_RE.search(text)
    if not m:
        return {}
    return {
        "wns_ns": float(m.group(1)),
        "tns_ns": float(m.group(2)),
        "failing_endpoints": int(m.group(3)),
        "timing_met": int(m.group(3)) == 0,
    }


def _parse_power(text: str) -> dict[str, Any]:
    m = _POWER_RE.search(text)
    return {"total_power_w": float(m.group(1))} if m else {}


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _to_win_path(path: str | Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        capture_output=True, text=True, timeout=5,
    )
    return result.stdout.strip()


def _win_temp_base() -> Path:
    """Return WSL-side path to the Windows user TEMP directory.

    Uses cmd.exe first so %TEMP% expands to the 8.3 short path (no spaces),
    which Vivado 2018.x requires.  Suppress stderr to avoid a UnicodeDecodeError
    from the Chinese-locale UNC-path warning cmd.exe prints when CWD is a WSL path.
    """
    try:
        r1 = subprocess.run(
            ["cmd.exe", "/c", "echo %TEMP%"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
        )
        win_temp = r1.stdout.decode("ascii", errors="ignore").strip()
        if win_temp:
            r2 = subprocess.run(
                ["wslpath", "-u", win_temp],
                capture_output=True, text=True, timeout=5,
            )
            p = Path(r2.stdout.strip())
            if p.exists():
                return p
    except Exception:
        pass
    # PowerShell fallback — returns long path (may contain spaces, works on 2019+).
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "[System.Environment]::GetEnvironmentVariable('TEMP','User')"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=8,
        )
        win_temp = r.stdout.decode("utf-8", errors="ignore").strip()
        if win_temp:
            r2 = subprocess.run(
                ["wslpath", "-u", win_temp],
                capture_output=True, text=True, timeout=5,
            )
            p = Path(r2.stdout.strip())
            if p.exists():
                return p
    except Exception:
        pass
    return Path("/mnt/c/Windows/Temp")


def find_vivado_wsl() -> str | None:
    """
    Scan common Vivado install locations visible from WSL2.
    Returns the WSL path to vivado.bat, or None if not found.
    """
    # (base_dir, subpath_to_bat)
    search_configs = [
        # C:\Xilinx\Vivado\<version>\bin\vivado.bat
        (Path("/mnt/c/Xilinx/Vivado"),  None),
        (Path("/mnt/d/Xilinx/Vivado"),  None),
        # C:\AMD\Vivado\<version>\bin\vivado.bat
        (Path("/mnt/c/AMD/Vivado"),      None),
        (Path("/mnt/d/AMD/Vivado"),      None),
        # C:\AMDDesignTools\<version>\Vivado\bin\vivado.bat
        (Path("/mnt/c/AMDDesignTools"),  "Vivado"),
        (Path("/mnt/d/AMDDesignTools"),  "Vivado"),
    ]
    for base, subdir in search_configs:
        if not base.exists():
            continue
        try:
            versions = sorted(base.iterdir(), reverse=True)
        except PermissionError:
            continue
        for v in versions:
            if subdir:
                bat = v / subdir / "bin" / "vivado.bat"
            else:
                bat = v / "bin" / "vivado.bat"
            if bat.exists():
                return str(bat)
    return None


def _build_cmd(vivado_bin: str, tcl_arg: str, wsl_mode: bool) -> list[str]:
    """
    Build the subprocess command for Vivado batch mode.

    vvgl.exe — Vivado's underlying launcher — requires 'vivado' as the first arg.
    vivado.bat — standard launcher — needs cmd.exe /c in WSL2.
    vivado / vivado.exe — call directly.
    """
    base = Path(vivado_bin).name.lower()
    batch_args = ["-mode", "batch", "-source", tcl_arg, "-nojournal", "-nolog"]

    if base == "vvgl.exe":
        return [vivado_bin, "vivado"] + batch_args
    if wsl_mode and vivado_bin.endswith(".bat"):
        win_bat = _to_win_path(vivado_bin)
        return ["cmd.exe", "/c", win_bat] + batch_args
    return [vivado_bin] + batch_args


class VivadoPPAAgent(SubAgent):
    def __init__(self, vivado_bin: str = "vivado", part: str = "xc7a35tcpg236-1"):
        self.vivado_bin = vivado_bin
        self.part = part

    def _make_temp_dir(self, wsl_mode: bool) -> Path:
        if wsl_mode:
            base = _win_temp_base()
            d = base / f"verigen_{uuid.uuid4().hex[:8]}"
            d.mkdir(parents=True, exist_ok=True)
            return d
        import tempfile
        return Path(tempfile.mkdtemp())

    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> AgentResult:
        config = config or {}
        timeout = config.get("timeout", 600)
        top_module = config.get("top_module", "top")
        part = config.get("part", self.part)

        vivado_bin = self.vivado_bin
        wsl_mode = _is_wsl() and vivado_bin.startswith("/mnt/")

        start = time.monotonic()
        tmppath = self._make_temp_dir(wsl_mode)
        result = None

        try:
            sources: list[str] = []
            for name, content in rtl_files.items():
                fp = tmppath / name
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
                sources.append(str(fp))

            if testbench_files:
                tb_dir = tmppath / "tb"
                tb_dir.mkdir(parents=True, exist_ok=True)
                for name, content in testbench_files.items():
                    fp = tb_dir / name
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(content, encoding="utf-8")

            if wsl_mode:
                src_paths  = [_to_win_path(s) for s in sources]
                util_rpt   = _to_win_path(tmppath / "util.rpt")
                timing_rpt = _to_win_path(tmppath / "timing.rpt")
                power_rpt  = _to_win_path(tmppath / "power.rpt")
            else:
                src_paths  = sources
                util_rpt   = str(tmppath / "util.rpt")
                timing_rpt = str(tmppath / "timing.rpt")
                power_rpt  = str(tmppath / "power.rpt")

            read_cmds = "\n".join(f"read_verilog -sv {{{s}}}" for s in src_paths)
            tcl = tmppath / "synth.tcl"
            tcl.write_text(
                f"""\
{read_cmds}
synth_design -top {top_module} -part {part} -mode out_of_context
opt_design
report_utilization -file {{{util_rpt}}}
report_timing_summary -file {{{timing_rpt}}}
report_power -file {{{power_rpt}}}
""",
                encoding="utf-8",
            )

            tcl_arg = _to_win_path(tcl) if wsl_mode else str(tcl)
            cmd = _build_cmd(vivado_bin, tcl_arg, wsl_mode)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=False,
                    timeout=timeout,
                    cwd=str(tmppath),
                )
                # Smart decode function for local Chinese encoding (CP950/GBK)
                def decode_bytes(data: bytes) -> str:
                    if not data:
                        return ""
                    for enc in ("utf-8", "cp950", "gbk"):
                        try:
                            return data.decode(enc)
                        except UnicodeDecodeError:
                            continue
                    return data.decode("utf-8", errors="replace")

                stdout_str = decode_bytes(result.stdout)
                stderr_str = decode_bytes(result.stderr)
            except subprocess.TimeoutExpired:
                return AgentResult(
                    pass_=False,
                    errors=[{"message": f"Vivado timed out after {timeout}s"}],
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            except FileNotFoundError:
                return AgentResult(
                    pass_=False,
                    errors=[{"message": f"Vivado not found: {vivado_bin}"}],
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            elapsed = (time.monotonic() - start) * 1000

            if result.returncode != 0:
                return AgentResult(
                    pass_=False,
                    errors=[{"message": stderr_str.strip() or stdout_str[-2000:]}],
                    raw_output=stdout_str,
                    duration_ms=elapsed,
                )

            metrics: dict[str, Any] = {"part": part, "top_module": top_module}
            for fname, parser in [
                ("util.rpt",   _parse_utilization),
                ("timing.rpt", _parse_timing),
                ("power.rpt",  _parse_power),
            ]:
                rpt = tmppath / fname
                if rpt.exists():
                    metrics.update(parser(rpt.read_text(encoding="utf-8", errors="replace")))

        finally:
            shutil.rmtree(tmppath, ignore_errors=True)

        return AgentResult(
            pass_=True,
            metrics=metrics,
            raw_output=stdout_str if result else "",
            duration_ms=(time.monotonic() - start) * 1000,
        )

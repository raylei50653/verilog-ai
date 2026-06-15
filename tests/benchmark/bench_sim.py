"""Benchmark simulation infrastructure throughput.

Uses a simple counter module + testbench to measure:
1. Docker container startup + cocotb runtime
2. Sim per-problem overhead (file mounting, etc.)
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from src.cvdp.loader import CVDPDataset
from src.agents.simulation import SimulationAgent

load_dotenv()

SIM_IMAGE = os.getenv("OSS_SIM_IMAGE", "nvidia/cvdp-sim:v1.0.0")
SUBSET = "nonagentic_no_commercial"

RTL_TEMPLATE = """module {name} (
    input  wire        clk,
    input  wire        rst,
    input  wire [3:0]  data_in,
    output reg  [3:0]  data_out
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= 4'b0;
        else
            data_out <= data_in;
    end
endmodule
"""

TB_TEMPLATE = """import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock

@cocotb.test()
async def basic_test(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    dut.rst.value = 1
    await Timer(20, unit="ns")
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    dut.data_in.value = 5
    await RisingEdge(dut.clk)
    assert dut.data_out.value == 5, f"Expected 5, got {dut.data_out.value}"
"""

TB_RUNNER = """import os
from cocotb_tools.runner import get_runner

def test_runner():
    runner = get_runner(os.getenv("SIM", "icarus"))
    runner.build(
        sources=os.getenv("VERILOG_SOURCES").split(),
        hdl_toplevel=os.getenv("TOPLEVEL"),
        always=True, clean=True, verbose=True,
        timescale=("1ns", "1ns"), log_file="sim.log"
    )
    runner.test(
        hdl_toplevel=os.getenv("TOPLEVEL"),
        test_module=os.getenv("MODULE"), waves=False
    )
"""

ENV_TEMPLATE = """VERILOG_SOURCES={rtl_list}
TOPLEVEL_LANG=verilog
SIM=icarus
TOPLEVEL={toplevel}
MODULE=test_{toplevel}
WAVE=0
"""


def check_docker() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def run_sim_docker(toplevel: str, rtl_content: str, tb_content: str, timeout: int = 120) -> dict:
    work = Path(tempfile.mkdtemp(prefix="benchsim_"))
    try:
        rtl_dir = work / "rtl"
        rtl_dir.mkdir(parents=True)
        rtl_file = rtl_dir / f"{toplevel}.v"
        rtl_file.write_text(rtl_content)

        src_dir = work / "src"
        src_dir.mkdir(parents=True)
        (src_dir / f"test_{toplevel}.py").write_text(tb_content)
        (src_dir / "test_runner.py").write_text(TB_RUNNER)
        (src_dir / ".env").write_text(
            ENV_TEMPLATE.format(rtl_list=f"rtl/{toplevel}.v", toplevel=toplevel)
        )

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{work}:/work",
            "-w", "/work/src",
            SIM_IMAGE,
            "pytest", "-s", "-v", "--tb=line", "--timeout=60",
        ]

        t0 = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - t0
        passed = result.returncode == 0

        if not passed:
            last_lines = result.stdout.strip().split("\n")[-5:]
            error = "\n".join(last_lines)
        else:
            error = ""

        return {"pass": passed, "elapsed": elapsed, "error": error}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    has_docker = check_docker()
    if not has_docker:
        print("Docker not available. Exiting.")
        return

    print(f"Docker: OK | SIM_IMAGE: {SIM_IMAGE}")
    print()

    ds = CVDPDataset(subset=SUBSET)

    # Group problems by category, pick 2 per category for timing
    categories: dict[str, list] = {}
    for p in ds:
        for c in p.cids:
            categories.setdefault(c, []).append(p)

    sample_count = min(2, len(list(ds)))
    problems = list(ds)[:sample_count * 4]  # sample various problems

    if not problems:
        problems = list(ds)[:8]

    print(f"Testing simulation infra with {len(problems)} problems...")
    print(f"{'#':>3} {'Problem':<50} {'Time(s)':>8} {'Result':>8}")
    print("-" * 72)

    times = []
    passed = 0
    for i, p in enumerate(problems, 1):
        toplevel = "sim_bench_top"
        rtl = RTL_TEMPLATE.format(name=toplevel)
        tb = TB_TEMPLATE
        result = run_sim_docker(toplevel, rtl, tb, timeout=60)
        times.append(result["elapsed"])
        if result["pass"]:
            passed += 1
        status = "PASS" if result["pass"] else "FAIL"
        print(f"{i:>3} {p.id:<50} {result['elapsed']:>8.2f} {status:>8}")

    avg = sum(times) / len(times) if times else 0
    mn = min(times) if times else 0
    mx = max(times) if times else 0
    print("-" * 72)
    print(f"Results: {passed}/{len(times)} passed")
    print(f"Per-sim: min={mn:.2f}s  max={mx:.2f}s  avg={avg:.2f}s")
    print(f"Estimated 302 problems: {avg * 302:.0f}s = {avg * 302 / 60:.1f}min serial")
    print(f"  (actual varies by RTL complexity — this is best-case)")
    print()
    print(f"With LLM ~15s gen + {avg:.1f}s sim = ~{15+avg:.1f}s per problem")
    print(f"  302 problems serial: ~{(15+avg)*302/60:.0f}min")


if __name__ == "__main__":
    main()

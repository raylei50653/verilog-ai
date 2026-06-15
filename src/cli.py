import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from src.llm import create_backend
from src.pipeline import TrialRunner
from src.cvdp.loader import CVDPDataset, download_dataset
from src.cvdp.scoring import ProblemScore, BenchmarkReport


def load_env():
    load_dotenv()


@click.group()
def main():
    pass


@main.command()
@click.option("--spec", required=True, help="Spec text or path to spec file, or 'cvdp:<problem_id>'.")
@click.option("--params", default="{}", help="JSON design parameters.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--temperature", default=0.2, type=float)
@click.option("--max-tokens", default=8192, type=int)
@click.option("--max-retries", default=3, type=int)
@click.option("--model", default=None, help="Override default model.")
@click.option("--baseline", is_flag=True, help="Compatibility flag. The baseline workflow is always used.")
@click.option("--vivado", is_flag=True, help="Run Vivado synthesis analysis on passing trials.")
@click.option("--part", default=None, help="Xilinx part for Vivado analysis (default: VIVADO_PART env).")
def run(spec, params, verbose, temperature, max_tokens, max_retries, model, baseline, vivado, part):
    load_env()

    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        click.echo(f"Invalid JSON params: {params}", err=True)
        return

    problem_id = "custom"
    context_files = None
    tb_files = None
    system_msg = None

    if spec.startswith("cvdp:"):
        problem_id = spec.split(":", 1)[1]
        cvdp = CVDPDataset()
        problem = cvdp.get_by_id(problem_id)
        if problem is None:
            click.echo(f"Problem not found: {problem_id}", err=True)
            return
        spec = problem.prompt
        context_files = dict(problem.context)
        tb_files = problem.get_testbench_files()
        click.echo(f"Problem: {problem.id} ({problem.difficulty})")

    elif os.path.exists(spec):
        spec = Path(spec).read_text()

    backend = create_backend(model=model)
    vivado_agent = None
    if vivado:
        from src.agents.vivado_ppa import VivadoPPAAgent, find_vivado_wsl
        vbin = os.getenv("VIVADO_BIN") or find_vivado_wsl() or "vivado"
        vpart = part or os.getenv("VIVADO_PART", "xc7a35tcpg236-1")
        vivado_agent = VivadoPPAAgent(vivado_bin=vbin, part=vpart)
        click.echo(f"Vivado: {vbin}  Part: {vpart}")
    runner = TrialRunner(backend, vivado_agent=vivado_agent)

    code, score = runner.run_trial(
        spec=spec,
        problem_id=problem_id,
        params=parsed_params,
        context_files=context_files,
        testbench_files=tb_files,
        system_message=system_msg,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        verbose=verbose,
        baseline_mode=True,
    )

    click.echo(f"\nResult: {'PASS' if score.pass_ else 'FAIL'} ({score.duration_ms:.0f}ms)")
    click.echo(f"  Syntax:    {'PASS' if score.syntax_pass else 'FAIL'}")
    click.echo(f"  Simulation: {'PASS' if score.simulation_pass else 'FAIL'}")
    if vivado_agent and score.vivado_metrics:
        click.echo(f"  Vivado:    PASS")
        for key, val in score.vivado_metrics.items():
            click.echo(f"    {key}: {val}")
    if score.errors:
        click.echo(f"  Errors: {len(score.errors)}")

    click.echo(f"\n{code}")


@main.command()
@click.option("--dataset", default="nonagentic_no_commercial", help="CVDP dataset subset.")
@click.option("--samples", "-n", default=10, type=int, help="Samples per problem.")
@click.option("--pass-k", default="1", help="Comma-separated k values.")
@click.option("--max-problems", type=int, default=0, help="Limit number of problems (0=all).")
@click.option("--category", "-c", multiple=True, help="Filter by category (repeatable).")
@click.option("--difficulty", "-d", multiple=True, help="Filter by difficulty (repeatable).")
@click.option("--output", "-o", default="results/benchmark.json", help="Output file.")
@click.option("--temperature", default=0.2, type=float)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Just count problems, don't call LLM.")
@click.option("--model", default=None, help="Override default model.")
@click.option("--baseline", is_flag=True, help="Compatibility flag. The baseline workflow is always used.")
def benchmark(dataset, samples, pass_k, max_problems, category, difficulty,
              output, temperature, verbose, dry_run, model, baseline):
    load_env()

    k_values = [int(k.strip()) for k in pass_k.split(",")]
    cvdp = CVDPDataset(subset=dataset)

    problems = list(cvdp)
    if category:
        filtered = []
        for cid in category:
            filtered.extend(cvdp.filter_by_category(cid))
        seen = {p.id for p in filtered}
        problems = [p for p in problems if p.id in seen]
    if difficulty:
        filtered = []
        for d in difficulty:
            filtered.extend(cvdp.filter_by_difficulty(d))
        seen = {p.id for p in filtered}
        problems = [p for p in problems if p.id in seen]
    if max_problems > 0:
        problems = problems[:max_problems]

    if dry_run:
        click.echo(f"Dry run: {len(problems)} problems × {samples} samples = {len(problems) * samples} trials")
        for p in problems:
            tb = "TB" if p.has_testbench() else "no-TB"
            click.echo(f"  {p.id} [{', '.join(p.categories)}] {tb}")
        return

    backend = create_backend(model=model)
    runner = TrialRunner(backend)
    problem_scores: dict[str, ProblemScore] = {}

    for pi, problem in enumerate(problems):
        click.echo(f"\n[{pi+1}/{len(problems)}] {problem.id} ({problem.difficulty})")
        ps = ProblemScore(problem_id=problem.id, categories=problem.categories)

        for si in range(samples):
            click.echo(f"  sample {si+1}/{samples}... ", nl=False)

            code, score = runner.run_trial(
                spec=problem.prompt,
                problem_id=problem.id,
                context_files=dict(problem.context) if problem.context else None,
                testbench_files=problem.get_testbench_files() if problem.has_testbench() else None,
                temperature=temperature,
                verbose=False,
                baseline_mode=True,
            )
            score.trial_index = si
            ps.trials.append(score)

            status = "PASS" if score.pass_ else "FAIL"
            click.echo(f"{status} ({score.duration_ms:.0f}ms)")

        problem_scores[problem.id] = ps

    report = BenchmarkReport.from_scores(dataset, problem_scores, samples, k_values)
    click.echo(f"\n{report.summary()}")


    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    click.echo(f"\nSaved to {output_path}")


@main.command()
@click.option("--subset", default="nonagentic_no_commercial", help="CVDP subset to download.")
def download(subset):
    download_dataset(subset=subset)


@main.command()
@click.option("--dataset", default="nonagentic_no_commercial", help="CVDP dataset subset.")
def info(dataset):
    cvdp = CVDPDataset(subset=dataset)
    click.echo(f"Subset: {cvdp.subset}")
    click.echo(f"Problems: {len(cvdp)}")
    click.echo(f"Categories: {cvdp.get_categories()}")

    by_cats: dict[str, int] = {}
    for p in cvdp:
        for c in p.cids:
            by_cats[c] = by_cats.get(c, 0) + 1
    click.echo("\nBy CID:")
    for cid, count in sorted(by_cats.items()):
        click.echo(f"  {cid}: {count}")

    with_tb = sum(1 for p in cvdp if p.has_testbench())
    click.echo(f"\nWith testbench: {with_tb}/{len(cvdp)}")


@main.command(name="vivado-detect")
def vivado_detect():
    """Scan common Windows install paths and print the detected Vivado binary (WSL2 only)."""
    from src.agents.vivado_ppa import find_vivado_wsl
    found = find_vivado_wsl()
    if found:
        click.echo(f"Found: {found}")
        click.echo(f"\nAdd to .env:\n  VIVADO_BIN={found}")
    else:
        click.echo("Vivado not found in common Windows install paths.")
        click.echo("Set VIVADO_BIN manually, e.g.:")
        click.echo("  VIVADO_BIN=/mnt/c/AMDDesignTools/2025.2/Vivado/bin/vivado.bat")
        click.echo("  VIVADO_BIN=/mnt/c/AMDDesignTools/2025.2/Vivado/bin/unwrapped/win64.o/vvgl.exe")


@main.command(name="vivado-analyze")
@click.option("--rtl", "rtl_path", default=None, help="Path to RTL file (.v/.sv).")
@click.option("--trial-id", default=None, help="Trial ID to look up RTL from data/outputs/.")
@click.option("--top", default=None, help="Top module name (default: infer from filename).")
@click.option("--part", default=None, help="Xilinx part number (default: VIVADO_PART env or xc7a35tcpg236-1).")
@click.option("--vivado-bin", default=None, help="Path to vivado binary (default: VIVADO_BIN env or auto-detect).")
@click.option("--timeout", default=600, type=int, help="Synthesis timeout in seconds.")
@click.option("--output", "-o", default=None, help="Write JSON results to this file.")
def vivado_analyze(rtl_path, trial_id, top, part, vivado_bin, timeout, output):
    """Run Vivado synthesis analysis on an RTL file or stored trial (standalone, no pipeline)."""
    load_env()

    if not rtl_path and not trial_id:
        click.echo("Error: provide --rtl <file> or --trial-id <id>", err=True)
        raise SystemExit(1)
    if rtl_path and trial_id:
        click.echo("Error: --rtl and --trial-id are mutually exclusive", err=True)
        raise SystemExit(1)

    rtl_files: dict[str, str] = {}

    if trial_id:
        from src.mcp.server import MCPServer
        mcp = MCPServer()
        out_dir = mcp.db_path.parent.parent / "outputs" / trial_id
        if not out_dir.exists():
            click.echo(f"Error: trial output directory not found: {out_dir}", err=True)
            raise SystemExit(1)
        found = list(out_dir.rglob("*.sv")) + list(out_dir.rglob("*.v"))
        if not found:
            click.echo(f"Error: no .sv/.v files found in {out_dir}", err=True)
            raise SystemExit(1)
        for f in found:
            rtl_files[f.name] = f.read_text()
        if not top:
            top = found[0].stem
        click.echo(f"Trial: {trial_id}  files: {[f.name for f in found]}")
    else:
        p = Path(rtl_path)
        if not p.exists():
            click.echo(f"Error: file not found: {rtl_path}", err=True)
            raise SystemExit(1)
        rtl_files[p.name] = p.read_text()
        if not top:
            top = p.stem

    resolved_part = part or os.getenv("VIVADO_PART", "xc7a35tcpg236-1")

    # Resolve vivado binary: CLI arg > env var > auto-detect
    if vivado_bin:
        resolved_bin = vivado_bin
    elif os.getenv("VIVADO_BIN"):
        resolved_bin = os.getenv("VIVADO_BIN")
    else:
        from src.agents.vivado_ppa import find_vivado_wsl
        resolved_bin = find_vivado_wsl() or "vivado"
        if resolved_bin != "vivado":
            click.echo(f"Auto-detected Vivado: {resolved_bin}")

    click.echo(f"Part:   {resolved_part}")
    click.echo(f"Top:    {top}")
    click.echo(f"Vivado: {resolved_bin}")
    click.echo("Running synthesis... (this may take several minutes)")

    from src.agents.vivado_ppa import VivadoPPAAgent
    agent = VivadoPPAAgent(vivado_bin=resolved_bin, part=resolved_part)
    result = agent.execute(
        rtl_files=rtl_files,
        config={"top_module": top, "part": resolved_part, "timeout": timeout},
    )

    click.echo(f"\nStatus: {'PASS' if result.pass_ else 'FAIL'}  ({result.duration_ms / 1000:.1f}s)")

    if result.pass_:
        m = result.metrics
        click.echo("\n--- Utilization ---")
        for key in ("luts", "luts_logic", "luts_memory", "registers", "dsps", "brams"):
            if key in m:
                click.echo(f"  {key:<14}: {m[key]}")
        click.echo("\n--- Timing ---")
        for key in ("wns_ns", "tns_ns", "failing_endpoints", "timing_met"):
            if key in m:
                click.echo(f"  {key:<20}: {m[key]}")
        click.echo("\n--- Power ---")
        if "total_power_w" in m:
            click.echo(f"  total_power_w       : {m['total_power_w']}")
    else:
        for err in result.errors:
            click.echo(f"  Error: {err.get('message', err)}", err=True)

    if output:
        import json as _json
        out = {
            "pass": result.pass_,
            "duration_s": result.duration_ms / 1000,
            "metrics": result.metrics,
            "errors": result.errors,
        }
        if trial_id:
            out["trial_id"] = trial_id
        if rtl_path:
            out["rtl_path"] = rtl_path
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(_json.dumps(out, indent=2))
        click.echo(f"\nSaved to {output}")


@main.command()
def tui():
    """Start the interactive Terminal User Interface (TUI)."""
    from src.tui.app import VeriGenTUI
    app = VeriGenTUI()
    app.run()


if __name__ == "__main__":
    main()

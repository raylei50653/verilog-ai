import json
import os
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from src.llm import create_backend
from src.pipeline import TrialRunner
from src.cvdp.loader import CVDPDataset, download_dataset
from src.cvdp.scoring import ProblemScore, TrialScore, BenchmarkReport


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
@click.option("--reuse-rtl", is_flag=True, help="Enable RTL reuse across problems.")
def run(spec, params, verbose, temperature, max_tokens, max_retries, model, reuse_rtl):
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
    runner = TrialRunner(backend)

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
        enable_rtl_reuse=reuse_rtl,
        verbose=verbose,
    )

    click.echo(f"\nResult: {'PASS' if score.pass_ else 'FAIL'} ({score.duration_ms:.0f}ms)")
    click.echo(f"  Syntax:    {'PASS' if score.syntax_pass else 'FAIL'}")
    click.echo(f"  Simulation: {'PASS' if score.simulation_pass else 'FAIL'}")
    if score.ppa_metrics:
        click.echo(f"  PPA: {json.dumps(score.ppa_metrics)}")
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
@click.option("--reuse-rtl", is_flag=True, help="Enable RTL reuse across problems.")
def benchmark(dataset, samples, pass_k, max_problems, category, difficulty,
              output, temperature, verbose, dry_run, model, reuse_rtl):
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
                testbench_files=problem.get_testbench_files() if problem.has_testbench() else None,
                temperature=temperature,
                enable_rtl_reuse=reuse_rtl,
                verbose=False,
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

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    click.echo(f"\nReport saved to {output_path}")


@main.command()
@click.option("--problem-id", required=True, help="CVDP problem ID.")
@click.option("--trials", default=50, type=int, help="Number of Optuna trials.")
@click.option("--objective", default="area", help="Optimization objective.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--model", default=None, help="Override default model.")
@click.option("--reuse-rtl", is_flag=True, help="Enable RTL reuse across problems.")
def optimize(problem_id, trials, objective, verbose, model, reuse_rtl):
    load_env()
    storage_uri = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna/optuna.db")

    cvdp = CVDPDataset()
    problem = cvdp.get_by_id(problem_id)
    if problem is None:
        click.echo(f"Problem not found: {problem_id}", err=True)
        return

    backend = create_backend(model=model)
    runner = TrialRunner(backend)

    from src.optimizer.optuna_runner import OptunaRunner
    opt_runner = OptunaRunner(
        trial_runner=runner,
        storage_uri=storage_uri,
    )

    click.echo(f"Starting Optuna optimization for {problem.id}...")
    click.echo(f"  Objective: {objective}")
    click.echo(f"  Trials:    {trials}")
    click.echo(f"  Storage:   {storage_uri}")

    study = opt_runner.optimize_problem(
        problem=problem,
        n_trials=trials,
        objective_metric=objective,
        enable_rtl_reuse=reuse_rtl,
        verbose=verbose,
    )

    click.echo("\nOptimization Completed!")
    try:
        best_trial = study.best_trial
        click.echo(f"Best Trial Value: {best_trial.value}")
        click.echo(f"Best Parameters:   {best_trial.params}")
    except ValueError:
        click.echo("No successful trials found.")


@main.command(name="optimize-all")
@click.option("--dataset", default="nonagentic_no_commercial", help="CVDP dataset subset.")
@click.option("--trials-per-problem", default=10, type=int, help="Number of trials per problem.")
@click.option("--objective", default="area", help="Optimization objective.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--model", default=None, help="Override default model.")
@click.option("--reuse-rtl", is_flag=True, help="Enable RTL reuse across problems.")
def optimize_all(dataset, trials_per_problem, objective, verbose, model, reuse_rtl):
    load_env()
    storage_uri = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna/optuna.db")

    cvdp = CVDPDataset(subset=dataset)
    backend = create_backend(model=model)
    runner = TrialRunner(backend)

    from src.optimizer.optuna_runner import OptunaRunner
    opt_runner = OptunaRunner(
        trial_runner=runner,
        storage_uri=storage_uri,
    )

    click.echo(f"Starting multi-problem Optuna optimization over {len(cvdp)} problems...")
    for idx, problem in enumerate(cvdp):
        click.echo(f"\n[{idx+1}/{len(cvdp)}] Optimizing {problem.id}...")
        study = opt_runner.optimize_problem(
            problem=problem,
            n_trials=trials_per_problem,
            objective_metric=objective,
            enable_rtl_reuse=reuse_rtl,
            verbose=verbose,
        )
        try:
            click.echo(f"  Best value: {study.best_trial.value}")
        except ValueError:
            click.echo("  No successful trials.")


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


@main.command()
def tui():
    """Start the interactive Terminal User Interface (TUI)."""
    from src.tui.app import VeriGenTUI
    app = VeriGenTUI()
    app.run()


if __name__ == "__main__":
    main()

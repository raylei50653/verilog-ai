import math
import statistics
from dataclasses import dataclass, field
from typing import Any


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < k:
        return 0.0
    if num_correct < 1:
        return 0.0
    if k == 1:
        return num_correct / num_samples
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


@dataclass
class TrialScore:
    problem_id: str
    trial_index: int
    syntax_pass: bool = False
    simulation_pass: bool = False
    ppa_metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0
    retry_count: int = 0

    @property
    def pass_(self) -> bool:
        return self.syntax_pass and self.simulation_pass


@dataclass
class ProblemScore:
    problem_id: str
    categories: list[str] = field(default_factory=list)
    trials: list[TrialScore] = field(default_factory=list)

    @property
    def num_correct(self) -> int:
        return sum(1 for t in self.trials if t.pass_)

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @property
    def pass_at_1(self) -> float:
        return self.num_correct / max(self.num_trials, 1)

    def pass_at_k(self, k: int) -> float:
        return estimate_pass_at_k(self.num_trials, self.num_correct, k)

    @property
    def syntax_pass_rate(self) -> float:
        return sum(1 for t in self.trials if t.syntax_pass) / max(self.num_trials, 1)

    @property
    def sim_pass_rate(self) -> float:
        syntax_ok = sum(1 for t in self.trials if t.syntax_pass)
        if syntax_ok == 0:
            return 0.0
        return sum(1 for t in self.trials if t.simulation_pass) / syntax_ok

    @property
    def avg_ppa(self) -> dict[str, float]:
        metrics: dict[str, list[float]] = {}
        for t in self.trials:
            if t.pass_ and t.ppa_metrics:
                for key, val in t.ppa_metrics.items():
                    if isinstance(val, (int, float)):
                        metrics.setdefault(key, []).append(float(val))
        return {k: statistics.mean(v) for k, v in metrics.items()}


@dataclass
class BenchmarkReport:
    subset: str
    total_problems: int
    total_trials: int
    samples_per_problem: int
    pass_at_1: float = 0.0
    pass_at_5: float = 0.0
    syntax_pass_rate: float = 0.0
    sim_pass_rate: float = 0.0
    problem_scores: dict[str, ProblemScore] = field(default_factory=dict)
    avg_duration_ms: float = 0.0

    @classmethod
    def from_scores(
        cls,
        subset: str,
        scores: dict[str, ProblemScore],
        samples_per_problem: int,
        k_values: list[int] | None = None,
    ) -> "BenchmarkReport":
        k_values = k_values or [1, 5]
        total = len(scores)
        if total == 0:
            return cls(subset=subset, total_problems=0, total_trials=0, samples_per_problem=0)

        total_trials = sum(s.num_trials for s in scores.values())

        p1_values = [s.pass_at_k(1) for s in scores.values()]
        pass_at_1 = statistics.mean(p1_values) if p1_values else 0.0

        k5 = min(k_values[1], samples_per_problem) if len(k_values) > 1 else 1
        p5_values = [s.pass_at_k(k5) for s in scores.values()]
        pass_at_5 = statistics.mean(p5_values) if p5_values else 0.0

        syn_values = [s.syntax_pass_rate for s in scores.values()]
        syn_rate = statistics.mean(syn_values) if syn_values else 0.0

        sim_values = [s.sim_pass_rate for s in scores.values()]
        sim_rate = statistics.mean(sim_values) if sim_values else 0.0

        durations = [
            t.duration_ms for s in scores.values() for t in s.trials if t.duration_ms > 0
        ]
        avg_dur = statistics.mean(durations) if durations else 0.0

        return cls(
            subset=subset,
            total_problems=total,
            total_trials=total_trials,
            samples_per_problem=samples_per_problem,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            syntax_pass_rate=syn_rate,
            sim_pass_rate=sim_rate,
            problem_scores=scores,
            avg_duration_ms=avg_dur,
        )

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.subset}",
            f"  Problems: {self.total_problems}",
            f"  Samples per problem: {self.samples_per_problem}",
            f"  Total trials: {self.total_trials}",
            f"  Pass@1:  {self.pass_at_1:.2%}",
            f"  Pass@5:  {self.pass_at_5:.2%}",
            f"  Syntax:  {self.syntax_pass_rate:.2%}",
            f"  Sim:     {self.sim_pass_rate:.2%}",
            f"  Avg time: {self.avg_duration_ms:.0f}ms/trial",
        ]
        passed = sum(1 for s in self.problem_scores.values() if s.num_correct > 0)
        lines.append(f"  Problems solved: {passed}/{self.total_problems}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "total_problems": self.total_problems,
            "total_trials": self.total_trials,
            "samples_per_problem": self.samples_per_problem,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "syntax_pass_rate": self.syntax_pass_rate,
            "sim_pass_rate": self.sim_pass_rate,
            "avg_duration_ms": self.avg_duration_ms,
        }

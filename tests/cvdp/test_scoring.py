import pytest
from src.cvdp.scoring import (
    estimate_pass_at_k,
    TrialScore,
    ProblemScore,
    BenchmarkReport,
)


class TestPassAtK:
    def test_k1_all_correct(self):
        assert estimate_pass_at_k(10, 10, 1) == 1.0

    def test_k1_half_correct(self):
        assert estimate_pass_at_k(10, 5, 1) == 0.5

    def test_k1_none_correct(self):
        assert estimate_pass_at_k(10, 0, 1) == 0.0

    def test_k5_some_correct(self):
        p = estimate_pass_at_k(10, 3, 5)
        assert 0.0 <= p <= 1.0

    def test_n_less_than_k(self):
        assert estimate_pass_at_k(3, 3, 5) == 0.0

    def test_pass_at_k_is_probability(self):
        for n in [5, 10, 20]:
            for c in range(n + 1):
                for k in [1, min(5, n)]:
                    p = estimate_pass_at_k(n, c, k)
                    assert 0.0 <= p <= 1.0, f"n={n} c={c} k={k} p={p}"


class TestTrialScore:
    def test_pass_when_both_pass(self):
        t = TrialScore("p1", 0, True, True)
        assert t.pass_

    def test_fail_when_syntax_fail(self):
        t = TrialScore("p1", 0, False, True)
        assert not t.pass_

    def test_fail_when_sim_fail(self):
        t = TrialScore("p1", 0, True, False)
        assert not t.pass_


class TestProblemScore:
    def test_empty(self):
        ps = ProblemScore("test")
        assert ps.num_trials == 0
        assert ps.num_correct == 0

    def test_all_pass(self):
        ps = ProblemScore("test")
        for i in range(5):
            ps.trials.append(TrialScore("test", i, True, True))
        assert ps.num_correct == 5
        assert ps.pass_at_1 == 1.0
        assert ps.syntax_pass_rate == 1.0
        assert ps.sim_pass_rate == 1.0

    def test_mixed(self):
        ps = ProblemScore("test")
        ps.trials.append(TrialScore("test", 0, True, True))
        ps.trials.append(TrialScore("test", 1, True, False))
        ps.trials.append(TrialScore("test", 2, False, False))
        assert ps.num_correct == 1
        assert ps.pass_at_1 == 1.0 / 3.0
        assert ps.syntax_pass_rate == 2.0 / 3.0
        assert ps.sim_pass_rate == 0.5  # 1 of 2 syntax-pass trials

    def test_avg_ppa(self):
        ps = ProblemScore("test")
        ps.trials.append(TrialScore("test", 0, True, True, ppa_metrics={"area": 100}))
        ps.trials.append(TrialScore("test", 1, True, True, ppa_metrics={"area": 200}))
        ps.trials.append(TrialScore("test", 2, True, False))  # not pass
        assert ps.avg_ppa == {"area": 150.0}


class TestBenchmarkReport:
    def test_empty(self):
        report = BenchmarkReport.from_scores("test", {}, 10)
        assert report.total_problems == 0

    def test_single_problem(self):
        ps = ProblemScore("p1")
        ps.trials.append(TrialScore("p1", 0, True, True, duration_ms=100))
        ps.trials.append(TrialScore("p1", 1, True, False, duration_ms=200))

        report = BenchmarkReport.from_scores("test", {"p1": ps}, 2)
        assert report.total_problems == 1
        assert report.total_trials == 2
        assert report.pass_at_1 == 0.5

    def test_to_dict(self):
        ps = ProblemScore("p1")
        ps.trials.append(TrialScore("p1", 0, True, True))
        report = BenchmarkReport.from_scores("test", {"p1": ps}, 1)
        d = report.to_dict()
        assert d["total_problems"] == 1
        assert d["total_trials"] == 1

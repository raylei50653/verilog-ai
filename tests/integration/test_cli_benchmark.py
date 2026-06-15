from unittest.mock import MagicMock, patch


class TestBenchmarkCLI:
    @patch("src.cli.create_backend")
    @patch("src.cli.TrialRunner")
    @patch("src.cli.CVDPDataset")
    def test_benchmark_baseline_passes_context_files(
        self, mock_cvdp_cls, mock_runner_cls, mock_create_backend
    ):
        from src.cvdp.scoring import TrialScore
        from src.cli import benchmark

        problem = MagicMock()
        problem.id = "cvdp_counter_0001"
        problem.difficulty = "medium"
        problem.prompt = "Implement a counter"
        problem.categories = ["cid001", "medium"]
        problem.context = {"docs/spec.txt": "counter should wrap"}
        problem.has_testbench.return_value = True
        problem.get_testbench_files.return_value = {"tb.sv": "module tb; endmodule"}

        mock_cvdp = MagicMock()
        mock_cvdp.__iter__.return_value = iter([problem])
        mock_cvdp_cls.return_value = mock_cvdp

        runner = MagicMock()
        score = TrialScore(problem_id=problem.id, trial_index=0, syntax_pass=True, simulation_pass=True)
        runner.run_trial.return_value = ("module counter; endmodule", score)
        mock_runner_cls.return_value = runner
        mock_create_backend.return_value = MagicMock()

        benchmark.callback(
            dataset="nonagentic_no_commercial",
            samples=1,
            pass_k="1",
            max_problems=1,
            category=(),
            difficulty=(),
            output="results/test_benchmark.json",
            temperature=0.2,
            verbose=False,
            dry_run=False,
            model=None,
            baseline=True,
        )

        kwargs = runner.run_trial.call_args.kwargs
        assert kwargs["baseline_mode"] is True
        assert kwargs["context_files"] == {"docs/spec.txt": "counter should wrap"}

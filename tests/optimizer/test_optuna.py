import pytest
from unittest.mock import MagicMock, patch
import optuna
from pathlib import Path
from src.optimizer.optuna_runner import OptunaRunner
from src.cvdp.loader import CVDPProblem
from src.cvdp.scoring import TrialScore


class TestOptunaRunner:
    @pytest.fixture
    def mock_trial_runner(self):
        runner = MagicMock()
        return runner

    @pytest.fixture
    def sample_problem(self):
        row = {
            "id": "cvdp_counter_0001",
            "categories": ["counter", "easy"],
            "input": {
                "prompt": "Implement a 4-bit counter",
                "context": {}
            },
            "output": {
                "context": {"rtl/counter.sv": ""}
            },
            "harness": {
                "files": {
                    "src/test_counter.py": "import cocotb"
                }
            }
        }
        return CVDPProblem(row)

    def test_load_config_fallback(self, mock_trial_runner):
        # Pass a nonexistent path to trigger fallback
        runner = OptunaRunner(mock_trial_runner, config_path="nonexistent_path.json")
        assert "counter" in runner.params_config
        assert "pipeline" in runner.params_config
        assert "default" in runner.params_config

    def test_get_space_for_problem(self, mock_trial_runner, sample_problem):
        runner = OptunaRunner(mock_trial_runner)
        space = runner.get_space_for_problem(sample_problem)
        assert "width" in space
        assert "direction" in space

    def test_optimize_problem_success(self, mock_trial_runner, sample_problem):
        # Set up trial runner to return a passing TrialScore
        score = TrialScore(problem_id=sample_problem.id, trial_index=0)
        score.syntax_pass = True
        score.simulation_pass = True
        score.ppa_metrics = {"area": 42.0, "num_wires": 15.0}
        mock_trial_runner.run_trial.return_value = ("module counter; endmodule", score)

        runner = OptunaRunner(mock_trial_runner, storage_uri=None)
        study = runner.optimize_problem(
            problem=sample_problem,
            n_trials=3,
            objective_metric="area",
            verbose=False,
        )

        assert len(study.trials) == 3
        assert study.best_trial.value == 42.0
        assert "width" in study.best_trial.params

    def test_optimize_problem_failure_penalty(self, mock_trial_runner, sample_problem):
        # Set up trial runner to return a failing TrialScore
        score = TrialScore(problem_id=sample_problem.id, trial_index=0)
        score.syntax_pass = False
        score.simulation_pass = False
        score.ppa_metrics = {}
        mock_trial_runner.run_trial.return_value = ("module counter; endmodule", score)

        runner = OptunaRunner(mock_trial_runner, storage_uri=None)
        study = runner.optimize_problem(
            problem=sample_problem,
            n_trials=2,
            objective_metric="area",
            verbose=False,
        )

        assert len(study.trials) == 2
        # All trials should have returned the penalty value 1e9
        for trial in study.trials:
            assert trial.value == 1e9

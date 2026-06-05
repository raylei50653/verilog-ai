import pytest
from unittest.mock import MagicMock, patch
from src.cvdp.scoring import TrialScore


class TestTrialRunner:
    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock()
        response = MagicMock()
        response.text = "module counter(); endmodule"
        response.model = "test"
        response.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        backend.generate.return_value = response
        backend.generate_with_thinking.return_value = response
        return backend

    @patch("src.agents.syntax.SyntaxAgent.execute")
    def test_syntax_pass_no_tb(self, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(spec="counter", problem_id="test_001")

        assert "module counter" in code
        assert score.syntax_pass
        assert not score.simulation_pass  # no testbench

    @patch("src.agents.syntax.SyntaxAgent.execute")
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_full_pass(self, mock_sim, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)
        mock_sim.return_value = AgentResult(pass_=True)

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            testbench_files={"tb.sv": "module tb; endmodule"},
        )

        assert score.syntax_pass
        assert score.simulation_pass
        assert score.pass_

    @patch("src.agents.syntax.SyntaxAgent.execute")
    def test_syntax_retry_then_pass(self, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.side_effect = [
            AgentResult(pass_=False, errors=[{"line": 1, "message": "bad"}]),
            AgentResult(pass_=True),
        ]

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter", problem_id="test_001", max_retries=2
        )

        assert score.syntax_pass

    @patch("src.agents.syntax.SyntaxAgent.execute")
    def test_syntax_max_retries_exceeded(self, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(
            pass_=False, errors=[{"line": 1, "message": "bad"}]
        )

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter", problem_id="test_001", max_retries=1
        )

        assert not score.syntax_pass
        assert not score.pass_
        assert len(score.errors) > 0

    @patch("src.agents.syntax.SyntaxAgent.execute")
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_simulation_retry_then_pass(self, mock_sim, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)
        mock_sim.side_effect = [
            AgentResult(pass_=False, errors=[{"message": "assert failed"}]),
            AgentResult(pass_=True),
        ]

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            testbench_files={"tb.sv": "module tb; endmodule"},
            max_retries=2
        )

        assert score.syntax_pass
        assert score.simulation_pass
        assert score.pass_
        assert score.retry_count == 1

    @patch("src.agents.syntax.SyntaxAgent.execute")
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_simulation_max_retries_exceeded(self, mock_sim, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)
        mock_sim.return_value = AgentResult(pass_=False, errors=[{"message": "assert failed"}])

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            testbench_files={"tb.sv": "module tb; endmodule"},
            max_retries=1
        )

        assert score.syntax_pass
        assert not score.simulation_pass
        assert not score.pass_
        assert len(score.errors) > 0

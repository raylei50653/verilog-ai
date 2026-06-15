import pytest
from unittest.mock import MagicMock, patch
from src.cvdp.scoring import TrialScore
from src.cancellation import CancellationToken, PipelineCancelled


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
        assert score.trial_id
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

    @patch("src.agents.syntax.SyntaxAgent.execute")
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_simulation_infra_failure_aborts_retries(self, mock_sim, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)
        # Simulate infrastructure failure
        mock_sim.return_value = AgentResult(
            pass_=False,
            errors=[{"message": "Docker pytest exited with code 125"}],
            is_infra_failure=True,
        )

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            testbench_files={"tb.sv": "module tb; endmodule"},
            max_retries=3
        )

        # It should exit immediately on the first failure, making mock_sim called exactly once
        assert mock_sim.call_count == 1
        assert not score.simulation_pass
        assert not score.pass_

    @patch("src.agents.diagnosis.DiagnosisAgent.execute")
    @patch("src.agents.syntax.SyntaxAgent.execute")
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_baseline_mode_uses_diagnosis_with_source_context(
        self, mock_sim, mock_syntax, mock_diagnosis, mock_backend
    ):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.side_effect = [
            AgentResult(pass_=False, errors=[{"line": 1, "message": "bad"}]),
            AgentResult(pass_=True),
            AgentResult(pass_=True),
        ]
        mock_sim.return_value = AgentResult(pass_=True)
        mock_diagnosis.return_value = AgentResult(
            pass_=False,
            raw_output=(
                "TODO 1\n"
                "LOCATION: line 1\n"
                "SNIPPET: `module counter(); endmodule`\n"
                "BUG: fix line 1\n"
                "FIX: correct the syntax issue\n"
                "REVIEW: verify syntax passes\n"
            ),
        )

        runner = TrialRunner(mock_backend)
        _, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            context_files={"docs/spec.txt": "counter should wrap"},
            testbench_files={"tb.sv": "module tb; endmodule"},
            max_retries=2,
            baseline_mode=True,
        )

        assert score.pass_
        mock_diagnosis.assert_called_once()
        diag_config = mock_diagnosis.call_args.kwargs["config"]
        assert diag_config["errors"] == [{"line": 1, "message": "bad"}]
        assert len(diag_config["source_context"]) == 1
        assert diag_config["source_context"][0]["error_line"] == 1

    @patch("src.agents.diagnosis.DiagnosisAgent.execute")
    @patch("src.agents.syntax.SyntaxAgent.execute")
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_simulation_failure_uses_testbench_context(
        self, mock_sim, mock_syntax, mock_diagnosis, mock_backend
    ):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)
        mock_sim.side_effect = [
            AgentResult(pass_=False, errors=[{"message": "assert failed"}]),
            AgentResult(pass_=True),
        ]
        mock_diagnosis.return_value = AgentResult(
            pass_=False,
            raw_output=(
                "TODO 1\n"
                "LOCATION: reset branch\n"
                "SNIPPET: `if (rst)`\n"
                "BUG: reset behavior does not match the testbench\n"
                "FIX: align reset polarity and branch logic with the testbench\n"
                "REVIEW: verify reset assertion passes\n"
            ),
        )

        runner = TrialRunner(mock_backend)
        _, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            testbench_files={
                "tb.sv": (
                    "module tb;\n"
                    "  reg clk;\n"
                    "  reg reset;\n"
                    "  my_dut dut(.clk(clk), .reset(reset));\n"
                    "  initial assert(reset === 1'b0);\n"
                    "endmodule\n"
                )
            },
            max_retries=2,
            baseline_mode=True,
        )

        assert score.pass_
        diag_config = mock_diagnosis.call_args.kwargs["config"]
        assert len(diag_config["testbench_context"]) > 0
        assert diag_config["testbench_context"][0]["path"].endswith("tb.sv")

    @patch("src.agents.diagnosis.DiagnosisAgent.execute")
    @patch("src.agents.syntax.SyntaxAgent.execute")
    def test_repairs_all_todos_in_single_call(self, mock_syntax, mock_diagnosis, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.side_effect = [
            AgentResult(pass_=False, errors=[{"line": 3, "message": "bad reset"}]),
            AgentResult(pass_=True),
            AgentResult(pass_=True),
        ]
        mock_diagnosis.return_value = AgentResult(
            pass_=False,
            raw_output=(
                "TODO 1\n"
                "LOCATION: line 3\n"
                "BUG: reset polarity mismatch\n"
                "FIX: invert reset condition\n\n"
                "TODO 2\n"
                "LOCATION: line 4\n"
                "BUG: missing enable gate\n"
                "FIX: guard increment with en\n"
            ),
        )

        runner = TrialRunner(mock_backend)
        code, score = runner.run_trial(
            spec="counter",
            problem_id="test_001",
            max_retries=2,
        )

        assert score.syntax_pass
        # 1 call for RTL generation + 1 call for merged TODO fix
        assert mock_backend.generate.call_count == 2
        fix_prompt = mock_backend.generate.call_args_list[1].kwargs["user_prompt"]
        assert "TODO [1]" in fix_prompt
        assert "TODO [2]" in fix_prompt

    def test_todo_source_context_prefers_snippet_anchor(self, mock_backend, tmp_path):
        from src.pipeline import TrialRunner
        from src.mcp.server import MCPServer

        server = MCPServer(db_path=str(tmp_path / "trials.db"))
        runner = TrialRunner(mock_backend, mcp=server)
        rtl_path = server.write_trial_source(
            "trial-001",
            "rtl.sv",
            "module top;\nif (rst) count <= 0;\nelse count <= count + 1;\nendmodule\n",
        )
        todo = {
            "id": "todo-1",
            "order": 1,
            "status": "pending",
            "location": {"raw": "line 99", "line_start": 99, "line_end": 99},
            "snippet": "if (rst) count <= 0;",
            "bug": "reset polarity mismatch",
            "fix": "invert reset condition",
            "review": "verify reset branch",
        }

        ctx = runner._build_todo_source_context(str(rtl_path), todo)

        assert len(ctx) == 1
        assert ctx[0]["error_line"] == 2
        assert any("if (rst) count <= 0;" in line["text"] for line in ctx[0]["excerpt"])

    @patch("src.agents.syntax.SyntaxAgent.execute")
    def test_run_trial_hard_stop_raises_and_skips_db_write(self, mock_syntax, mock_backend):
        from src.pipeline import TrialRunner
        from src.agents.base import AgentResult

        mock_syntax.return_value = AgentResult(pass_=True)
        runner = TrialRunner(mock_backend)
        token = CancellationToken()

        def cancel_on_gen_success(step, status):
            if step == "gen" and status == "success":
                token.cancel()

        runner.mcp.write_trial = MagicMock()

        with pytest.raises(PipelineCancelled):
            runner.run_trial(
                spec="counter",
                problem_id="test_001",
                on_step_change=cancel_on_gen_success,
                cancel_token=token,
            )

        runner.mcp.write_trial.assert_not_called()

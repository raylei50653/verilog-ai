import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import TrialRunner
from src.mcp.server import MCPServer, TrialRecord
from src.cvdp.scoring import TrialScore


class TestRTLReuse:
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
    @patch("src.agents.simulation.SimulationAgent.execute")
    def test_rtl_reuse_integration(self, mock_sim, mock_syntax, mock_backend, tmp_path):
        mock_syntax.return_value = MagicMock(pass_=True)
        mock_sim.return_value = MagicMock(pass_=True)

        # Setup temporary MCPServer with DB
        db_path = tmp_path / "trials.db"
        mcp = MCPServer(db_path=str(db_path))
        mcp.init_db()

        # Write a previous successful trial
        prev_trial = TrialRecord(
            trial_id="prev-123",
            problem_id="cvdp_counter_0001",
            params={"width": 4},
            spec_prompt="Implement a 4-bit counter",
            generated_code="module counter #(parameter WIDTH = 4) (); endmodule",
            syntax_pass=True,
            simulation_pass=True,
            pass_=True,
        )
        mcp.write_trial(prev_trial)

        # Initialize runner with this MCPServer
        runner = TrialRunner(backend=mock_backend, mcp=mcp)

        # Patch generator's generate_rtl to spy on its arguments
        with patch.object(runner.generator, "generate_rtl", wraps=runner.generator.generate_rtl) as mock_generate:
            code, score = runner.run_trial(
                spec="Implement an 8-bit counter",
                problem_id="cvdp_counter_0002",
                testbench_files={"tb.sv": "module tb; endmodule"},
                enable_rtl_reuse=True,
            )

            # Assert that generate_rtl was called and that examples were supplied
            mock_generate.assert_called_once()
            called_kwargs = mock_generate.call_args[1]
            examples = called_kwargs.get("examples")
            
            assert examples is not None
            assert len(examples) == 1
            assert examples[0]["spec"] == "Implement a 4-bit counter"
            assert "module counter" in examples[0]["code"]

import pytest
from unittest.mock import patch, MagicMock


class TestSimulationAgent:
    def test_no_testbench(self):
        from src.agents.simulation import SimulationAgent

        agent = SimulationAgent()
        result = agent.execute({"top.sv": "module top(); endmodule"})
        assert not result.pass_
        assert any("testbench" in str(e).lower() for e in result.errors)

    @patch("src.agents.simulation.SimulationAgent._check_docker")
    def test_cocotb_testbench_skipped(self, mock_check_docker):
        mock_check_docker.return_value = False
        from src.agents.simulation import SimulationAgent

        agent = SimulationAgent()
        result = agent.execute(
            {"top.sv": "module top(); endmodule"},
            {"src/test_top.py": "import cocotb"},
        )
        assert result.pass_
        assert "cocotb" in result.warnings[0]["message"].lower()

    @patch("src.agents.simulation.shutil.which")
    @patch("src.agents.simulation.subprocess.run")
    def test_iverilog_sim_pass(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/iverilog"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "PASSED"
        mock_result.stderr = ""
        mock_run.side_effect = [mock_result, mock_result]

        from src.agents.simulation import SimulationAgent

        agent = SimulationAgent()
        result = agent.execute(
            {"top.sv": "module top(); endmodule"},
            {"tb.sv": "module tb; endmodule"},
        )
        assert result.pass_

    @patch("src.agents.simulation.shutil.which")
    @patch("src.agents.simulation.subprocess.run")
    def test_sim_timeout(self, mock_run, mock_which):
        import subprocess

        mock_which.return_value = "/usr/bin/iverilog"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="iverilog", timeout=1)

        from src.agents.simulation import SimulationAgent

        agent = SimulationAgent()
        result = agent.execute(
            {"top.sv": "module top(); endmodule"},
            {"tb.sv": "module tb; endmodule"},
        )
        assert not result.pass_

    def test_is_cocotb(self):
        from src.agents.simulation import SimulationAgent

        agent = SimulationAgent()
        assert agent._is_cocotb_testbench({"test.py": "import cocotb"})
        assert not agent._is_cocotb_testbench({"tb.sv": "module tb; endmodule"})

    @patch("src.agents.simulation.SimulationAgent._check_docker")
    @patch("src.agents.simulation.subprocess.run")
    def test_cocotb_testbench_docker(self, mock_run, mock_check_docker):
        mock_check_docker.return_value = True
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "pytest passed"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        from src.agents.simulation import SimulationAgent

        agent = SimulationAgent()
        result = agent.execute(
            {"rtl/top.sv": "module top(); endmodule"},
            {"src/.env": "SIM=icarus\n", "src/test_top.py": "import cocotb", "src/test_runner.py": ""},
        )
        assert result.pass_
        assert mock_run.called
        # Verify docker command args
        args = mock_run.call_args[0][0]
        assert "docker" in args
        assert "run" in args
        assert "pytest" in args

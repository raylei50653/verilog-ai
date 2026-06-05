import json
import pytest
from pathlib import Path
from src.mcp.server import MCPServer, TrialRecord


@pytest.fixture
def temp_config_dir(tmp_path):
    constraints = {"default": {"clock_freq_mhz": 100}, "counter": {"width": [4, 8, 16]}}
    interfaces = {"axi4_lite": {"handshake": "valid-ready"}}
    constraints_path = tmp_path / "constraints.json"
    interfaces_path = tmp_path / "interfaces.json"
    constraints_path.write_text(json.dumps(constraints))
    interfaces_path.write_text(json.dumps(interfaces))
    return constraints_path, interfaces_path, tmp_path


class TestMCPServer:
    def test_get_constraints_with_type(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip))
        result = server.get_constraints("counter")
        assert result["clock_freq_mhz"] == 100
        assert result["width"] == [4, 8, 16]

    def test_get_constraints_default(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip))
        result = server.get_constraints("unknown_type")
        assert result["clock_freq_mhz"] == 100
        assert "width" not in result

    def test_get_interface_found(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip))
        result = server.get_interface("axi4_lite")
        assert result is not None
        assert result["handshake"] == "valid-ready"

    def test_get_interface_not_found(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip))
        result = server.get_interface("nonexistent")
        assert result is None

    def test_write_trial(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        trial = TrialRecord(
            trial_id="test-001",
            problem_id="cvdp_counter_0001",
            params={"width": 8},
            spec_prompt="implement a counter",
            generated_code="module counter ...",
            syntax_pass=True,
            simulation_pass=True,
            pass_=True,
        )
        tid = server.write_trial(trial)
        assert tid == "test-001"

        # Verify it can be retrieved
        history = server.get_history("cvdp_counter_0001")
        assert len(history) == 1
        assert history[0]["trial_id"] == "test-001"
        assert history[0]["params"] == {"width": 8}
        assert history[0]["pass_"] is True

        # Verify successful trials
        successes = server.get_successful_trials("cvdp_counter_0001")
        assert len(successes) == 1
        assert successes[0]["trial_id"] == "test-001"

    def test_get_history_empty(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        history = server.get_history("cvdp_counter_0001")
        assert history == []

    def test_init_db(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        server.init_db()
        assert db_path.parent.exists()
        assert db_path.exists()

    def test_get_successful_trials_by_pattern(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        trial = TrialRecord(
            trial_id="test-001",
            problem_id="cvdp_counter_0001",
            params={"width": 8},
            spec_prompt="implement a counter",
            generated_code="module counter ...",
            syntax_pass=True,
            simulation_pass=True,
            pass_=True,
        )
        server.write_trial(trial)
        
        # Query matching pattern
        matches = server.get_successful_trials_by_pattern("counter")
        assert len(matches) == 1
        assert matches[0]["trial_id"] == "test-001"

        # Query non-matching pattern
        no_matches = server.get_successful_trials_by_pattern("fifo")
        assert len(no_matches) == 0

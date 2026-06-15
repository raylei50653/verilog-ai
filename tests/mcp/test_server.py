import json
import pytest
from pathlib import Path
from unittest.mock import patch
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

    def test_read_file_lines(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        file_path = tp / "rtl.sv"
        file_path.write_text("line1\nline2\nline3\nline4\n")

        excerpt = server.read_file_lines(file_path, start_line=2, end_line=3)

        assert excerpt["start_line"] == 2
        assert excerpt["end_line"] == 3
        assert excerpt["lines"] == [
            {"line": 2, "text": "line2"},
            {"line": 3, "text": "line3"},
        ]

    def test_get_error_source_context(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        file_path = tp / "rtl.sv"
        file_path.write_text("a\nb\nc\nd\ne\n")

        contexts = server.get_error_source_context(
            file_path,
            [{"line": 3, "message": "bad token"}],
            context_radius=1,
        )

        assert len(contexts) == 1
        assert contexts[0]["error_line"] == 3
        assert contexts[0]["message"] == "bad token"
        assert contexts[0]["excerpt"] == [
            {"line": 2, "text": "b"},
            {"line": 3, "text": "c"},
            {"line": 4, "text": "d"},
        ]

    def test_get_testbench_context(self, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        file_path = tp / "tb.sv"
        file_path.write_text(
            "module tb;\n"
            "  reg clk;\n"
            "  reg reset;\n"
            "  my_dut dut(.clk(clk));\n"
            "  initial assert(clk === 0);\n"
            "endmodule\n"
        )

        contexts = server.get_testbench_context([file_path], context_radius=1, max_matches_per_file=2)

        assert len(contexts) == 2
        assert contexts[0]["path"].endswith("tb.sv")
        assert "excerpt" in contexts[0]

    @patch("builtins.print")
    def test_parse_and_manage_todos(self, mock_print, temp_config_dir):
        cp, ip, tp = temp_config_dir
        db_path = tp / "trials.db"
        server = MCPServer(constraints_path=str(cp), interfaces_path=str(ip), db_path=str(db_path))
        report = (
            "TODO 1\n"
            "LOCATION: line 3\n"
            "SNIPPET: `if (rst) count <= 0;`\n"
            "BUG: reset polarity mismatch\n"
            "FIX: invert reset condition\n"
            "REVIEW: verify active-low reset\n\n"
            "TODO 2\n"
            "LOCATION: always block\n"
            "SNIPPET: `count <= count + 1;`\n"
            "BUG: missing enable gate\n"
            "FIX: guard increment with en\n"
            "REVIEW: verify hold behavior\n"
        )

        todos = server.parse_todo_report(report)
        server.write_todos("trial-001", todos)

        assert len(todos) == 2
        assert todos[0]["location"]["line_start"] == 3
        assert todos[0]["snippet"] == "if (rst) count <= 0;"
        assert server.get_next_pending_todo("trial-001")["id"] == "todo-1"

        server.update_todo_status("trial-001", "todo-1", "done", review_notes="checked reset path")
        next_todo = server.get_next_pending_todo("trial-001")
        stored = server.read_todos("trial-001")

        assert next_todo["id"] == "todo-2"
        assert stored[0]["review_notes"] == "checked reset path"
        assert any(
            call.args[0] == "[update TODO] wrote 2 item(s) for trial trial-001"
            for call in mock_print.call_args_list
        )
        assert any(
            call.args[0] == "[update TODO] todo-1 -> done (reset polarity mismatch)"
            for call in mock_print.call_args_list
        )

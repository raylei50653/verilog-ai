import os
import shutil
import sqlite3
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
import pytest

from src.tui.app import VeriGenTUI
from src.mcp.server import MCPServer

class MockRichLog:
    def __init__(self):
        self.messages = []

    def write(self, msg: str):
        self.messages.append(msg)

@pytest.fixture
def temp_trial_dir(tmp_path):
    # Setup temporary directory for trial output
    trial_id = "test_trial_123"
    out_dir = tmp_path / "outputs" / trial_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy RTL file
    rtl_file = out_dir / "dut.v"
    rtl_file.write_text("""
module uut(
    input clk,
    input rst_n,
    output reg [3:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 0;
        else count <= count + 1;
    end
endmodule
""")
    return trial_id, out_dir

def test_build_vivado_project_no_iverilog(temp_trial_dir, tmp_path):
    trial_id, out_dir = temp_trial_dir
    log = MockRichLog()

    # Mock MCPServer methods
    with mock.patch.object(MCPServer, "get_trial_output_dir", return_value=out_dir), \
         mock.patch.object(MCPServer, "_get_connection") as mock_conn, \
         mock.patch("shutil.which", return_value=None), \
         mock.patch("src.llm.create_backend") as mock_create_backend, \
         mock.patch("os.getenv", return_value=str(tmp_path / "vivado_projects")):

        # Mock SQL lookup
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {
            "spec_prompt": "A simple counter",
            "generated_code": "module uut..."
        }
        conn.cursor.return_value = cursor
        mock_conn.return_value.__enter__.return_value = conn

        # Mock LLM backend
        mock_backend = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "```systemverilog\nmodule tb;\n  initial $finish;\nendmodule\n```"
        mock_backend.generate.return_value = mock_resp
        mock_create_backend.return_value = mock_backend

        # Run
        app_instance = MagicMock()
        proj_dir, rtl, tb, top, proj_name = VeriGenTUI._build_vivado_project(
            app_instance, trial_id, "uut", "xc7a35tcpg236-1", log
        )

        assert proj_dir is not None
        assert len(rtl) == 1
        assert len(tb) == 1
        assert "iverilog not found, skipping local syntax verification" in "".join(log.messages)

def test_build_vivado_project_success_first_try(temp_trial_dir, tmp_path):
    trial_id, out_dir = temp_trial_dir
    log = MockRichLog()

    with mock.patch.object(MCPServer, "get_trial_output_dir", return_value=out_dir), \
         mock.patch.object(MCPServer, "_get_connection") as mock_conn, \
         mock.patch("shutil.which", return_value="/usr/bin/iverilog"), \
         mock.patch("subprocess.run") as mock_run, \
         mock.patch("src.llm.create_backend") as mock_create_backend, \
         mock.patch("os.getenv", return_value=str(tmp_path / "vivado_projects")):

        # Mock SQL
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {
            "spec_prompt": "A simple counter",
            "generated_code": "module uut..."
        }
        conn.cursor.return_value = cursor
        mock_conn.return_value.__enter__.return_value = conn

        # Mock LLM backend
        mock_backend = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "```systemverilog\nmodule tb;\n  initial $finish;\nendmodule\n```"
        mock_backend.generate.return_value = mock_resp
        mock_create_backend.return_value = mock_backend

        # Mock subprocess.run for iverilog (0 means success)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_run.return_value = mock_res

        # Run
        app_instance = MagicMock()
        proj_dir, rtl, tb, top, proj_name = VeriGenTUI._build_vivado_project(
            app_instance, trial_id, "uut", "xc7a35tcpg236-1", log
        )

        assert proj_dir is not None
        assert "Testbench successfully verified with iverilog (Attempt 1)" in "".join(log.messages)
        assert "Testbench syntax verified successfully with iverilog" in "".join(log.messages)

def test_build_vivado_project_repair_success(temp_trial_dir, tmp_path):
    trial_id, out_dir = temp_trial_dir
    log = MockRichLog()

    with mock.patch.object(MCPServer, "get_trial_output_dir", return_value=out_dir), \
         mock.patch.object(MCPServer, "_get_connection") as mock_conn, \
         mock.patch("shutil.which", return_value="/usr/bin/iverilog"), \
         mock.patch("subprocess.run") as mock_run, \
         mock.patch("src.llm.create_backend") as mock_create_backend, \
         mock.patch("os.getenv", return_value=str(tmp_path / "vivado_projects")):

        # Mock SQL
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {
            "spec_prompt": "A simple counter",
            "generated_code": "module uut..."
        }
        conn.cursor.return_value = cursor
        mock_conn.return_value.__enter__.return_value = conn

        # Mock LLM backend (different response on generate call)
        mock_backend = MagicMock()
        mock_resp_fail = MagicMock()
        mock_resp_fail.text = "```systemverilog\nmodule tb;\n  broken_syntax;\nendmodule\n```"
        mock_resp_ok = MagicMock()
        mock_resp_ok.text = "```systemverilog\nmodule tb;\n  initial $finish;\nendmodule\n```"
        mock_backend.generate.side_effect = [mock_resp_fail, mock_resp_ok]
        mock_create_backend.return_value = mock_backend

        # Mock subprocess.run: first run fails, second run succeeds
        mock_res_fail = MagicMock()
        mock_res_fail.returncode = 1
        mock_res_fail.stderr = "syntax error at line 3"
        
        mock_res_ok = MagicMock()
        mock_res_ok.returncode = 0
        
        mock_run.side_effect = [mock_res_fail, mock_res_ok, mock_res_ok]  # third is the final general verification

        # Run
        app_instance = MagicMock()
        proj_dir, rtl, tb, top, proj_name = VeriGenTUI._build_vivado_project(
            app_instance, trial_id, "uut", "xc7a35tcpg236-1", log
        )

        assert proj_dir is not None
        assert "Testbench verification failed (Attempt 1/3)" in "".join(log.messages)
        assert "syntax error at line 3" in "".join(log.messages)
        assert "AI is repairing the testbench..." in "".join(log.messages)
        assert "Testbench successfully verified with iverilog (Attempt 2)" in "".join(log.messages)

def test_build_vivado_project_repair_fails_fallback(temp_trial_dir, tmp_path):
    trial_id, out_dir = temp_trial_dir
    log = MockRichLog()

    with mock.patch.object(MCPServer, "get_trial_output_dir", return_value=out_dir), \
         mock.patch.object(MCPServer, "_get_connection") as mock_conn, \
         mock.patch("shutil.which", return_value="/usr/bin/iverilog"), \
         mock.patch("subprocess.run") as mock_run, \
         mock.patch("src.llm.create_backend") as mock_create_backend, \
         mock.patch("os.getenv", return_value=str(tmp_path / "vivado_projects")):

        # Mock SQL
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {
            "spec_prompt": "A simple counter",
            "generated_code": "module uut..."
        }
        conn.cursor.return_value = cursor
        mock_conn.return_value.__enter__.return_value = conn

        # Mock LLM backend: always return broken syntax
        mock_backend = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "```systemverilog\nmodule tb;\n  broken_syntax;\nendmodule\n```"
        mock_backend.generate.return_value = mock_resp
        mock_create_backend.return_value = mock_backend

        # Mock subprocess.run: always fail
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stderr = "syntax error"
        mock_run.return_value = mock_res

        # Run
        app_instance = MagicMock()
        proj_dir, rtl, tb, top, proj_name = VeriGenTUI._build_vivado_project(
            app_instance, trial_id, "uut", "xc7a35tcpg236-1", log
        )

        assert proj_dir is not None
        assert "Failed to repair testbench after 3 attempts" in "".join(log.messages)
        assert "Warning: Testbench compilation failed under iverilog" in "".join(log.messages)

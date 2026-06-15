import pytest
from unittest.mock import patch, MagicMock


class TestLlamaServerBackend:
    def test_backend_creation(self):
        from src.llm import LlamaServerBackend

        backend = LlamaServerBackend(base_url="http://localhost:8080/v1", model="test-model")
        assert backend.model == "test-model"
        assert backend.base_url == "http://localhost:8080/v1"

    @patch("src.llm.requests.Session")
    def test_generate_with_content(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "test-model",
            "choices": [{
                "message": {"role": "assistant", "content": "module counter(); endmodule"}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50},
        }
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        from src.llm import LlamaServerBackend

        backend = LlamaServerBackend(base_url="http://localhost:8080/v1", model="test-model")
        response = backend.generate(
            system_prompt="You are a Verilog expert.",
            user_prompt="Write a counter module.",
        )

        assert response.text == "module counter(); endmodule"
        assert response.model == "test-model"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 50

    @patch("src.llm.requests.Session")
    def test_generate_with_only_reasoning(self, mock_session_cls):
        reasoning = """
module counter(
    input clk, rst,
    output reg [3:0] count
);
    always @(posedge clk) begin
        if (rst) count <= 0;
        else count <= count + 1;
    end
endmodule
"""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "test-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning,
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100},
        }
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        from src.llm import LlamaServerBackend

        backend = LlamaServerBackend(base_url="http://localhost:8080/v1", model="test-model")
        response = backend.generate(
            system_prompt="You are a Verilog expert.",
            user_prompt="Write a counter.",
        )

        assert "module counter" in response.text
        assert "endmodule" in response.text

    @patch("src.llm.requests.Session")
    def test_generate_executes_tool_calls(self, mock_session_cls):
        mock_session = MagicMock()
        tool_resp = MagicMock()
        tool_resp.json.return_value = {
            "model": "test-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/demo.txt\"}",
                        },
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        final_resp = MagicMock()
        final_resp.json.return_value = {
            "model": "test-model",
            "choices": [{
                "message": {"role": "assistant", "content": "class names: A, B"}
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        mock_session.post.side_effect = [tool_resp, final_resp]
        mock_session_cls.return_value = mock_session

        with patch("src.llm._execute_local_tool", return_value="file content") as mock_tool:
            from src.llm import LlamaServerBackend, build_local_file_tools

            backend = LlamaServerBackend(base_url="http://localhost:8080/v1", model="test-model")
            response = backend.generate(
                system_prompt="You are a coding assistant.",
                user_prompt="Read file",
                tools=build_local_file_tools(),
            )

        assert response.text == "class names: A, B"
        mock_tool.assert_called_once_with("read_file", {"path": "/tmp/demo.txt"})
        second_payload = mock_session.post.call_args_list[1].kwargs["json"]
        assert second_payload["messages"][-1]["role"] == "tool"
        assert second_payload["messages"][-1]["content"] == "file content"

    @patch("src.llm.requests.Session")
    def test_tool_call_loop_aborts_on_depth_limit(self, mock_session_cls):
        mock_session = MagicMock()
        tool_resp = MagicMock()
        tool_resp.json.return_value = {
            "model": "test-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "thinking...",
                    "tool_calls": [{
                        "id": "call-loop",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/demo.txt\"}",
                        },
                    }],
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_session.post.return_value = tool_resp
        mock_session_cls.return_value = mock_session

        with patch("src.llm._execute_local_tool", return_value="file content"):
            from src.llm import LlamaServerBackend, build_local_file_tools

            backend = LlamaServerBackend(base_url="http://localhost:8080/v1", model="test-model")
            response = backend.generate(
                system_prompt="You are a coding assistant.",
                user_prompt="Read file",
                tools=build_local_file_tools(),
            )

        assert response.text == "thinking..."
        assert mock_session.post.call_count == 6

    @patch("builtins.print")
    def test_execute_local_tool_logs_usage(self, mock_print, tmp_path):
        file_path = tmp_path / "demo.txt"
        file_path.write_text("hello", encoding="utf-8")

        from src.llm import _execute_local_tool

        result = _execute_local_tool("read_file", {"path": str(file_path)})

        assert result == "hello"
        mock_print.assert_called_once_with(f"[tool used] read_file {file_path}")

    @patch("builtins.print")
    def test_execute_local_edit_file(self, mock_print, tmp_path):
        file_path = tmp_path / "edited.txt"

        from src.llm import _execute_local_tool, build_local_file_tools

        result = _execute_local_tool(
            "edit_file",
            {"path": str(file_path), "content": "updated"},
        )

        assert file_path.read_text(encoding="utf-8") == "updated"
        assert "Wrote" in result
        mock_print.assert_called_once_with(f"[tool used] edit_file {file_path}")
        assert any(tool["function"]["name"] == "edit_file" for tool in build_local_file_tools())

    def test_create_backend_llama_server(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "llama_server")
        monkeypatch.setenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("DEFAULT_MODEL", "test-model")

        from src.llm import create_backend, LlamaServerBackend

        backend = create_backend()
        assert isinstance(backend, LlamaServerBackend)

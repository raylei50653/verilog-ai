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

    def test_create_backend_llama_server(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "llama_server")
        monkeypatch.setenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("DEFAULT_MODEL", "test-model")

        from src.llm import create_backend, LlamaServerBackend

        backend = create_backend()
        assert isinstance(backend, LlamaServerBackend)

import pytest
from unittest.mock import patch, MagicMock
from src.main_model.generator import ModelGenerator


class TestModelGenerator:
    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock()
        response = MagicMock()
        response.text = """```verilog
module counter #(parameter WIDTH = 8) (
    input wire clk, rst_n, en,
    output reg [WIDTH-1:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 0;
        else if (en) count <= count + 1;
    end
endmodule
```"""
        response.model = "test-model"
        response.usage = {"prompt_tokens": 50, "completion_tokens": 100}
        backend.generate.return_value = response
        backend.generate_with_thinking.return_value = response
        return backend

    def test_generate_rtl_basic(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        code, resp = gen.generate_rtl("implement a counter")
        assert "module counter" in code
        assert "endmodule" in code
        assert resp.model == "test-model"

    def test_generate_rtl_with_params(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        code, _ = gen.generate_rtl(
            "implement a counter",
            params={"WIDTH": 16, "DIRECTION": "up"},
        )
        assert "module counter" in code

    def test_generate_rtl_with_constraints(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        constraints = {"clock_freq_mhz": 200, "setup_time_ns": 0.5}
        code, _ = gen.generate_rtl(
            "implement a pipeline multiplier", constraints=constraints
        )
        assert "module counter" in code

    def test_fix_errors(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        errors = [{"line": 5, "message": "syntax error: unexpected ';'"}]
        diagnosis_report = (
            "TODO 1\n"
            "LOCATION: line 5\n"
            "SNIPPET: `endmodule`\n"
            "BUG: syntax error near module terminator\n"
            "FIX: replace the malformed terminator with a valid endmodule\n"
            "REVIEW: verify the parser accepts the module terminator\n"
        )
        code, _ = gen.fix_errors("counter", "module bad(); enmodule", errors, diagnosis_report=diagnosis_report)
        assert "module counter" in code

        call_kwargs = mock_backend.generate.call_args.kwargs
        assert "debug" in call_kwargs["system_prompt"].lower()
        assert "TODO 1" in call_kwargs["user_prompt"]
        assert "tools" not in call_kwargs

    def test_extract_verilog_from_code_block(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        assert "module counter" in gen._extract_verilog("""
        Here is the code:
        ```verilog
        module counter();
        endmodule
        ```
        """)

    def test_extract_verilog_no_fence(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        text = "module counter(); endmodule"
        assert gen._extract_verilog(text) == text

    def test_extract_verilog_no_module(self, mock_backend):
        gen = ModelGenerator(mock_backend)
        text = "This is not Verilog code."
        assert gen._extract_verilog(text) == text


class TestPromptBuilding:
    @pytest.fixture
    def gen(self):
        backend = MagicMock()
        return ModelGenerator(backend)

    def test_basic_prompt(self, gen):
        prompt = gen._build_prompt("implement a counter", None, None, None, None)
        assert "## Specification" in prompt
        assert "implement a counter" in prompt

    def test_prompt_with_params(self, gen):
        prompt = gen._build_prompt(
            "implement a counter",
            {"WIDTH": 8},
            None,
            None,
            None,
        )
        assert "## Design Parameters" in prompt
        assert "WIDTH = 8" in prompt

    def test_prompt_with_constraints(self, gen):
        prompt = gen._build_prompt(
            "implement a counter",
            None,
            {"clock_freq_mhz": 100},
            None,
            None,
        )
        assert "## Timing and Design Constraints" in prompt
        assert "clock_freq_mhz: 100" in prompt

    def test_prompt_with_interface(self, gen):
        prompt = gen._build_prompt(
            "implement an AXI slave",
            None,
            None,
            {"signals": {"clk": "input wire", "data": "output reg [31:0]"}},
            None,
        )
        assert "## Interface Specification" in prompt
        assert "clk: input wire" in prompt

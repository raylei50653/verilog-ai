import shutil
import pytest
from src.agents.syntax import SyntaxAgent

IVERILOG_AVAILABLE = shutil.which("iverilog") is not None
pytestmark = pytest.mark.skipif(
    not IVERILOG_AVAILABLE, reason="iverilog not installed"
)

VALID_COUNTER_SV = """module counter #(
    parameter WIDTH = 8
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,
    output wire [WIDTH-1:0] count
);
    reg [WIDTH-1:0] count_reg;
    assign count = count_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count_reg <= 0;
        else if (en)
            count_reg <= count_reg + 1;
    end
endmodule
"""

SYNTAX_ERROR_SV = """module counter(
    input wire clk,
    output reg [WIDTH-1:0] count
);

    always @(posedge clk) begin
        count <= count + 1  // missing semicolon
    end
endmodule
"""


class TestSyntaxAgent:
    def test_syntax_pass(self):
        agent = SyntaxAgent()
        result = agent.execute({"counter.sv": VALID_COUNTER_SV})
        assert result.pass_
        assert len(result.errors) == 0

    def test_syntax_fail(self):
        agent = SyntaxAgent()
        result = agent.execute({"counter.sv": SYNTAX_ERROR_SV})
        assert not result.pass_
        assert len(result.errors) > 0

    def test_multiple_files(self):
        agent = SyntaxAgent()
        sub_sv = "module sub(input wire a, output wire b); assign b = ~a; endmodule\n"
        result = agent.execute({"top.sv": VALID_COUNTER_SV, "sub.sv": sub_sv})
        assert result.pass_

    def test_error_parsing(self):
        agent = SyntaxAgent()
        result = agent.execute({"bad.sv": SYNTAX_ERROR_SV})
        error_messages = " ".join(e.get("message", "").lower() for e in result.errors)
        assert any(
            kw in error_messages for kw in ["syntax", "error", "width", "undefined"]
        )

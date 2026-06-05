from src.agents.base import SubAgent, AgentResult
from src.llm import LLMBackend

class DiagnosisAgent(SubAgent):
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict | None = None,
    ) -> AgentResult:
        if not config or "errors" not in config:
            return AgentResult(pass_=True)

        errors = config["errors"]
        spec = config.get("spec", "")
        
        # Get the first RTL file content
        rtl_content = list(rtl_files.values())[0] if rtl_files else ""
        
        error_text = "\n".join(
            f"  Line {e.get('line', '?')}: {e.get('message', str(e))}"
            for e in errors
        )

        system_prompt = (
            "[DIAGNOSIS PROTOCOL]\n"
            "ROLE: Expert Verilog/SystemVerilog RTL Debug and Code Analyzer\n"
            "OBJECTIVE: Perform deep root-cause analysis on compiler/simulation failures.\n\n"
            "1. PRE-CONDITIONS:\n"
            "   - Buggy Verilog code, Spec, and compiler/simulation error traceback are provided.\n\n"
            "2. OBLIGATIONS:\n"
            "   - Traceability: Analyze why each error occurred and identify the exact buggy line(s).\n"
            "   - Rectification: Plan precise changes required to satisfy the Design Spec and fix compiling/sim issues.\n\n"
            "3. POST-CONDITIONS (OUTPUT FORMAT):\n"
            "   - The analysis MUST be formatted as a structured TODO List:\n"
            "     - TODO [N]: [Line number/Location] [Description of bug] -> [Precise fix recommendation]\n"
            "[/DIAGNOSIS PROTOCOL]"
        )

        user_prompt = (
            f"## Verilog Code:\n"
            f"```verilog\n{rtl_content}\n```\n\n"
            f"## Errors / Failures:\n"
            f"{error_text}\n\n"
            f"## Original Design Specification:\n"
            f"{spec}\n\n"
            f"Please analyze the errors, locate the buggy code, and write a clear diagnostic report indicating what needs to be changed."
        )

        response = self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        return AgentResult(
            pass_=False,
            raw_output=response.text,
        )

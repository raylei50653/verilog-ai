from src.agents.base import SubAgent, AgentResult
from src.cancellation import CancellationToken
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
        source_context = config.get("source_context", [])
        testbench_context = config.get("testbench_context", [])
        cancel_token: CancellationToken | None = config.get("cancel_token")

        rtl_content = list(rtl_files.values())[0] if rtl_files else ""

        MAX_ERRORS_SHOWN = 10
        truncated = len(errors) > MAX_ERRORS_SHOWN
        shown_errors = errors[:MAX_ERRORS_SHOWN]
        error_text = "\n".join(
            f"  Line {e.get('line', '?')}: {e.get('message', str(e))}"
            for e in shown_errors
        )
        if truncated:
            error_text += f"\n  ... ({len(errors) - MAX_ERRORS_SHOWN} more errors omitted — fix root causes first)"

        system_prompt = (
            "You are an expert Verilog RTL debugger. Analyze the errors below and produce an ordered TODO list for fixing them.\n\n"
            "Rules:\n"
            "- Group errors that share the same root cause into one TODO.\n"
            "- Every TODO must describe a concrete RTL code change — not an observation or question.\n"
            "- LOCATION must be a specific line number or named block.\n"
            "- Ignore errors about the test runner, pytest, or simulation infrastructure. Do not create TODOs for them.\n"
            "- For multi-driver conflicts, write ONE TODO to merge the drivers into a single block.\n"
            "- For `<=` in combinational blocks, write ONE TODO to move the assignments to the clocked block.\n\n"
            "Output format — one block per TODO, no extra text:\n\n"
            "TODO N\n"
            "LOCATION: <line number or block name>\n"
            "BUG: <what is wrong>\n"
            "FIX: <what to change>"
        )

        user_prompt = (
            f"## Verilog Code:\n"
            f"```verilog\n{rtl_content}\n```\n\n"
            f"## Errors / Failures:\n"
            f"{error_text}\n\n"
        )
        if source_context:
            user_prompt += "## Error Source Context:\n"
            for ctx in source_context:
                user_prompt += (
                    f"File: {ctx.get('path')}\n"
                    f"Error line: {ctx.get('error_line')}\n"
                    f"Message: {ctx.get('message')}\n"
                    "```text\n"
                )
                for line in ctx.get("excerpt", []):
                    user_prompt += f"{line['line']:>4}: {line['text']}\n"
                user_prompt += "```\n\n"
        if testbench_context:
            user_prompt += "## Testbench Context:\n"
            for ctx in testbench_context:
                user_prompt += (
                    f"File: {ctx.get('path')}\n"
                    f"Focus line: {ctx.get('focus_line')}\n"
                    "```text\n"
                )
                for line in ctx.get("excerpt", []):
                    user_prompt += f"{line['line']:>4}: {line['text']}\n"
                user_prompt += "```\n\n"
        user_prompt += (
            f"## Original Design Specification:\n"
            f"{spec}\n\n"
            f"Please analyze the errors, locate the buggy code, and write a clear diagnostic report indicating what needs to be changed."
        )

        response = self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            cancel_token=cancel_token,
        )

        return AgentResult(
            pass_=False,
            raw_output=response.text,
        )

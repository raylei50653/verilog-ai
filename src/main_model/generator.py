import re
from typing import Callable, Any
from src.llm import LLMBackend, LLMResponse, build_fix_tools, _local_storage
from src.cancellation import CancellationToken


class ModelGenerator:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def generate_rtl(
        self,
        spec: str,
        system_message: str | None = None,
        params: dict | None = None,
        constraints: dict | None = None,
        interface_specs: dict | None = None,
        examples: list[dict] | None = None,
        testbench_files: dict[str, str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> tuple[str, LLMResponse]:
        if system_message is None:
            system_message = (
                "Expert Verilog/SystemVerilog RTL designer. Rules:\n"
                "- Reset: active-high `if (rst)`. Sync by default (`always_ff @(posedge clk)`); async only if spec says so (`always_ff @(posedge clk or posedge rst)`). Init all regs in reset.\n"
                "- No self-assignment (`reg <= reg`). No `===`/`!==` (use `==`/`!=`). No `#delays`, `initial`, or `specify`.\n"
                "- Every `if` needs `else` or a prior default; every `case` needs `default`.\n"
                "- Single-driver: one always block per signal. Multi-clock: each domain's sync chain gets distinct signal names.\n"
                "- `=` in `always_comb`, `<=` in `always_ff`. Never mix.\n"
                "- FSM (only if spec has state): two blocks — clocked (state reg + registered outputs), combinational (next-state + combinational outputs). Comb block drives only `next_state` and combinational outputs; datapath regs go in the clocked block.\n"
                "- No added regs or pipeline stages the spec doesn't require. Combinational outputs use `assign` or `always_comb`.\n"
                "- `.sv` files: `logic`, `always_ff`, `always_comb`. All signals declared at module level. Ports match spec exactly.\n"
                "- No comments. Output: one ```verilog ... ``` block, nothing else."
            )

        user_prompt = self._build_prompt(
            spec, params, constraints, interface_specs, examples, testbench_files
        )

        if enable_thinking and hasattr(self.backend, "generate_with_thinking"):
            response = self.backend.generate_with_thinking(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
                cancel_token=cancel_token,
            )
        else:
            response = self.backend.generate(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
                cancel_token=cancel_token,
            )

        code = self._extract_verilog(response.text)
        return code, response

    def fix_errors(
        self,
        original_spec: str,
        current_code: str,
        errors: list[dict],
        diagnosis_report: str | None = None,
        source_context: list[dict] | None = None,
        testbench_context: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> tuple[str, "LLMResponse"]:
        system_message = (
            "You are an expert Verilog RTL debug engineer. Apply the minimal fixes needed to resolve the TODOs below.\n\n"
            "Rules:\n"
            "- For each TODO: first locate the exact LOCATION and SNIPPET in the code, read the surrounding context (at least 5 lines above and below), then apply the fix. Never patch a line you haven't read.\n"
            "- Fix each TODO in order using the smallest targeted change. Do not rewrite unrelated logic.\n"
            "- Preserve the original reset style (sync vs async) unless a TODO explicitly requires changing it.\n"
            "- Active-high reset: `if (rst)`. Correct any `if (!rst)` or `if (~rst)` you encounter.\n"
            "- Eliminate self-assignment patterns (`X <= X`).\n"
            "- Ensure every `if`/`else if`/`else` chain is syntactically closed.\n"
            "- Single-driver: if two always blocks drive the same signal, merge them into one.\n"
            "- No `<=` inside `always @(*)` or `always_comb`. Move register updates to the clocked block.\n"
            "- FSM canonical form: state register and registered outputs in one clocked block; next-state decode and combinational outputs in one combinational block.\n"
            "- Skip any TODO that describes a test runner, pytest, or simulation infrastructure error — output the current code unchanged for that item.\n"
            "- The corrected module must remain strictly synthesizable.\n"
            "- If your fix introduces any new signal, wire, or variable, declare it at the top of the module before use.\n\n"
            "Output: one ```verilog ... ``` block containing the complete corrected module. No explanations, no change log."
        )

        error_text = "\n".join(
            f"  Line {e.get('line', '?')}: {e.get('message', str(e))}"
            for e in errors
        )

        user_prompt = (
            f"The following Verilog code has errors:\n\n"
            f"```verilog\n{current_code}\n```\n\n"
            f"Errors:\n{error_text}\n\n"
        )
        if diagnosis_report:
            user_prompt += (
                f"## Sub-Agent Diagnosis TODO List:\n"
                f"{diagnosis_report}\n\n"
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
            f"Original specification: {original_spec}\n\n"
            "Fix all identified bugs while keeping other correct logic unchanged.\n"
            "Output the complete corrected Verilog module."
        )

        if enable_thinking and hasattr(self.backend, "generate_with_thinking"):
            response = self.backend.generate_with_thinking(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
                cancel_token=cancel_token,
            )
        else:
            response = self.backend.generate(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
                cancel_token=cancel_token,
            )

        code = self._extract_verilog(response.text)
        return code or current_code, response

    def _build_prompt(
        self,
        spec: str,
        params: dict | None,
        constraints: dict | None,
        interface_specs: dict | None,
        examples: list[dict] | None,
        testbench_files: dict[str, str] | None = None,
    ) -> str:
        parts: list[str] = []

        if params:
            parts.append("## Design Parameters\n")
            for key, value in params.items():
                parts.append(f"- {key} = {value}")
            parts.append("")

        if constraints:
            parts.append("## Timing and Design Constraints\n")
            for key, value in constraints.items():
                parts.append(f"- {key}: {value}")
            parts.append("")

        if interface_specs:
            parts.append("## Interface Specification\n")
            if "signals" in interface_specs:
                parts.append("Port signals:")
                for sig_name, sig_dir in interface_specs["signals"].items():
                    parts.append(f"  - {sig_name}: {sig_dir}")
            parts.append("")

        if examples:
            parts.append("## Reference Examples\n")
            for i, ex in enumerate(examples):
                parts.append(f"### Example {i + 1}")
                parts.append(f"Spec: {ex.get('spec', 'N/A')}")
                parts.append(f"```verilog\n{ex.get('code', 'N/A')}\n```")
            parts.append("")

        parts.append("## Specification\n")
        parts.append(spec)
        parts.append("")

        if testbench_files:
            parts.append("## Testbench (read-only — use this to understand expected port names, signal types, and behavior):\n")
            for tb_name, tb_content in testbench_files.items():
                parts.append(f"### {tb_name}")
                parts.append(f"```verilog\n{tb_content}\n```")
            parts.append("")

        parts.append("Generate the complete Verilog RTL module based on the specification above.")

        return "\n".join(parts)

    def _extract_verilog(self, text: str) -> str:
        pattern = re.compile(r"```(?:verilog|systemverilog|sv)?\s*\n(.*?)```", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[0].strip()

        module_pattern = re.compile(r"^\s*module\s+\w+", re.MULTILINE)
        endmodule_pattern = re.compile(r"^\s*endmodule\b", re.MULTILINE)

        module_match = module_pattern.search(text)
        endmodule_match = endmodule_pattern.search(text)

        if module_match and endmodule_match:
            return text[module_match.start() : endmodule_match.end() + len("endmodule")].strip()

        return text.strip()

    def create_plan(
        self,
        spec: str,
        params: dict | None = None,
        constraints: dict | None = None,
    ) -> list[str]:
        system_message = (
            "You are a master hardware architect and planner.\n"
            "Your task is to break down the user's Verilog RTL design specification into a list of logical implementation steps.\n"
            "Provide a clean, numbered list of steps (e.g. 1. Define ports, 2. Implement reset logic...) with no other text."
        )
        user_prompt = (
            f"Specification: {spec}\n"
            f"Parameters: {params}\n"
            f"Constraints: {constraints}\n\n"
            f"Break this down into 3 to 5 logical design steps."
        )
        response = self.backend.generate(
            system_prompt=system_message,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        steps = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            clean = re.sub(r"^\d+[\.\)\s:-]+", "", line)
            if clean:
                steps.append(clean)
        if not steps:
            steps = ["Define ports and parameters", "Implement core hardware logic", "Verify and initialize registers"]
        return steps

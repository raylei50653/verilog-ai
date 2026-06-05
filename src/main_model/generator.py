import re
from typing import Callable, Any
from src.llm import LLMBackend, LLMResponse


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
        temperature: float = 0.2,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, LLMResponse]:
        if system_message is None:
            system_message = (
                "[DESIGN CONTRACT]\n"
                "ROLE: Expert Verilog/SystemVerilog RTL Design Engineer\n"
                "OBJECTIVE: Generate high-quality, synthesizable, and robust RTL code based on Client Spec.\n\n"
                "1. PRE-CONDITIONS:\n"
                "   - Client Spec, Parameters, and Constraints are provided in the prompt.\n\n"
                "2. HARD DESIGN INVARIANTS (TECHNICAL CONSTRAINTS):\n"
                "   - Synthesizability: Use standard synthesizable constructs only. No #delays, initial blocks, or Specify blocks.\n"
                "   - Register Initialization: All sequential registers MUST be initialized inside a reset block.\n"
                "   - Declarations: All ports, wires, registers, and internal variables must be declared explicitly.\n"
                "   - Port Matching: Port names, directions, and parameters must match the specification exactly.\n\n"
                "3. POST-CONDITIONS (OUTPUT PROTOCOL):\n"
                "   - Code Wrap: Output the complete Verilog code inside a single ```verilog ... ``` block.\n"
                "   - Zero Conversational Text: Do not output any explanation, commentary, greetings, or markdown outside the code block.\n"
                "[/DESIGN CONTRACT]"
            )

        user_prompt = self._build_prompt(
            spec, params, constraints, interface_specs, examples
        )

        if enable_thinking and hasattr(self.backend, "generate_with_thinking"):
            response = self.backend.generate_with_thinking(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
            )
        else:
            response = self.backend.generate(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
            )

        code = self._extract_verilog(response.text)
        return code, response

    def fix_errors(
        self,
        original_spec: str,
        current_code: str,
        errors: list[dict],
        diagnosis_report: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, "LLMResponse"]:
        system_message = (
            "[DEBUGGING CONTRACT]\n"
            "ROLE: Expert Verilog RTL Debug and Rectification Engineer\n"
            "OBJECTIVE: Analyze compile/simulation errors and apply precise bug fixes.\n\n"
            "1. PRE-CONDITIONS:\n"
            "   - Buggy Verilog code, compiler/simulation error list, and Sub-Agent TODO List are provided.\n\n"
            "2. HARD DEBUGGING INVARIANTS:\n"
            "   - Regression Prevention: Fix identified bugs while preserving all other correct logic and interface specs.\n"
            "   - Synthesizability: The corrected module must remain strictly synthesizable and compile-ready.\n\n"
            "3. POST-CONDITIONS (OUTPUT PROTOCOL):\n"
            "   - Code Wrap: Output the fully corrected Verilog code inside a single ```verilog ... ``` block.\n"
            "   - Zero Conversational Text: Do not write any explanations, change-logs, or chat outside the code block.\n"
            "[/DEBUGGING CONTRACT]"
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
        user_prompt += (
            f"Original specification: {original_spec}\n\n"
            f"Please carefully address each TODO item in the Sub-Agent TODO List one by one. "
            f"Ensure all identified bugs are fixed, while keeping other correct logic unchanged, and output the fully corrected Verilog code."
        )

        if hasattr(self.backend, 'generate_with_thinking'):
            response = self.backend.generate_with_thinking(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
            )
        else:
            response = self.backend.generate(
                system_prompt=system_message,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_token=on_token,
            )

        code = self._extract_verilog(response.text)
        return code, response

    def _build_prompt(
        self,
        spec: str,
        params: dict | None,
        constraints: dict | None,
        interface_specs: dict | None,
        examples: list[dict] | None,
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

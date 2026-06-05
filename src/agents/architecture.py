import re
from src.agents.base import SubAgent, AgentResult
from src.llm import LLMBackend

class ArchitectureAgent(SubAgent):
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict | None = None,
    ) -> AgentResult:
        if not config:
            return AgentResult(pass_=True)

        spec = config.get("spec", "")
        params = config.get("params", {})
        constraints = config.get("constraints", {})

        system_prompt = (
            "[ARCHITECTURE PROTOCOL]\n"
            "ROLE: Lead Hardware Architect & RTL Design Planner\n"
            "OBJECTIVE: Define a precise module interface and architecture specification before implementation.\n\n"
            "1. OBLIGATIONS:\n"
            "   - Interface Specification: Define all parameters (name, default value) and port signals (name, direction, bit-width, description).\n"
            "   - Reset & Clocking: Define active-low/high reset, synchronous/asynchronous reset, clocking requirements.\n"
            "   - Implementation Plan: Break down the design into a list of TODO tasks.\n\n"
            "2. OUTPUT FORMAT:\n"
            "   Your output MUST follow this Markdown template exactly:\n"
            "   # RTL Architecture Specification\n"
            "   ## Parameters\n"
            "   | Parameter | Default Value | Description |\n"
            "   | --- | --- | --- |\n"
            "   | ... | ... | ... |\n\n"
            "   ## Port Signals\n"
            "   | Port Name | Direction | Bit-width | Description |\n"
            "   | --- | --- | --- | --- |\n"
            "   | ... | ... | ... | ... |\n\n"
            "   ## Implementation TODOs\n"
            "   - TODO 1: ...\n"
            "   - TODO 2: ...\n"
            "[/ARCHITECTURE PROTOCOL]"
        )

        user_prompt = (
            f"## Design Specification:\n"
            f"{spec}\n\n"
            f"## Given Parameters:\n"
            f"{params}\n\n"
            f"## Given Constraints:\n"
            f"{constraints}\n\n"
            f"Analyze the design requirement, define the parameters, ports, reset behavior, and list 3-5 logical TODO steps to implement this."
        )

        response = self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        return AgentResult(
            pass_=True,
            raw_output=response.text,
        )

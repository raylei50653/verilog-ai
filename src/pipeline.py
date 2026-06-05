import json
import time
import uuid
from pathlib import Path
from typing import Any

from src.llm import LLMBackend
from src.agents.base import SubAgent, AgentResult
from src.agents.syntax import SyntaxAgent
from src.agents.simulation import SimulationAgent
from src.agents.ppa import PPAAgent
from src.agents.diagnosis import DiagnosisAgent
from src.agents.architecture import ArchitectureAgent
from src.main_model.generator import ModelGenerator
from src.mcp.server import MCPServer, TrialRecord
from src.cvdp.scoring import TrialScore


class TrialRunner:
    def __init__(
        self,
        backend: LLMBackend,
        mcp: MCPServer | None = None,
        syntax: SyntaxAgent | None = None,
        simulation: SimulationAgent | None = None,
        ppa: PPAAgent | None = None,
        diagnosis: DiagnosisAgent | None = None,
        architecture: ArchitectureAgent | None = None,
    ):
        self.backend = backend
        self.mcp = mcp or MCPServer()
        self.syntax = syntax or SyntaxAgent()
        self.simulation = simulation or SimulationAgent()
        self.ppa = ppa or PPAAgent()
        self.diagnosis = diagnosis or DiagnosisAgent(backend)
        self.architecture = architecture or ArchitectureAgent(backend)
        self.generator = ModelGenerator(backend)

    def run_trial(
        self,
        spec: str,
        problem_id: str = "unknown",
        params: dict[str, Any] | None = None,
        context_files: dict[str, str] | None = None,
        testbench_files: dict[str, str] | None = None,
        system_message: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        max_retries: int = 3,
        enable_rtl_reuse: bool = False,
        verbose: bool = False,
        on_step_change: Any | None = None,
        on_token: Any | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, TrialScore]:
        if on_step_change:
            on_step_change("plan", "active")
            on_step_change("gen", "inactive")
            on_step_change("syntax", "inactive")
            on_step_change("sim", "inactive")
            on_step_change("ppa", "inactive")
            on_step_change("db", "inactive")

        trial_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        latest_diagnosis_report = None

        constraints = self.mcp.get_constraints(problem_id.split("_")[0]
            if "_" in problem_id else "default")

        # AI Planning: Architecture sub-agent designs the interface & spec
        architecture_spec = None
        plan_todos = []
        try:
            if on_step_change:
                on_step_change("plan", "active")
            arch_result = self.architecture.execute(
                rtl_files={},
                config={"spec": spec, "params": params, "constraints": constraints}
            )
            architecture_spec = arch_result.raw_output

            # Extract TODOs from the architecture specification
            plan_todos = [line.strip() for line in architecture_spec.splitlines() if "todo" in line.lower()]
            if not plan_todos:
                plan_todos = ["Define ports and parameters", "Implement core hardware logic", "Verify and initialize registers"]

            self.mcp.init_workbench(trial_id, problem_id, spec, plan_todos, architecture_spec)
            if on_step_change:
                on_step_change("plan", "success")
        except Exception:
            if on_step_change:
                on_step_change("plan", "fail")

        # RTL Reuse: find similar successful trials
        examples = []
        if enable_rtl_reuse:
            # Detect pattern keywords from problem_id
            keywords = ["fifo", "arbiter", "counter", "pipeline", "mult", "fsm", "adder", "mux"]
            pattern = None
            for kw in keywords:
                if kw in problem_id.lower():
                    pattern = kw
                    break
            
            if pattern:
                successful_trials = self.mcp.get_successful_trials_by_pattern(pattern)
                for t in successful_trials:
                    if t.get("generated_code") and len(examples) < 2:
                        examples.append({
                            "spec": t.get("spec_prompt", ""),
                            "code": t.get("generated_code", ""),
                        })
                if verbose and examples:
                    print(f"  [RTL Reuse] Loaded {len(examples)} examples matching pattern: {pattern}")

        # AI Reads TODO & Architecture Spec from Workbench to guide main Agent
        augmented_spec = spec
        try:
            wb_data = self.mcp.read_workbench()
            arch_spec = wb_data.get("architecture_spec")
            pending_todos = [t["description"] for t in wb_data.get("todos", []) if t["status"] == "pending"]

            spec_parts = []
            if arch_spec:
                spec_parts.append(arch_spec)
                spec_parts.append("\n---\n")

            spec_parts.append("## Original Design Specification")
            spec_parts.append(spec)

            if pending_todos:
                todo_text = "\n".join(f"- TODO: {todo}" for todo in pending_todos)
                spec_parts.append("## Implementation Tasks (TODO List)")
                spec_parts.append(todo_text)

            augmented_spec = "\n\n".join(spec_parts)
        except Exception:
            pass

        if on_token:
            on_token("")
        if on_step_change:
            on_step_change("gen", "active")
        code, _ = self.generator.generate_rtl(
            spec=augmented_spec,
            system_message=system_message,
            params=params,
            constraints=constraints,
            examples=examples if examples else None,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            on_token=on_token,
        )
        
        # Mark initial TODO tasks as completed
        try:
            self.mcp.update_workbench_status("completed_initial", mark_all_completed=True)
        except Exception:
            pass
        if on_step_change:
            on_step_change("gen", "success")

        # Determine RTL filename from CVDP problem if available
        rtl_filename = "rtl.sv"
        if problem_id and problem_id.startswith("cvdp_"):
            try:
                from src.cvdp.loader import CVDPDataset
                cvdp = CVDPDataset()
                problem = cvdp.get_by_id(problem_id)
                if problem and problem.output_files:
                    rtl_filename = problem.output_files[0]
            except Exception:
                pass

        score = TrialScore(problem_id=problem_id, trial_index=0)
        score.syntax_pass = False
        score.simulation_pass = False
        actual_retries = 0

        for retry in range(max_retries + 1):
            actual_retries = retry
            # 1. Syntax check
            if on_step_change:
                on_step_change("syntax", "active")
            syntax_result = self.syntax.execute({rtl_filename: code})
            if verbose:
                print(f"  [{retry}] Syntax: {'PASS' if syntax_result.pass_ else 'FAIL'}")

            if not syntax_result.pass_:
                if on_step_change:
                    on_step_change("syntax", "fail")
                if retry >= max_retries:
                    score.syntax_pass = False
                    score.errors = syntax_result.errors
                    break
                # Fix syntax errors
                if on_step_change:
                    on_step_change("gen", "active")
                if verbose:
                    print("  [Sub-Agent] Debugging syntax errors...")
                diagnosis_result = self.diagnosis.execute(
                    rtl_files={rtl_filename: code},
                    config={"errors": syntax_result.errors, "spec": spec}
                )
                diagnosis_report = diagnosis_result.raw_output
                latest_diagnosis_report = diagnosis_report
                if verbose:
                    print(f"  [Sub-Agent] Diagnosis Report:\n{diagnosis_report}\n")

                # Parse TODOs from diagnosis_report and add to Workbench
                try:
                    debug_todos = [line.strip() for line in diagnosis_report.splitlines() if "todo" in line.lower()]
                    if debug_todos:
                        self.mcp.add_debug_todos(debug_todos)
                except Exception:
                    pass

                if on_token:
                    on_token("")
                code, _ = self.generator.fix_errors(
                    original_spec=spec,
                    current_code=code,
                    errors=syntax_result.errors,
                    diagnosis_report=diagnosis_report,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    on_token=on_token,
                )

                # Mark debug TODO tasks as completed
                try:
                    self.mcp.update_workbench_status("completed_syntax_debug", mark_all_completed=True)
                except Exception:
                    pass
                if on_step_change:
                    on_step_change("gen", "success")
                continue

            score.syntax_pass = True
            if on_step_change:
                on_step_change("syntax", "success")

            # 2. Simulation check
            if not testbench_files:
                score.simulation_pass = False
                score.errors = [{"message": "No testbench available"}]
                break

            if on_step_change:
                on_step_change("sim", "active")
            sim_result = self.simulation.execute({rtl_filename: code}, testbench_files)
            if verbose:
                print(f"  [{retry}] Simulation: {'PASS' if sim_result.pass_ else 'FAIL'}")

            if sim_result.pass_:
                score.simulation_pass = True
                if on_step_change:
                    on_step_change("sim", "success")
                break

            # Simulation failed
            if on_step_change:
                on_step_change("sim", "fail")
            if retry >= max_retries:
                score.simulation_pass = False
                score.errors = sim_result.errors
                break

            # Fix simulation failures
            if on_step_change:
                on_step_change("gen", "active")
            if verbose:
                print("  [Sub-Agent] Debugging simulation failures...")
            diagnosis_result = self.diagnosis.execute(
                rtl_files={rtl_filename: code},
                testbench_files=testbench_files,
                config={"errors": sim_result.errors, "spec": spec}
            )
            diagnosis_report = diagnosis_result.raw_output
            latest_diagnosis_report = diagnosis_report
            if verbose:
                print(f"  [Sub-Agent] Diagnosis Report:\n{diagnosis_report}\n")

            # Parse TODOs from diagnosis_report and add to Workbench
            try:
                debug_todos = [line.strip() for line in diagnosis_report.splitlines() if "todo" in line.lower()]
                if debug_todos:
                    self.mcp.add_debug_todos(debug_todos)
            except Exception:
                pass

            if on_token:
                on_token("")
            code, _ = self.generator.fix_errors(
                original_spec=spec,
                current_code=code,
                errors=sim_result.errors,
                diagnosis_report=diagnosis_report,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                on_token=on_token,
            )

            # Mark debug TODO tasks as completed
            try:
                self.mcp.update_workbench_status("completed_sim_debug", mark_all_completed=True)
            except Exception:
                pass
            if on_step_change:
                on_step_change("gen", "success")

        # 3. PPA check (only run if compilation succeeded on the final code)
        ppa_result = AgentResult(pass_=False, metrics={})
        if score.syntax_pass:
            if on_step_change:
                on_step_change("ppa", "active")
            import re
            top_mod = "top"
            match = re.search(r"\bmodule\s+(\w+)", code)
            if match:
                top_mod = match.group(1)
            ppa_result = self.ppa.execute({rtl_filename: code}, config={"top_module": top_mod})
            if on_step_change:
                on_step_change("ppa", "success" if ppa_result.pass_ or score.syntax_pass else "fail")

        score.ppa_metrics = ppa_result.metrics
        if verbose and score.syntax_pass:
            area = ppa_result.metrics.get("area", "?")
            print(f"  PPA: area={area} cells")

        score.duration_ms = (time.monotonic() - start) * 1000
        score.retry_count = actual_retries

        if on_step_change:
            on_step_change("db", "active")
        self.mcp.write_trial(TrialRecord(
            trial_id=trial_id,
            problem_id=problem_id,
            params=params or {},
            spec_prompt=spec,
            generated_code=code,
            syntax_pass=score.syntax_pass,
            simulation_pass=score.simulation_pass,
            ppa_score=score.ppa_metrics,
            pass_=score.pass_,
            retry_count=actual_retries,
            duration_ms=score.duration_ms,
            diagnosis_report=latest_diagnosis_report,
        ))
        if on_step_change:
            on_step_change("db", "success")

        return code, score

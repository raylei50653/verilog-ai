import json
import re
import time
import uuid
from typing import Any

from src.agents.base import AgentResult
from src.agents.diagnosis import DiagnosisAgent
from src.agents.simulation import SimulationAgent
from src.agents.syntax import SyntaxAgent
from src.agents.ppa import PPAAgent
from src.cvdp.scoring import TrialScore
from src.cancellation import CancellationToken
from src.llm import LLMBackend
from src.main_model.generator import ModelGenerator
from src.mcp.server import MCPServer, TrialRecord


class TrialRunner:
    def __init__(
        self,
        backend: LLMBackend,
        mcp: MCPServer | None = None,
        syntax: SyntaxAgent | None = None,
        simulation: SimulationAgent | None = None,
        diagnosis: DiagnosisAgent | None = None,
        ppa: PPAAgent | None = None,
        vivado_agent: Any | None = None,
    ):
        self.backend = backend
        self.mcp = mcp or MCPServer()
        self.syntax = syntax or SyntaxAgent()
        self.simulation = simulation or SimulationAgent()
        self.diagnosis = diagnosis or DiagnosisAgent(backend)
        self.ppa = ppa or PPAAgent()
        self.generator = ModelGenerator(backend)
        self.vivado_agent = vivado_agent

    def _build_prompt(
        self,
        spec: str,
        context_files: dict[str, str] | None = None,
    ) -> str:
        if not context_files:
            return spec

        parts = ["## Original Design Specification", spec]
        for path, content in context_files.items():
            parts.extend(
                [
                    f"## Context File: {path}",
                    "```text",
                    content,
                    "```",
                ]
            )
        return "\n\n".join(parts)

    def _build_todo_report(self, todo: dict[str, Any]) -> str:
        location = todo.get("location", {}).get("raw") or "unknown location"
        bug = todo.get("bug", "").strip()
        snippet = todo.get("snippet", "").strip()
        fix = todo.get("fix", "").strip()
        review = todo.get("review", "").strip() or "Verify the targeted bug is resolved without changing unrelated logic."
        snippet_part = f" `{snippet}`" if snippet else ""
        return f"TODO [{todo.get('order', '?')}]: [{location}] {bug}{snippet_part} -> {fix} | REVIEW: {review}"

    def _find_snippet_line(self, rtl_path: str, snippet: str) -> int | None:
        if not snippet:
            return None
        lines = self.mcp.read_file_lines(rtl_path)["lines"]
        normalized_snippet = " ".join(snippet.split())
        for line in lines:
            normalized_text = " ".join(line["text"].split())
            if normalized_snippet and normalized_snippet in normalized_text:
                return line["line"]
        return None

    def _build_todo_source_context(self, rtl_path: str, todo: dict[str, Any]) -> list[dict[str, Any]]:
        location = todo.get("location", {})
        snippet_line = self._find_snippet_line(rtl_path, todo.get("snippet", ""))
        line_start = snippet_line or location.get("line_start")
        line_end = snippet_line or location.get("line_end")
        if not isinstance(line_start, int) or line_start < 1:
            return []
        excerpt = self.mcp.read_file_lines(
            rtl_path,
            start_line=max(1, line_start - 10),
            end_line=(line_end or line_start) + 10,
        )
        return [
            {
                "path": excerpt["path"],
                "error_line": line_start,
                "message": todo.get("bug", ""),
                "excerpt": excerpt["lines"],
            }
        ]

    def _review_fix(
        self,
        spec: str,
        code: str,
        current_todo: dict[str, Any],
        trial_id: str,
        testbench_context: list[dict[str, Any]] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        todos = self.mcp.read_todos(trial_id)
        if not todos:
            return {"updates": [], "new_todos": []}

        todo_list_text = "\n".join(
            f"  {t['id']} [status={t.get('status','?')}] BUG={t.get('bug','?')} FIX={t.get('fix','?')}"
            for t in todos
        )

        system_prompt = (
            "You are a strict Verilog code reviewer. Assess every TODO against the actual post-fix code.\n\n"
            "Rules:\n"
            "- Include every existing TODO in the \"updates\" list — no omissions.\n"
            "- Status values: \"done\" (verified fixed), \"pending\" (still broken or incomplete), \"failed\" (unfixable as stated).\n"
            "- To mark \"done\": you MUST find the corrected code in the listing and confirm the bug described in BUG no longer exists. If you cannot locate the fix in the code, mark \"pending\".\n"
            "- Default to \"pending\" when in doubt. Never mark \"done\" based on intent — only on what the code actually shows.\n"
            "- If the fix introduced a new bug, add it to \"new_todos\" with location, bug, and fix fields.\n\n"
            "Output: valid JSON only — no markdown, no text outside the JSON.\n\n"
            "{\n"
            "  \"updates\": [\n"
            "    {\"id\": \"todo-1\", \"status\": \"done\", \"reason\": \"<cite the specific line or construct that proves it is fixed>\"},\n"
            "    {\"id\": \"todo-2\", \"status\": \"pending\", \"reason\": \"<cite what is still wrong in the code>\"}\n"
            "  ],\n"
            "  \"new_todos\": []\n"
            "}"
        )

        user_prompt = (
            f"## Original Specification:\n{spec}\n\n"
            f"## Fixed Verilog Code:\n```verilog\n{code}\n```\n\n"
            f"## TODO That Was Just Fixed:\n"
            f"  ID={current_todo['id']} BUG={current_todo.get('bug','?')} FIX={current_todo.get('fix','?')}\n\n"
            f"## Full Current TODO List:\n{todo_list_text}\n\n"
        )
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
            "Review the fixed code and return a JSON block with your assessment of ALL TODOs. "
            "Be strict: only mark a TODO as 'done' if the bug is definitively resolved."
        )

        response = self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=4096,
            cancel_token=cancel_token,
        )
        return self._parse_review_response(response.text, current_todo["id"], todos)

    def _parse_review_response(
        self,
        text: str,
        current_todo_id: str,
        existing_todos: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = {
            "updates": [{"id": current_todo_id, "status": "done", "reason": "fix applied (review parsing fallback)"}],
            "new_todos": [],
        }

        import re as _re
        json_match = _re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            return fallback

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return fallback

        updates = data.get("updates", [])
        new_todos = data.get("new_todos", [])

        if not updates:
            return fallback

        existing_ids = {t["id"] for t in existing_todos}
        validated_updates = []
        for u in updates:
            uid = u.get("id", "")
            if not isinstance(uid, str) or uid not in existing_ids:
                continue
            status = u.get("status", "pending")
            if status not in ("done", "pending", "failed"):
                status = "pending"
            validated_updates.append({
                "id": uid,
                "status": status,
                "reason": u.get("reason", ""),
            })

        if not validated_updates:
            validated_updates = fallback["updates"]

        return {"updates": validated_updates, "new_todos": new_todos}

    def _consolidate_todos(
        self,
        todos: list[dict[str, Any]],
        spec: str,
        code: str,
        cancel_token: CancellationToken | None = None,
    ) -> list[dict[str, Any]]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        todo_text = "\n".join(
            f"TODO {t['order']} | {t.get('location', {}).get('raw', '?')} | "
            f"BUG: {t.get('bug', '')} | FIX: {t.get('fix', '')}"
            for t in todos
        )
        system_prompt = (
            "You are a Verilog RTL triage specialist. Reduce the TODO list below to at most 5 high-impact items.\n\n"
            "Rules:\n"
            "- Merge TODOs that share the same root cause into one.\n"
            "- Prioritize by impact: fixing one item should implicitly resolve downstream errors.\n"
            "- Output at most 5 TODOs using exactly this format:\n"
            "  TODO N\n"
            "  LOCATION: <line or block>\n"
            "  BUG: <description>\n"
            "  FIX: <action>\n"
            "- No text outside the TODO blocks."
        )
        user_prompt = (
            f"## Original TODO List ({len(todos)} items):\n{todo_text}\n\n"
            f"## Verilog Code:\n```verilog\n{code}\n```\n\n"
            f"## Spec:\n{spec}\n\n"
            "Consolidate the above into at most 5 essential TODOs."
        )
        try:
            response = self.backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=2048,
                cancel_token=cancel_token,
            )
            consolidated = self.mcp.parse_todo_report(response.text)
            if consolidated:
                print(f"[consolidate] {len(todos)} → {len(consolidated)} TODOs")
                return consolidated
        except Exception:
            pass
        return todos[:5]

    def _diagnose_and_write_todos(
        self,
        trial_id: str,
        rtl_filename: str,
        code: str,
        spec: str,
        errors: list[dict[str, Any]],
        rtl_path: str,
        testbench_files: dict[str, str] | None = None,
        testbench_context: list[dict[str, Any]] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> str:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        source_context = self.mcp.get_error_source_context(rtl_path, errors)
        diagnosis_result = self.diagnosis.execute(
            rtl_files={rtl_filename: code},
            testbench_files=testbench_files,
            config={
                "errors": errors,
                "spec": spec,
                "source_context": source_context,
                "testbench_context": testbench_context or [],
                "cancel_token": cancel_token,
            },
        )
        diagnosis_report = diagnosis_result.raw_output
        todos = self.mcp.parse_todo_report(diagnosis_report)
        if len(todos) > 5:
            todos = self._consolidate_todos(todos, spec, code, cancel_token)
        if not todos:
            first_error = errors[0] if errors else {}
            todos = [
                {
                    "id": "todo-1",
                    "order": 1,
                    "status": "pending",
                    "location": {
                        "raw": f"line {first_error.get('line', '?')}",
                        "line_start": first_error.get("line") if isinstance(first_error.get("line"), int) else None,
                        "line_end": first_error.get("line") if isinstance(first_error.get("line"), int) else None,
                    },
                    "bug": first_error.get("message", str(first_error)),
                    "fix": "Correct the failing logic while preserving the rest of the module.",
                    "review": "Verify the reported error no longer occurs.",
                    "source_line": 1,
                }
            ]
        self.mcp.write_todos(trial_id, todos)
        return diagnosis_report

    def _apply_todos(
        self,
        trial_id: str,
        spec: str,
        rtl_filename: str,
        code: str,
        rtl_path: str,
        testbench_context: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        on_token: Any | None,
        cancel_token: CancellationToken | None,
    ) -> tuple[str, str]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        todos = [t for t in self.mcp.read_todos(trial_id) if t.get("status") == "pending"]
        if not todos:
            return code, rtl_path

        for todo in todos:
            self.mcp.update_todo_status(trial_id, todo["id"], "active")

        combined_report = "\n".join(self._build_todo_report(t) for t in todos)
        source_context: list[dict[str, Any]] = []
        for todo in todos:
            source_context.extend(self._build_todo_source_context(rtl_path, todo))

        errors = [
            {"line": t.get("location", {}).get("line_start", "?"), "message": t.get("bug", "")}
            for t in todos
        ]

        code_before_fix = code
        code, _ = self.generator.fix_errors(
            original_spec=spec,
            current_code=code,
            errors=errors,
            diagnosis_report=combined_report,
            source_context=source_context,
            testbench_context=testbench_context,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            on_token=on_token,
            cancel_token=cancel_token,
        )

        if not re.search(r"\bmodule\b", code, re.IGNORECASE):
            print("[fix guard] fix destroyed module structure — reverting all TODOs")
            code = code_before_fix
            self.mcp.batch_update_todos(
                trial_id,
                [{"id": t["id"], "status": "failed", "review_notes": "Fix destroyed module structure; reverted."} for t in todos],
            )
            return code, rtl_path

        rtl_path = str(self.mcp.write_trial_source(trial_id, rtl_filename, code))

        max_syntax_retries = 2
        for syntax_retry in range(max_syntax_retries + 1):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            syntax_result = self.syntax.execute(
                {rtl_filename: code},
                config={"cancel_token": cancel_token},
            )
            if syntax_result.pass_:
                break
            if syntax_retry >= max_syntax_retries:
                break
            syntax_errors_text = "\n".join(
                f"  Line {e.get('line', '?')}: {e.get('message', str(e))}"
                for e in syntax_result.errors
            )
            code, _ = self.generator.fix_errors(
                original_spec=spec,
                current_code=code,
                errors=syntax_result.errors,
                diagnosis_report=f"Syntax errors introduced by the fix:\n{syntax_errors_text}",
                source_context=source_context,
                testbench_context=testbench_context,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                on_token=on_token,
                cancel_token=cancel_token,
            )
            rtl_path = str(self.mcp.write_trial_source(trial_id, rtl_filename, code))

        self.mcp.batch_update_todos(
            trial_id,
            [{"id": t["id"], "status": "done"} for t in todos],
        )

        return code, rtl_path

    def _batch_apply_review_updates(
        self,
        trial_id: str,
        review_result: dict[str, Any],
        current_todo_id: str,
    ) -> None:
        updates = review_result.get("updates", [])
        new_todos = review_result.get("new_todos", [])

        review_statuses: dict[str, str] = {}
        for u in updates:
            uid = u.get("id", "")
            status = u.get("status", "pending")
            review_statuses[uid] = status

        if current_todo_id not in review_statuses:
            review_statuses[current_todo_id] = "done"

        enriched_updates = []
        for uid, status in review_statuses.items():
            reason = ""
            for u in updates:
                if u.get("id") == uid:
                    reason = u.get("reason", "")
                    break
            enriched_updates.append({
                "id": uid,
                "status": status,
                "reason": reason,
                "review_notes": reason,
            })

        self.mcp.batch_update_todos(trial_id, enriched_updates, new_todos if new_todos else None)

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
        baseline_mode: bool = True,
        cancel_token: CancellationToken | None = None,
        on_trial_start: Any | None = None,
        on_sim_output: Any | None = None,
    ) -> tuple[str, TrialScore]:
        del enable_rtl_reuse, baseline_mode

        if on_step_change:
            on_step_change("plan", "inactive")
            on_step_change("gen", "inactive")
            on_step_change("todo", "inactive")
            on_step_change("fix", "inactive")
            on_step_change("syntax", "inactive")
            on_step_change("sim", "inactive")
            on_step_change("ppa", "inactive")
            on_step_change("db", "inactive")

        trial_id = uuid.uuid4().hex[:12]
        if on_trial_start:
            on_trial_start(trial_id)

        # Restrict the LLM tools to the output directory of this trial and the workbench directory
        from src.llm import set_allowed_paths
        out_dir = self.mcp.get_trial_output_dir(trial_id)
        wb_dir = self.mcp.db_path.parent / "workbench" / trial_id
        set_allowed_paths([str(out_dir), str(wb_dir)])

        try:
            start = time.monotonic()
            constraints = self.mcp.get_constraints(problem_id.split("_")[0] if "_" in problem_id else "default")

            augmented_spec = self._build_prompt(spec, context_files)
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            if on_token:
                on_token("")
            if on_step_change:
                on_step_change("gen", "active")
            code, _ = self.generator.generate_rtl(
                spec=augmented_spec,
                system_message=system_message,
                params=params,
                constraints=constraints,
                testbench_files=testbench_files,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                on_token=on_token,
                cancel_token=cancel_token,
            )
            if on_step_change:
                on_step_change("gen", "success")

            rtl_filename = "rtl.sv"
            if problem_id.startswith("cvdp_"):
                try:
                    from src.cvdp.loader import CVDPDataset

                    problem = CVDPDataset().get_by_id(problem_id)
                    if problem and problem.output_files:
                        rtl_filename = problem.output_files[0]
                except Exception:
                    pass

            score = TrialScore(problem_id=problem_id, trial_index=0, trial_id=trial_id)
            latest_errors: list[dict[str, Any]] = []
            latest_diagnosis_report: str | None = None
            actual_retries = 0
            rtl_path = str(self.mcp.write_trial_source(trial_id, rtl_filename, code))
            testbench_paths: list[str] = []
            if testbench_files:
                for tb_name, tb_content in testbench_files.items():
                    testbench_paths.append(str(self.mcp.write_trial_source(trial_id, tb_name, tb_content)))

            for retry in range(max_retries + 1):
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()
                actual_retries = retry

                if on_step_change:
                    on_step_change("syntax", "active")
                syntax_result = self.syntax.execute(
                    {rtl_filename: code},
                    config={"cancel_token": cancel_token},
                )
                if verbose:
                    print(f"  [{retry}] Syntax: {'PASS' if syntax_result.pass_ else 'FAIL'}")

                if not syntax_result.pass_:
                    latest_errors = syntax_result.errors
                    if on_step_change:
                        on_step_change("syntax", "fail")

                    if getattr(syntax_result, "is_infra_failure", False):
                        if verbose:
                            print("  [CRITICAL] Syntax check infrastructure failure detected! Aborting trial retries early.")
                            for err in latest_errors:
                                print(f"    Reason: {err.get('message', str(err))}")
                        score.syntax_pass = False
                        score.errors = latest_errors
                        break

                    if retry >= max_retries:
                        score.syntax_pass = False
                        score.errors = latest_errors
                        break

                    if on_step_change:
                        on_step_change("todo", "active")
                    latest_diagnosis_report = self._diagnose_and_write_todos(
                        trial_id=trial_id,
                        rtl_filename=rtl_filename,
                        code=code,
                        spec=spec,
                        errors=latest_errors,
                        rtl_path=rtl_path,
                        cancel_token=cancel_token,
                    )
                    if on_step_change:
                        on_step_change("todo", "success")
                        on_step_change("fix", "active")
                    if on_token:
                        on_token("")
                    code, rtl_path = self._apply_todos(
                        trial_id=trial_id,
                        spec=spec,
                        rtl_filename=rtl_filename,
                        code=code,
                        rtl_path=rtl_path,
                        testbench_context=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                        on_token=on_token,
                        cancel_token=cancel_token,
                    )
                    if on_step_change:
                        on_step_change("fix", "success")
                    continue

                score.syntax_pass = True
                if on_step_change:
                    on_step_change("syntax", "success")

                if not testbench_files:
                    score.simulation_pass = False
                    score.errors = [{"message": "No testbench available"}]
                    break

                if on_step_change:
                    on_step_change("sim", "active")

                def _default_sim_line(line: str) -> None:
                    stripped = line.strip()
                    if stripped:
                        print(f"  [SIM] {stripped}")

                sim_output_cb = on_sim_output if on_sim_output is not None else (_default_sim_line if verbose else None)
                sim_result = self.simulation.execute(
                    {rtl_filename: code},
                    testbench_files,
                    config={
                        "cancel_token": cancel_token,
                        "on_sim_output": sim_output_cb,
                    },
                )
                if verbose:
                    print(f"  [{retry}] Simulation: {'PASS' if sim_result.pass_ else 'FAIL'}")

                if sim_result.pass_:
                    score.simulation_pass = True
                    if on_step_change:
                        on_step_change("sim", "success")
                    break

                latest_errors = sim_result.errors
                if on_step_change:
                    on_step_change("sim", "fail")

                if getattr(sim_result, "is_infra_failure", False):
                    if verbose:
                        print("  [CRITICAL] Simulation infrastructure failure detected! Aborting trial retries early.")
                        for err in latest_errors:
                            print(f"    Reason: {err.get('message', str(err))}")
                    score.simulation_pass = False
                    score.errors = latest_errors
                    break

                if retry >= max_retries:
                    score.simulation_pass = False
                    score.errors = latest_errors
                    break

                testbench_context = self.mcp.get_testbench_context(testbench_paths) if testbench_paths else []
                if on_step_change:
                    on_step_change("todo", "active")
                latest_diagnosis_report = self._diagnose_and_write_todos(
                    trial_id=trial_id,
                    rtl_filename=rtl_filename,
                    code=code,
                    spec=spec,
                    errors=latest_errors,
                    rtl_path=rtl_path,
                    testbench_files=testbench_files,
                    testbench_context=testbench_context,
                    cancel_token=cancel_token,
                )
                if on_step_change:
                    on_step_change("todo", "success")
                    on_step_change("fix", "active")
                if on_token:
                    on_token("")
                code, rtl_path = self._apply_todos(
                    trial_id=trial_id,
                    spec=spec,
                    rtl_filename=rtl_filename,
                    code=code,
                    rtl_path=rtl_path,
                    testbench_context=testbench_context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    on_token=on_token,
                    cancel_token=cancel_token,
                )
                if on_step_change:
                    on_step_change("fix", "success")

            score.duration_ms = (time.monotonic() - start) * 1000
            score.retry_count = actual_retries

            top_module = "top"
            if rtl_filename.endswith(".sv") or rtl_filename.endswith(".v"):
                top_module = rtl_filename.rsplit(".", 1)[0]

            ppa_score = {}
            if on_step_change:
                on_step_change("ppa", "success")

            vivado_metrics = None
            if self.vivado_agent and score.pass_:
                if on_step_change:
                    on_step_change("ppa", "active")
                vivado_result = self.vivado_agent.execute(
                    {rtl_filename: code},
                    testbench_files=testbench_files,
                    config={
                        "top_module": top_module,
                        "cancel_token": cancel_token,
                    },
                )
                if vivado_result.pass_:
                    vivado_metrics = vivado_result.metrics
                    score.vivado_metrics = vivado_metrics
                    if on_step_change:
                        on_step_change("ppa", "success")
                else:
                    if on_step_change:
                        on_step_change("ppa", "fail")

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            if on_step_change:
                on_step_change("db", "active")
            self.mcp.write_trial(
                TrialRecord(
                    trial_id=trial_id,
                    problem_id=problem_id,
                    params=params or {},
                    spec_prompt=spec,
                    generated_code=code,
                    syntax_pass=score.syntax_pass,
                    simulation_pass=score.simulation_pass,
                    simulation_failures=[str(err) for err in score.errors] if score.errors else [],
                    ppa_score=ppa_score,
                    vivado_metrics=vivado_metrics,
                    pass_=score.pass_,
                    retry_count=actual_retries,
                    duration_ms=score.duration_ms,
                    diagnosis_report=latest_diagnosis_report,
                )
            )
            if on_step_change:
                on_step_change("db", "success")

            return code, score
        finally:
            set_allowed_paths(None)

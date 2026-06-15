import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class TrialRecord(BaseModel):
    trial_id: str
    problem_id: str
    params: dict[str, Any]
    spec_prompt: str
    generated_code: str
    syntax_pass: bool
    simulation_pass: bool
    simulation_failures: list[str] = []
    ppa_score: dict[str, Any] | None = None
    vivado_metrics: dict[str, Any] | None = None
    pass_: bool = False
    retry_count: int = 0
    duration_ms: float = 0.0
    diagnosis_report: str | None = None


class MCPServer:
    def __init__(
        self,
        constraints_path: str | None = None,
        interfaces_path: str | None = None,
        db_path: str | None = None,
    ):
        self.constraints_path = Path(constraints_path or "config/constraints.json")
        self.interfaces_path = Path(interfaces_path or "config/interfaces.json")
        self.db_path = Path(db_path or "data/trials/trials.db")

        self._constraints: dict[str, Any] = {}
        self._interfaces: dict[str, Any] = {}

    def get_trial_output_dir(self, trial_id: str) -> Path:
        out_dir = self.db_path.parent.parent / "outputs" / trial_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def get_trial_todo_path(self, trial_id: str) -> Path:
        return self.get_trial_output_dir(trial_id) / "todos.json"

    def write_trial_source(self, trial_id: str, filename: str, content: str) -> Path:
        out_dir = self.get_trial_output_dir(trial_id)
        file_path = out_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def read_file_lines(
        self,
        file_path: str | Path,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        lines = path.read_text().splitlines()
        start = max((start_line or 1), 1)
        end = min(end_line or len(lines), len(lines))
        excerpt = [
            {"line": line_no, "text": lines[line_no - 1]}
            for line_no in range(start, end + 1)
        ]
        return {
            "path": str(path),
            "start_line": start,
            "end_line": end,
            "lines": excerpt,
        }

    def get_error_source_context(
        self,
        file_path: str | Path,
        errors: list[dict[str, Any]],
        context_radius: int = 2,
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        for error in errors:
            line_no = error.get("line")
            if not isinstance(line_no, int) or line_no < 1:
                continue
            excerpt = self.read_file_lines(
                file_path,
                start_line=max(1, line_no - context_radius),
                end_line=line_no + context_radius,
            )
            contexts.append(
                {
                    "path": excerpt["path"],
                    "error_line": line_no,
                    "message": error.get("message", str(error)),
                    "excerpt": excerpt["lines"],
                }
            )
        return contexts

    def get_testbench_context(
        self,
        file_paths: list[str | Path],
        *,
        context_radius: int = 2,
        max_matches_per_file: int = 4,
    ) -> list[dict[str, Any]]:
        keywords = ("assert", "expect", "expected", "posedge", "negedge", "reset", "clk", "dut")
        contexts: list[dict[str, Any]] = []

        for file_path in file_paths:
            path = Path(file_path)
            lines = path.read_text().splitlines()
            matches: list[int] = []

            for idx, line in enumerate(lines, start=1):
                lower = line.lower()
                if any(keyword in lower for keyword in keywords):
                    matches.append(idx)
                if len(matches) >= max_matches_per_file:
                    break

            if not matches and lines:
                matches = [1]

            for line_no in matches:
                excerpt = self.read_file_lines(
                    path,
                    start_line=max(1, line_no - context_radius),
                    end_line=min(len(lines), line_no + context_radius),
                )
                contexts.append(
                    {
                        "path": str(path),
                        "focus_line": line_no,
                        "excerpt": excerpt["lines"],
                    }
                )
        return contexts

    def parse_todo_report(self, diagnosis_report: str) -> list[dict[str, Any]]:
        todos: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for idx, raw_line in enumerate(diagnosis_report.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            todo_match = re.match(r"^TODO\s+(\d+)\s*$", line, re.IGNORECASE)
            if todo_match:
                if current is not None:
                    todos.append(current)
                todo_num = todo_match.group(1)
                current = {
                    "id": f"todo-{todo_num}",
                    "order": len(todos) + 1,
                    "status": "pending",
                    "location": {"raw": "", "line_start": None, "line_end": None},
                    "snippet": "",
                    "bug": "",
                    "fix": "",
                    "review": "",
                    "source_line": idx,
                }
                continue

            if current is None or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "LOCATION":
                current["location"]["raw"] = value
                line_numbers = [int(num) for num in re.findall(r"\d+", value)]
                current["location"]["line_start"] = line_numbers[0] if line_numbers else None
                current["location"]["line_end"] = line_numbers[-1] if len(line_numbers) > 1 else current["location"]["line_start"]
            elif key == "SNIPPET":
                snippet_match = re.search(r"`([^`]+)`", value)
                current["snippet"] = snippet_match.group(1).strip() if snippet_match else value
            elif key == "BUG":
                current["bug"] = value
            elif key == "FIX":
                current["fix"] = value
            elif key == "REVIEW":
                current["review"] = value

        if current is not None:
            todos.append(current)

        return todos

    def write_todos(self, trial_id: str, todos: list[dict[str, Any]]) -> Path:
        todo_path = self.get_trial_todo_path(trial_id)
        todo_path.write_text(json.dumps(todos, indent=2))
        if todos:
            print(f"[update TODO] wrote {len(todos)} item(s) for trial {trial_id}")
        return todo_path

    def read_todos(self, trial_id: str) -> list[dict[str, Any]]:
        todo_path = self.get_trial_todo_path(trial_id)
        if not todo_path.exists():
            return []
        try:
            return json.loads(todo_path.read_text())
        except Exception:
            return []

    def get_next_pending_todo(self, trial_id: str) -> dict[str, Any] | None:
        for todo in self.read_todos(trial_id):
            if todo.get("status") == "pending":
                return todo
        return None

    def update_todo_status(
        self,
        trial_id: str,
        todo_id: str,
        status: str,
        *,
        review_notes: str | None = None,
    ) -> None:
        todos = self.read_todos(trial_id)
        for todo in todos:
            if todo.get("id") != todo_id:
                continue
            todo["status"] = status
            if review_notes is not None:
                todo["review_notes"] = review_notes
            summary = todo.get("bug", "").strip() or todo_id
            print(f"[update TODO] {todo_id} -> {status} ({summary})")
            break
        self.write_todos(trial_id, todos)

    def batch_update_todos(
        self,
        trial_id: str,
        updates: list[dict[str, Any]],
        new_todos: list[dict[str, Any]] | None = None,
    ) -> None:
        todos = self.read_todos(trial_id)
        update_map = {u["id"]: u for u in updates}

        for todo in todos:
            if todo.get("id") in update_map:
                up = update_map[todo["id"]]
                todo["status"] = up.get("status", todo["status"])
                if "review_notes" in up:
                    todo["review_notes"] = up["review_notes"]
                if "reason" in up:
                    todo["reason"] = up["reason"]
                summary = todo.get("bug", "").strip() or todo["id"]
                print(f"[batch TODO] {todo['id']} -> {todo['status']} ({summary})")

        if new_todos:
            next_order = max((t.get("order", 0) for t in todos), default=0)
            next_id_num = max(
                (int(t["id"].split("-")[-1]) for t in todos if "-" in t.get("id", "")),
                default=0,
            )
            for new_todo in new_todos:
                next_id_num += 1
                next_order += 1
                todo_item = {
                    "id": f"todo-{next_id_num}",
                    "order": next_order,
                    "status": "pending",
                    "location": new_todo.get("location", {"raw": "", "line_start": None, "line_end": None}),
                    "snippet": new_todo.get("snippet", ""),
                    "bug": new_todo.get("bug", ""),
                    "fix": new_todo.get("fix", ""),
                    "review": new_todo.get("review", ""),
                    "source_line": new_todo.get("source_line", 1),
                }
                todos.append(todo_item)
            print(f"[batch TODO] added {len(new_todos)} new todo(s) for trial {trial_id}")

        self.write_todos(trial_id, todos)

    def _set_todo_attempts(self, trial_id: str, todo_id: str, attempts: int) -> None:
        todos = self.read_todos(trial_id)
        for todo in todos:
            if todo.get("id") == todo_id:
                todo["attempts"] = attempts
                break
        self.write_todos(trial_id, todos)

    def _load_constraints(self) -> dict[str, Any]:
        if not self._constraints and self.constraints_path.exists():
            self._constraints = json.loads(self.constraints_path.read_text())
        return self._constraints

    def _load_interfaces(self) -> dict[str, Any]:
        if not self._interfaces and self.interfaces_path.exists():
            self._interfaces = json.loads(self.interfaces_path.read_text())
        return self._interfaces

    def get_constraints(self, module_type: str) -> dict[str, Any]:
        constraints = self._load_constraints()
        result: dict[str, Any] = dict(constraints.get("default", {}))
        if module_type in constraints:
            result.update(constraints[module_type])
        return result

    def get_interface(self, protocol: str) -> dict[str, Any] | None:
        interfaces = self._load_interfaces()
        return interfaces.get(protocol)

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_history(self, problem_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT trial_id, problem_id, params, spec_prompt, generated_code,
                           syntax_pass, simulation_pass, simulation_failures, ppa_score,
                           pass, retry_count, duration_ms, diagnosis_report, timestamp
                    FROM trials
                    WHERE problem_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (problem_id, limit),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "trial_id": row["trial_id"],
                        "problem_id": row["problem_id"],
                        "params": json.loads(row["params"]),
                        "spec_prompt": row["spec_prompt"],
                        "generated_code": row["generated_code"],
                        "syntax_pass": bool(row["syntax_pass"]),
                        "simulation_pass": bool(row["simulation_pass"]),
                        "simulation_failures": json.loads(row["simulation_failures"]),
                        "ppa_score": json.loads(row["ppa_score"]) if row["ppa_score"] else None,
                        "pass_": bool(row["pass"]),
                        "retry_count": row["retry_count"],
                        "duration_ms": row["duration_ms"],
                        "diagnosis_report": row["diagnosis_report"],
                        "timestamp": row["timestamp"],
                    })
                return results
        except sqlite3.OperationalError:
            # Table might not exist yet
            return []

    def get_successful_trials(self, problem_id: str) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT trial_id, problem_id, params, spec_prompt, generated_code,
                           syntax_pass, simulation_pass, simulation_failures, ppa_score,
                           pass, retry_count, duration_ms, diagnosis_report, timestamp
                    FROM trials
                    WHERE problem_id = ? AND pass = 1
                    ORDER BY timestamp DESC
                    """,
                    (problem_id,),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "trial_id": row["trial_id"],
                        "problem_id": row["problem_id"],
                        "params": json.loads(row["params"]),
                        "spec_prompt": row["spec_prompt"],
                        "generated_code": row["generated_code"],
                        "syntax_pass": bool(row["syntax_pass"]),
                        "simulation_pass": bool(row["simulation_pass"]),
                        "simulation_failures": json.loads(row["simulation_failures"]),
                        "ppa_score": json.loads(row["ppa_score"]) if row["ppa_score"] else None,
                        "pass_": bool(row["pass"]),
                        "retry_count": row["retry_count"],
                        "duration_ms": row["duration_ms"],
                        "diagnosis_report": row["diagnosis_report"],
                        "timestamp": row["timestamp"],
                    })
                return results
        except sqlite3.OperationalError:
            # Table might not exist yet
            return []

    def get_successful_trials_by_pattern(self, pattern: str) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT trial_id, problem_id, params, spec_prompt, generated_code,
                           syntax_pass, simulation_pass, simulation_failures, ppa_score,
                           pass, retry_count, duration_ms, diagnosis_report, timestamp
                    FROM trials
                    WHERE problem_id LIKE ? AND pass = 1
                    ORDER BY timestamp DESC
                    """,
                    (f"%{pattern}%",),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "trial_id": row["trial_id"],
                        "problem_id": row["problem_id"],
                        "params": json.loads(row["params"]),
                        "spec_prompt": row["spec_prompt"],
                        "generated_code": row["generated_code"],
                        "syntax_pass": bool(row["syntax_pass"]),
                        "simulation_pass": bool(row["simulation_pass"]),
                        "simulation_failures": json.loads(row["simulation_failures"]),
                        "ppa_score": json.loads(row["ppa_score"]) if row["ppa_score"] else None,
                        "pass_": bool(row["pass"]),
                        "retry_count": row["retry_count"],
                        "duration_ms": row["duration_ms"],
                        "diagnosis_report": row["diagnosis_report"],
                        "timestamp": row["timestamp"],
                    })
                return results
        except sqlite3.OperationalError:
            # Table might not exist yet
            return []

    def write_trial(self, trial: TrialRecord) -> str:
        self.init_db()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trials (
                    trial_id, problem_id, params, spec_prompt, generated_code,
                    syntax_pass, simulation_pass, simulation_failures, ppa_score,
                    pass, retry_count, duration_ms, diagnosis_report
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id,
                    trial.problem_id,
                    json.dumps(trial.params),
                    trial.spec_prompt,
                    trial.generated_code,
                    1 if trial.syntax_pass else 0,
                    1 if trial.simulation_pass else 0,
                    json.dumps(trial.simulation_failures),
                    json.dumps(trial.ppa_score) if trial.ppa_score is not None else None,
                    1 if trial.pass_ else 0,
                    trial.retry_count,
                    trial.duration_ms,
                    trial.diagnosis_report,
                ),
            )
            conn.commit()

        self._write_local_markdown_report(trial)
        return trial.trial_id

    def _write_local_markdown_report(self, trial: TrialRecord) -> None:
        report_dir = self.db_path.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"trial_{trial.trial_id}.md"

        status = "PASS" if trial.pass_ else "FAIL"
        syntax_status = "PASS" if trial.syntax_pass else "FAIL"
        sim_status = "PASS" if trial.simulation_pass else "FAIL"

        content = [
            f"# VeriGen Trial Report: {trial.trial_id}",
            f"- **Problem ID**: {trial.problem_id}",
            f"- **Overall Status**: {status}",
            f"- **Retry Count**: {trial.retry_count}",
            f"- **Duration**: {trial.duration_ms:.0f}ms",
            f"- **Syntax Check**: {syntax_status}",
            f"- **Simulation Check**: {sim_status}",
            "",
            "## 1. Design Specification",
            "```text",
            trial.spec_prompt,
            "```",
            "",
            "## 2. Generated Verilog Code",
            "```verilog",
            trial.generated_code,
            "```",
            ""
        ]

        if trial.diagnosis_report:
            content.extend([
                "## 3. Sub-Agent Diagnosis & TODO Tasks",
                trial.diagnosis_report,
                ""
            ])

        if trial.simulation_failures:
            content.extend([
                "## 4. Simulation Failures Trackback",
                "```text",
                "\n".join(trial.simulation_failures),
                "```",
                ""
            ])

        if trial.ppa_score:
            content.extend([
                "## 5. PPA Performance Metrics",
                "```json",
                json.dumps(trial.ppa_score, indent=2),
                "```",
                ""
            ])

        report_file.write_text("\n".join(content))

    def init_workbench(self, trial_id: str, problem_id: str, spec: str, plan_todos: list[str], architecture_spec: str | None = None) -> None:
        wb_dir = self.db_path.parent / "workbench"
        wb_dir.mkdir(parents=True, exist_ok=True)
        wb_file = wb_dir / "todo.json"

        todos = []
        for i, desc in enumerate(plan_todos, start=1):
            todos.append({
                "id": i,
                "description": desc,
                "status": "pending"
            })

        data = {
            "trial_id": trial_id,
            "problem_id": problem_id,
            "spec": spec,
            "status": "planning",
            "todos": todos,
            "architecture_spec": architecture_spec,
            "history": []
        }
        wb_file.write_text(json.dumps(data, indent=2))

    def read_workbench(self) -> dict[str, Any]:
        wb_file = self.db_path.parent / "workbench" / "todo.json"
        if not wb_file.exists():
            return {"todos": [], "status": "none"}
        try:
            return json.loads(wb_file.read_text())
        except Exception:
            return {"todos": [], "status": "none"}

    def update_workbench_status(self, status: str, mark_all_completed: bool = False) -> None:
        wb_file = self.db_path.parent / "workbench" / "todo.json"
        if not wb_file.exists():
            return
        try:
            data = json.loads(wb_file.read_text())
            data["status"] = status
            if mark_all_completed:
                for t in data["todos"]:
                    if t["status"] == "pending":
                        t["status"] = "completed"
            wb_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def add_debug_todos(self, debug_todos: list[str]) -> None:
        wb_file = self.db_path.parent / "workbench" / "todo.json"
        if not wb_file.exists():
            return
        try:
            data = json.loads(wb_file.read_text())
            max_id = max([t["id"] for t in data["todos"]]) if data["todos"] else 0
            for i, desc in enumerate(debug_todos, start=1):
                data["todos"].append({
                    "id": max_id + i,
                    "description": desc,
                    "status": "pending"
                })
            data["status"] = "debugging"
            wb_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    problem_id TEXT,
                    params TEXT,
                    spec_prompt TEXT,
                    generated_code TEXT,
                    syntax_pass INTEGER,
                    simulation_pass INTEGER,
                    simulation_failures TEXT,
                    ppa_score TEXT,
                    pass INTEGER,
                    retry_count INTEGER,
                    duration_ms REAL,
                    diagnosis_report TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

        # Defensively upgrade database schema if diagnosis_report is missing
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(trials)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "diagnosis_report" not in columns:
                try:
                    cursor.execute("ALTER TABLE trials ADD COLUMN diagnosis_report TEXT")
                    conn.commit()
                except Exception:
                    pass


    def clear_all_trials(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trials")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM trials")
            conn.commit()
        print(f"[DB] cleared {count} trial record(s)")
        return count


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        server = MCPServer()
        server.init_db()
        print(f"Initialized database at {server.db_path}")

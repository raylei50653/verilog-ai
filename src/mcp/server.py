import json
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        server = MCPServer()
        server.init_db()
        print(f"Initialized database at {server.db_path}")

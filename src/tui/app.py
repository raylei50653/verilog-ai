import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Label,
    Input,
    Button,
    RichLog,
    DataTable,
    TabbedContent,
    TabPane,
    Checkbox,
    TextArea,
    Select,
    ContentSwitcher,
)
from textual import work
from textual.worker import Worker, WorkerState

from dotenv import load_dotenv

load_dotenv()
from rich.markup import escape

from src.cancellation import CancellationToken, PipelineCancelled
from src.llm import create_backend
from src.pipeline import TrialRunner
from src.cvdp.loader import CVDPDataset
from src.mcp.server import MCPServer


class StdoutRedirector:
    def __init__(self, write_fn):
        self.write_fn = write_fn
        self._stdout = sys.stdout

    def write(self, s):
        self.write_fn(s)
        self._stdout.write(s)

    def flush(self):
        self._stdout.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout


try:
    _cvdp = CVDPDataset()
    _sorted_problems = sorted(list(_cvdp), key=lambda p: (
        {"easy": 0, "medium": 1, "hard": 2}.get(p.difficulty, 3),
        p.id
    ))
    CVDP_PROBLEM_IDS = [p.id for p in _sorted_problems]
    CVDP_PROBLEM_DIFFS = {p.id: p.difficulty for p in _sorted_problems}
except Exception:
    CVDP_PROBLEM_IDS = ["cvdp_copilot_16qam_mapper_0001"]
    CVDP_PROBLEM_DIFFS = {"cvdp_copilot_16qam_mapper_0001": "unknown"}


class VeriGenTUI(App):
    CSS = """
    Screen {
        background: #1e1e24;
    }
    Header {
        background: #282a36;
        color: #f8f8f2;
    }
    Footer {
        background: #282a36;
        color: #f8f8f2;
    }
    TabbedContent {
        margin: 0;
        padding: 0;
    }
    TabPane {
        padding: 0;
    }
    .left-panel {
        width: 35%;
        height: 100%;
        border-right: solid #44475a;
        padding: 1;
    }
    .right-panel {
        width: 65%;
        height: 100%;
        padding: 1;
    }
    .field-label {
        text-style: bold;
        margin-bottom: 0;
        margin-top: 1;
        color: #8be9fd;
    }
    .btn-group {
        margin-top: 2;
        height: 3;
        width: 100%;
    }
    #btn-run, #btn-vivado {
        width: 60%;
        background: #50fa7b;
        color: #282a36;
    }
    #btn-run:hover, #btn-vivado:hover {
        background: #85fa9b;
    }
    #btn-stop, #btn-vivado-stop {
        width: 40%;
        background: #ff5555;
        color: #f8f8f2;
    }
    #btn-stop:hover, #btn-vivado-stop:hover {
        background: #ff6e6e;
    }
    #btn-copy-logs, #btn-copy-sim-log {
        width: 50%;
        margin-top: 1;
    }
    #btn-to-vivado {
        width: 50%;
        margin-top: 1;
        background: #6272a4;
        color: #f8f8f2;
    }
    #btn-to-vivado:hover {
        background: #7a8ac0;
    }
    #btn-vivado-mkproj {
        width: 100%;
        margin-top: 1;
        background: #44475a;
        color: #f8f8f2;
    }
    #btn-vivado-mkproj:hover {
        background: #5a5f78;
    }
    #btn-vivado-gui {
        width: 100%;
        margin-top: 1;
        background: #6272a4;
        color: #f8f8f2;
    }
    #btn-vivado-gui:hover {
        background: #7a8ac0;
    }
    #docker-status {
        margin-left: 2;
        text-style: bold;
    }
    #btn-docker-retry {
        width: 3;
        min-width: 3;
        height: 1;
        border: none;
        background: transparent;
        color: #6272a4;
        margin-left: 0;
        padding: 0;
    }
    #btn-docker-retry:hover {
        color: #f8f8f2;
    }
    .docker-checking { color: #f1fa8c; }
    .docker-starting { color: #ffb86c; }
    .docker-running  { color: #50fa7b; }
    .docker-stopped  { color: #ff5555; }
    .log-view {
        height: 100%;
        background: #282a36;
    }
    .code-view {
        height: 100%;
        background: #282a36;
    }
    .status-view {
        height: 4;
        background: #282a36;
        border: solid #44475a;
        padding: 1;
        color: #f8f8f2;
    }
    #panel-toggle {
        height: 3;
        margin-top: 1;
    }
    .panel-tab {
        width: 25%;
        background: #44475a;
        color: #6272a4;
        border: none;
        text-style: bold;
    }
    .panel-tab:hover {
        background: #555770;
        color: #f8f8f2;
    }
    .panel-tab-active {
        background: #6272a4;
        color: #f8f8f2;
    }
    #panel-switcher {
        height: 1fr;
        border: solid #44475a;
    }
    #panel-logs, #panel-sim, #panel-code, #panel-todo {
        height: 100%;
    }
    #todo-log {
        height: 100%;
        background: #282a36;
    }
    .opt-trials-view {
        height: 70%;
        border: solid #44475a;
        margin-top: 1;
        background: #282a36;
    }
    #runner-status-bar {
        height: 5;
        content-align: center middle;
        margin-top: 1;
        margin-bottom: 1;
        background: #282a36;
        border: solid #44475a;
    }
    #btn-kill-sim {
        width: auto;
        min-width: 14;
        height: 3;
        background: #44475a;
        color: #6272a4;
        border: none;
        margin-left: 2;
    }
    #btn-kill-sim:enabled {
        background: #ff5555;
        color: #f8f8f2;
    }
    #btn-kill-sim:enabled:hover {
        background: #ff6e6e;
    }
    .status-step {
        padding: 0 1;
        text-style: bold;
        text-align: center;
        border: round #44475a;
    }
    .status-arrow {
        color: #6272a4;
        padding: 0 1;
    }
    #status-header {
        height: 1;
        margin-bottom: 1;
    }
    #todo-summary {
        width: 40%;
        content-align: right middle;
        color: #f1fa8c;
        text-style: bold;
    }
    .inactive {
        background: #44475a;
        color: #6272a4;
    }
    .active {
        background: #ffb86c;
        color: #282a36;
    }
    .success {
        background: #50fa7b;
        color: #282a36;
    }
    .fail {
        background: #ff5555;
        color: #f8f8f2;
    }
    """

    TITLE = "VeriGen - RTL Generator & Vivado"
    BINDINGS = [("q", "quit", "Quit App")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Runner", id="tab-runner"):
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Label("Spec Prompt / Description:", classes="field-label")
                        yield Input(value="Implement a simple 2-to-1 multiplexer", id="run-spec")
                        yield Label("CVDP Problem ID (optional):", classes="field-label")
                        run_options = [("Custom Specification", "custom")] + [
                            (f"[{CVDP_PROBLEM_DIFFS.get(pid, 'unknown').upper()}] {pid}", pid)
                            for pid in CVDP_PROBLEM_IDS
                        ]
                        yield Select(run_options, value="custom", id="run-problem-id")
                        yield Label("Parameters (JSON):", classes="field-label")
                        yield Input(value="{}", id="run-params")
                        yield Label("Model Name Override:", classes="field-label")
                        yield Input(placeholder="e.g. deepseek-coder-v2", id="run-model")
                        yield Label("Max Retries:", classes="field-label")
                        yield Input(value="3", id="run-max-retries")
                        yield Checkbox("Show AI Thinking", value=False, id="run-show-thinking")
                        with Horizontal(classes="btn-group"):
                            yield Button("Generate RTL", variant="success", id="btn-run")
                            yield Button("Stop", variant="error", id="btn-stop", disabled=True)
                        with Horizontal():
                            yield Button("Copy Logs", variant="primary", id="btn-copy-logs")
                            yield Button("Copy SIM Log", id="btn-copy-sim-log")
                            yield Button("Vivado →", id="btn-to-vivado", disabled=True)

                    with Vertical(classes="right-panel"):
                        with Horizontal(id="status-header"):
                            yield Label("Pipeline Progress Status:")
                            yield Label("No TODOs", id="todo-summary")
                            yield Label("◌ Docker: Checking...", id="docker-status", classes="docker-checking")
                            yield Button("⟳", id="btn-docker-retry")
                        with Horizontal(id="runner-status-bar"):
                            yield Label("GEN", id="step-gen", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("SYNTAX", id="step-syntax", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("SIM", id="step-sim", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("TODO", id="step-todo", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("FIX", id="step-fix", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("DB", id="step-db", classes="status-step inactive")
                            yield Button("⊗ Kill SIM", id="btn-kill-sim", disabled=True)
                        with Horizontal(id="panel-toggle"):
                            yield Button("Pipeline Logs", id="btn-panel-logs", classes="panel-tab panel-tab-active")
                            yield Button("SIM Log", id="btn-panel-sim", classes="panel-tab")
                            yield Button("Generated Code", id="btn-panel-code", classes="panel-tab")
                            yield Button("TODO List", id="btn-panel-todo", classes="panel-tab")
                        with ContentSwitcher(initial="panel-logs", id="panel-switcher"):
                            with Vertical(id="panel-logs"):
                                yield RichLog(highlight=True, markup=True, classes="log-view", id="run-log")
                            with Vertical(id="panel-sim"):
                                yield RichLog(highlight=True, markup=True, classes="log-view", id="sim-log")
                            with Vertical(id="panel-code"):
                                yield TextArea(read_only=True, language="verilog", classes="code-view", id="run-code")
                            with Vertical(id="panel-todo"):
                                yield RichLog(highlight=True, markup=True, id="todo-log")

            with TabPane("Vivado", id="tab-vivado"):
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Label("Trial ID:", classes="field-label")
                        yield Input(placeholder="e.g. abc123def456", id="vivado-trial-id")
                        yield Label("Top Module:", classes="field-label")
                        yield Input(value="", placeholder="auto-detect", id="vivado-top")
                        yield Label("Xilinx Part:", classes="field-label")
                        yield Input(value=os.getenv("VIVADO_PART", "xc7a35tcpg236-1"), id="vivado-part")
                        with Horizontal(classes="btn-group"):
                            yield Button("Run Vivado", variant="success", id="btn-vivado")
                            yield Button("Stop", variant="error", id="btn-vivado-stop", disabled=True)
                        yield Button("Make Project", id="btn-vivado-mkproj")
                        yield Button("Open Vivado GUI", id="btn-vivado-gui")

                    with Vertical(classes="right-panel"):
                        yield Label("Status:")
                        yield Label("No runs yet.", classes="status-view", id="vivado-status")
                        yield Label("Results:")
                        yield RichLog(highlight=True, markup=True, classes="log-view", id="vivado-log")

            with TabPane("Database History", id="tab-db"):
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Label("Search Pattern:", classes="field-label")
                        yield Input(placeholder="e.g. counter, fifo", id="db-search")
                        with Horizontal(classes="btn-group"):
                            yield Button("Refresh", variant="primary", id="btn-db-refresh")
                            yield Button("Clear All", variant="error", id="btn-db-clear")
                        with Horizontal():
                            yield Button("Vivado →", id="btn-db-to-vivado")
                    with Vertical(classes="right-panel"):
                        yield Label("Saved Trial Records:")
                        yield DataTable(classes="opt-trials-view", id="db-table", cursor_type="row")
                        yield Label("Selected Code:")
                        yield RichLog(highlight=True, markup=True, classes="code-view", id="db-code")

        yield Footer()

    SPINNER_CHARS = ["◐", "◓", "◑", "◒"]
    STEP_LABELS = {
        "plan": "PLAN",
        "gen": "GEN",
        "todo": "TODO",
        "fix": "FIX",
        "syntax": "SYNTAX",
        "sim": "SIM",
        "ppa": "PPA",
        "db": "DB",
    }

    def on_mount(self) -> None:
        self.run_log_buffer: list[str] = []
        self._spinner_idx = 0
        self._spinner_steps: set[str] = set()
        self.set_interval(0.15, self._update_spinner)

        db_table = self.query_one("#db-table", DataTable)
        db_table.add_columns("Trial ID", "Problem ID", "Parameters", "Passed", "Duration")
        self.refresh_database_table()
        self.check_docker_worker = self._check_and_start_docker()

    # ── Docker helpers ────────────────────────────────────────────────────────

    def _set_docker_label(self, state: str, msg: str) -> None:
        label = self.query_one("#docker-status", Label)
        label.remove_class("docker-checking", "docker-starting", "docker-running", "docker-stopped")
        label.add_class(f"docker-{state}")
        label.update(msg)

    @staticmethod
    def _docker_running() -> bool:
        import subprocess as _sp
        try:
            return _sp.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
        except Exception:
            return False

    @staticmethod
    def _find_docker_desktop() -> str | None:
        candidates = [
            "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe",
            "/mnt/c/Program Files (x86)/Docker/Docker/Docker Desktop.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return None

    @work(thread=True)
    def _check_and_start_docker(self) -> None:
        import subprocess as _sp, time as _time

        self.call_from_thread(self._set_docker_label, "checking", "◌ Docker: Checking...")

        if self._docker_running():
            self.call_from_thread(self._set_docker_label, "running", "● Docker: Running")
            return

        # Not running — try to start Docker Desktop
        desktop = self._find_docker_desktop()
        if desktop:
            self.call_from_thread(self._set_docker_label, "starting", "◐ Docker: Starting...")
            try:
                _sp.Popen([desktop], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            except Exception:
                pass

            for _ in range(20):          # poll up to 60 s
                _time.sleep(3)
                if self._docker_running():
                    self.call_from_thread(self._set_docker_label, "running", "● Docker: Running")
                    return

            self.call_from_thread(self._set_docker_label, "stopped", "✗ Docker: Timed out")
        else:
            self.call_from_thread(self._set_docker_label, "stopped", "✗ Docker: Not found")

    # ── Spinner ───────────────────────────────────────────────────────────────

    def _update_spinner(self) -> None:
        if not self._spinner_steps:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_CHARS)
        char = self.SPINNER_CHARS[self._spinner_idx]
        for step_name in list(self._spinner_steps):
            base = self.STEP_LABELS.get(step_name, step_name.upper())
            try:
                self.query_one(f"#step-{step_name}", Label).update(f"{char} {base}")
            except Exception:
                pass

    def update_todo_summary(self, trial_id: str) -> None:
        todo_summary = self.query_one("#todo-summary", Label)
        mcp = MCPServer()
        todos = mcp.read_todos(trial_id)
        if not todos:
            todo_summary.update("No TODOs")
            return

        total = len(todos)
        active = next((todo for todo in todos if todo.get("status") in {"active", "pending"}), None)
        if active is None:
            todo_summary.update(f"{total}/{total} (done)")
            return

        order = active.get("order", 1)
        name = active.get("bug", "").strip() or active.get("id", "task")
        todo_summary.update(f"{order}/{total} ({name})")

    def refresh_database_table(self) -> None:
        db_table = self.query_one("#db-table", DataTable)
        db_table.clear()
        mcp = MCPServer()
        search_pattern = self.query_one("#db-search", Input).value.strip()

        if search_pattern:
            trials = mcp.get_successful_trials_by_pattern(search_pattern)
        else:
            # Query all by using empty pattern or generic history
            try:
                with mcp._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT trial_id, problem_id, params, pass, duration_ms FROM trials ORDER BY timestamp DESC LIMIT 50")
                    trials = [dict(row) for row in cursor.fetchall()]
            except Exception:
                trials = []

        for t in trials:
            params_str = t.get("params")
            if isinstance(params_str, str):
                params_str = json.loads(params_str)
            db_table.add_row(
                t.get("trial_id", "?"),
                t.get("problem_id", "?"),
                json.dumps(params_str or {}),
                "PASS" if t.get("pass_") or t.get("pass") else "FAIL",
                f"{t.get('duration_ms', 0):.0f}ms",
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "run-problem-id":
            val = event.value
            is_cvdp = val and val != "custom" and val != Select.BLANK
            self.query_one("#btn-to-vivado", Button).disabled = not is_cvdp
            if is_cvdp:
                try:
                    cvdp = CVDPDataset()
                    prob = cvdp.get_by_id(val)
                    if prob:
                        self.query_one("#run-spec", Input).value = prob.prompt
                except Exception:
                    pass
            else:
                self.query_one("#run-spec", Input).value = "Implement a simple 2-to-1 multiplexer"

    def set_step_status(self, step_name: str, status: str) -> None:
        try:
            label = self.query_one(f"#step-{step_name}", Label)
            label.remove_class("inactive", "active", "success", "fail")
            label.add_class(status)
            if status == "active":
                self._spinner_steps.add(step_name)
            else:
                self._spinner_steps.discard(step_name)
                base = self.STEP_LABELS.get(step_name, step_name.upper())
                label.update(base)
            if step_name == "sim":
                try:
                    self.query_one("#btn-kill-sim", Button).disabled = (status != "active")
                except Exception:
                    pass
        except Exception:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = event.control
        if table.id == "db-table":
            row_data = table.get_row(event.row_key)
            trial_id = row_data[0]
            self._selected_db_trial_id = trial_id
            mcp = MCPServer()
            db_code = self.query_one("#db-code", RichLog)
            db_code.clear()

            try:
                with mcp._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT generated_code FROM trials WHERE trial_id = ?", (trial_id,))
                    row = cursor.fetchone()
                    if row:
                        db_code.write(row[0])
            except Exception as e:
                db_code.write(f"Error loading code: {e}")

    @work(thread=True)
    def run_generation_task(self) -> None:
        run_log = self.query_one("#run-log", RichLog)
        sim_log = self.query_one("#sim-log", RichLog)
        todo_log = self.query_one("#todo-log", RichLog)
        run_code = self.query_one("#run-code", TextArea)
        cancel_token = CancellationToken()
        self.current_cancel_token = cancel_token
        self.run_log_buffer = []
        self.sim_log_buffer: list[str] = []
        current_trial_id: list[str] = []

        self.call_from_thread(run_log.clear)
        self.call_from_thread(sim_log.clear)
        self.call_from_thread(todo_log.clear)
        self.call_from_thread(self.query_one("#todo-summary", Label).update, "No TODOs")
        self.call_from_thread(self._switch_panel, "panel-logs")
        def clear_code():
            run_code.text = ""
        self.call_from_thread(clear_code)
        self.call_from_thread(run_log.write, "Running baseline RTL pipeline...")

        spec = self.query_one("#run-spec", Input).value.strip()
        problem_id = self.query_one("#run-problem-id", Select).value
        params_str = self.query_one("#run-params", Input).value.strip()
        model = self.query_one("#run-model", Input).value.strip() or None
        max_retries = int(self.query_one("#run-max-retries", Input).value)
        show_thinking = self.query_one("#run-show-thinking", Checkbox).value

        # Resolve CVDP problem if requested
        context_files = None
        tb_files = None
        if problem_id and problem_id != "custom" and problem_id != Select.BLANK:
            try:
                cvdp = CVDPDataset()
                prob = cvdp.get_by_id(problem_id)
                if prob:
                    spec = prob.prompt
                    context_files = dict(prob.context)
                    tb_files = prob.get_testbench_files()
            except Exception as e:
                self.call_from_thread(run_log.write, f"[red]Error loading CVDP problem: {e}[/red]")
                self.call_from_thread(run_log.write, "[red]Failed loading CVDP problem.[/red]")
                return

        try:
            params = json.loads(params_str)
        except Exception as e:
            self.call_from_thread(run_log.write, f"[red]Invalid JSON params: {e}[/red]")
            self.call_from_thread(run_log.write, "[red]Failed due to invalid parameters.[/red]")
            return

        backend = create_backend(model=model)
        self.current_backend = backend
        runner = TrialRunner(backend)

        def write_to_log(s):
            text = s.strip()
            if text:
                self.run_log_buffer.append(text)
                self.call_from_thread(run_log.write, escape(text))

        def handle_step_change(step, status):
            cancel_token.raise_if_cancelled()
            self.call_from_thread(self.set_step_status, step, status)
            if step == "sim" and status == "active":
                self.call_from_thread(self._switch_panel, "panel-sim")
            if step in ("todo", "fix") and status in ("success", "active", "fail") and current_trial_id:
                tid = current_trial_id[0]
                self.call_from_thread(self._refresh_todo_panel, tid)

        def on_trial_start(trial_id: str) -> None:
            current_trial_id.clear()
            current_trial_id.append(trial_id)

        accumulated_text = ""
        switched_to_code = False
        def handle_token(token: str):
            cancel_token.raise_if_cancelled()
            nonlocal accumulated_text, switched_to_code
            if token == "":
                accumulated_text = ""
                switched_to_code = False
            else:
                if not switched_to_code:
                    switched_to_code = True
                    self.call_from_thread(self._switch_panel, "panel-code")
                accumulated_text += token
            def update_ui():
                run_code.text = accumulated_text
                lines = accumulated_text.splitlines()
                if lines:
                    run_code.cursor_location = (len(lines) - 1, len(lines[-1]))
            self.call_from_thread(update_ui)

        def _on_sim_line(line: str) -> None:
            stripped = line.strip()
            if stripped:
                self.sim_log_buffer.append(stripped)
                self.call_from_thread(sim_log.write, escape(stripped))

        try:
            with StdoutRedirector(write_to_log):
                code, score = runner.run_trial(
                    spec=spec,
                    problem_id=problem_id or "custom",
                    params=params,
                    context_files=context_files,
                    testbench_files=tb_files,
                    max_retries=max_retries,
                    verbose=True,
                    on_step_change=handle_step_change,
                    on_token=handle_token,
                    enable_thinking=show_thinking,
                    cancel_token=cancel_token,
                    on_trial_start=on_trial_start,
                    on_sim_output=_on_sim_line,
                )
            
            def set_final_code():
                run_code.text = code
            self.call_from_thread(set_final_code)
            status_text = (
                f"Result: {'PASS' if score.pass_ else 'FAIL'} ({score.duration_ms:.0f}ms)\n"
                f"  Syntax:     {'PASS' if score.syntax_pass else 'FAIL'}\n"
                f"  Simulation: {'PASS' if score.simulation_pass else 'FAIL'}\n"
                f"  Retries:    {score.retry_count}"
            )
            self.call_from_thread(run_log.write, status_text)
            self.call_from_thread(self.refresh_database_table)
            if getattr(score, "trial_id", ""):
                self.call_from_thread(self.update_todo_summary, score.trial_id)
            else:
                self.call_from_thread(self.query_one("#todo-summary", Label).update, "No TODOs")
        except PipelineCancelled:
            self.call_from_thread(run_log.write, "[yellow]Pipeline cancelled by user.[/yellow]")
        except Exception as e:
            self.call_from_thread(run_log.write, f"[red]Execution error: {e}[/red]")
            for step in ["gen", "todo", "syntax", "sim", "db"]:
                self.call_from_thread(self.set_step_status, step, "fail")
        finally:
            def restore_buttons():
                if getattr(self, "current_cancel_token", None) is cancel_token:
                    self.query_one("#btn-run", Button).disabled = False
                    self.query_one("#btn-stop", Button).disabled = True
                    self.current_cancel_token = None
            self.call_from_thread(restore_buttons)

    @work(thread=True)
    def run_vivado_task(self, trial_id: str, top_module: str | None, part: str, vivado_status: Label, vivado_log: RichLog) -> None:
        self.call_from_thread(vivado_status.update, "Running Vivado synthesis...")
        self.call_from_thread(vivado_log.clear)
        try:
            self._run_vivado_inner(trial_id, top_module, part, vivado_status, vivado_log)
        except Exception as e:
            self.call_from_thread(vivado_status.update, f"[red]Unexpected error: {e}[/red]")
        finally:
            self._restore_vivado_buttons()

    def _run_vivado_inner(self, trial_id: str, top_module: str | None, part: str, vivado_status: Label, vivado_log: RichLog) -> None:
        def log(msg: str) -> None:
            self.call_from_thread(vivado_log.write, msg)

        def status(msg: str) -> None:
            self.call_from_thread(vivado_status.update, msg)

        if not trial_id:
            status("[red]Enter a trial ID.[/red]")
            log("[red]No trial ID entered. Paste a trial ID from the Database tab.[/red]")
            return

        from src.agents.vivado_ppa import VivadoPPAAgent, find_vivado_wsl
        from src.mcp.server import MCPServer

        vbin = os.getenv("VIVADO_BIN") or find_vivado_wsl() or "vivado"
        if vbin == "vivado":
            log("[yellow]Vivado not auto-detected — set VIVADO_BIN in .env[/yellow]")

        out_dir = MCPServer().get_trial_output_dir(trial_id)
        if not out_dir.exists():
            status("[red]Trial not found.[/red]")
            log(f"[red]Trial output directory not found:[/red] {out_dir}")
            log("Use the Database tab to find a valid trial ID.")
            return

        found = list(out_dir.rglob("*.sv")) + list(out_dir.rglob("*.v"))
        if not found:
            status("[red]No RTL files.[/red]")
            log(f"[red]No .sv / .v files found under {out_dir}[/red]")
            return

        rtl_files = {f.name: f.read_text() for f in found}
        tb_found = [f for f in out_dir.rglob("*") if f.is_file() and f.suffix not in (".sv", ".v")]
        testbench_files = {f.name: f.read_text() for f in tb_found} if tb_found else None
        if not top_module:
            top_module = found[0].stem

        log(f"Trial:  {trial_id}")
        log(f"Files:  {[f.name for f in found]}")
        log(f"Vivado: {vbin}")
        log(f"Part:   {part}   Top: {top_module}")
        log("Synthesizing — this may take several minutes...")

        try:
            agent = VivadoPPAAgent(vivado_bin=vbin, part=part)
            result = agent.execute(rtl_files=rtl_files, testbench_files=testbench_files, config={"top_module": top_module})
        except Exception as e:
            status("[red]Vivado error.[/red]")
            log(f"[red]Exception: {e}[/red]")
            return

        if result.pass_:
            m = result.metrics
            luts = m.get("luts", m.get("luts_logic", 0) + m.get("luts_memory", 0))
            timing_met = m.get("timing_met")
            timing_str = "YES" if timing_met else ("NO" if timing_met is False else "N/A")
            status("[bold green]Synthesis PASSED[/bold green]")
            log(f"[bold green]PASS[/bold green] ({result.duration_ms / 1000:.1f}s)")
            log("")
            log("[bold]Utilization:[/bold]")
            log(f"  LUTs:        {luts}")
            log(f"  LUTs Logic:  {m.get('luts_logic', '?')}")
            log(f"  LUTs Memory: {m.get('luts_memory', '?')}")
            log(f"  Registers:   {m.get('registers', '?')}")
            log(f"  DSPs:        {m.get('dsps', '?')}")
            log(f"  BRAMs:       {m.get('brams', '?')}")
            log("")
            log("[bold]Timing:[/bold]")
            log(f"  WNS:         {m.get('wns_ns', 'N/A')} ns")
            log(f"  TNS:         {m.get('tns_ns', 'N/A')} ns")
            log(f"  Timing Met:  {timing_str}")
            log("")
            log("[bold]Power:[/bold]")
            log(f"  Total:       {m.get('total_power_w', '?')} W")
        else:
            err = result.errors[0].get("message", "unknown") if result.errors else "unknown"
            status("[red]Synthesis FAILED[/red]")
            log("[red]Synthesis FAILED[/red]")
            log(f"[red]{err[:2000]}[/red]")

    def _restore_vivado_buttons(self) -> None:
        def restore():
            self.query_one("#btn-vivado", Button).disabled = False
            self.query_one("#btn-vivado-stop", Button).disabled = True
        self.call_from_thread(restore)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR and event.worker is getattr(self, "vivado_worker", None):
            self.query_one("#btn-vivado", Button).disabled = False
            self.query_one("#btn-vivado-stop", Button).disabled = True
            try:
                self.query_one("#vivado-status", Label).update(
                    f"[red]Worker crashed: {event.worker.error}[/red]"
                )
            except Exception:
                pass

    def _build_vivado_project(
        self,
        trial_id: str,
        top_module: str | None,
        part: str,
        vivado_log: RichLog,
    ) -> tuple[Path | None, list[Path], list[Path], str | None, str | None]:
        """Prepare project dir, copy RTL+TB.

        Returns (proj_dir, rtl_found, tb_found, top_module, project_name) or (None, ...) on failure.
        """
        import shutil
        import re
        from src.agents.vivado_ppa import _is_wsl, _win_temp_base
        from src.mcp.server import MCPServer

        wsl_mode = _is_wsl()

        out_dir = MCPServer().get_trial_output_dir(trial_id)
        if not out_dir.exists():
            vivado_log.write(f"[red]Trial not found: {out_dir}[/red]")
            return None, [], [], None, None

        found = list(out_dir.rglob("*.sv")) + list(out_dir.rglob("*.v"))
        if not found:
            vivado_log.write(f"[red]No .sv/.v files found under {out_dir}[/red]")
            return None, [], [], None, None

        # Smart detection of testbenches:
        # 1. Filename contains "tb" or "test"
        # 2. Or the file contains a module declaration without ports (e.g., module tb;)
        tb_found = []
        rtl_found = []
        for f in found:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if "tb" in f.stem.lower() or "test" in f.stem.lower():
                    tb_found.append(f)
                elif re.search(r'\bmodule\s+\w+\s*;', content):
                    tb_found.append(f)
                else:
                    rtl_found.append(f)
            except Exception:
                rtl_found.append(f)

        if not rtl_found:
            rtl_found = [f for f in found if f not in tb_found] or found

        if not top_module:
            top_module = rtl_found[0].stem if rtl_found else tb_found[0].stem

        proj_base_env = os.getenv("VIVADO_PROJECT_DIR")
        if proj_base_env:
            proj_base = Path(proj_base_env)
        elif wsl_mode:
            proj_base = _win_temp_base()
        else:
            import tempfile
            proj_base = Path(tempfile.gettempdir())

        project_name = f"verigen_{trial_id[:12]}"
        proj_dir = proj_base / project_name

        if proj_dir.exists():
            for child in list(proj_dir.iterdir()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink()
                except Exception:
                    pass

        proj_dir.mkdir(parents=True, exist_ok=True)

        extra_files = [f for f in out_dir.rglob("*") if f.is_file() and f.suffix not in (".sv", ".v")]
        for f in rtl_found + tb_found + extra_files:
            shutil.copy2(f, proj_dir / f.name)

        # Generate a dummy or AI-generated testbench if no testbench is found
        if not tb_found:
            tb_code = ""
            spec_prompt = ""
            generated_code = ""
            try:
                with MCPServer()._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT spec_prompt, generated_code FROM trials WHERE trial_id = ?",
                        (trial_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        spec_prompt = row["spec_prompt"] or ""
                        generated_code = row["generated_code"] or ""
            except Exception:
                pass

            if spec_prompt and generated_code:
                try:
                    vivado_log.write("[dim]AI is generating a customized SystemVerilog testbench for this design...[/dim]")
                    from src.llm import create_backend
                    backend = create_backend()
                    system_prompt = (
                        "You are a skilled digital design engineer specializing in SystemVerilog testbenches.\n"
                        "Your task is to write a self-checking SystemVerilog testbench for the provided design.\n"
                        "Generate only valid SystemVerilog code, wrapped in a markdown block starting with ```systemverilog and ending with ```.\n"
                        "Include no conversational text outside the code block."
                    )
                    user_prompt = (
                        f"Here is the specification of the design:\n"
                        f"```\n{spec_prompt}\n```\n\n"
                        f"Here is the generated Verilog/SystemVerilog RTL code of the design:\n"
                        f"```systemverilog\n{generated_code}\n```\n\n"
                        f"Please write a self-checking SystemVerilog testbench module named `tb` that instantiates this design, "
                        f"drives the inputs (including generating a clock and a reset), and performs basic sanity checks or simulation stimulus.\n"
                        f"Ensure the testbench ends with `$finish;` so it terminates. Output the SystemVerilog code inside a ```systemverilog code block."
                    )
                    response = backend.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.2,
                        max_tokens=4096,
                    )
                    content = response.text
                    code_blocks = re.findall(r"```(?:systemverilog|verilog)?\s*(.*?)\s*```", content, re.DOTALL)
                    tb_code = code_blocks[0].strip() if code_blocks else content.strip()
                except Exception as e:
                    vivado_log.write(f"[yellow]Failed to generate testbench using AI: {e}. Falling back to default template.[/yellow]")

            if not tb_code:
                tb_code = f"""`timescale 1ns / 1ps

module tb;
    reg clk;
    reg rst_n;

    // Instantiate Unit Under Test (UUT)
    // Please update the port connections below to match your top module:
    /*
    {top_module} uut (
        .clk(clk),
        .rst_n(rst_n)
    );
    */

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        #20;
        rst_n = 1;
        #100;
        $finish;
    end
endmodule
"""
            tb_code = tb_code.encode("ascii", errors="ignore").decode("ascii")
            dummy_tb_path = proj_dir / "tb.sv"
            dummy_tb_path.write_text(tb_code, encoding="ascii")
            tb_found = [dummy_tb_path]

        vivado_log.write(f"Project: {proj_dir}")
        if rtl_found:
            vivado_log.write(f"RTL:     {[f.name for f in rtl_found]}")
        if tb_found:
            vivado_log.write(f"TB:      {[f.name for f in tb_found]}")

        return proj_dir, rtl_found, tb_found, top_module, project_name

    def _make_vivado_project(self) -> None:
        import shutil
        from src.agents.vivado_ppa import _to_win_path, _is_wsl

        vivado_log    = self.query_one("#vivado-log", RichLog)
        vivado_status = self.query_one("#vivado-status", Label)
        trial_id   = self.query_one("#vivado-trial-id", Input).value.strip()
        top_module = self.query_one("#vivado-top",  Input).value.strip() or None
        part       = self.query_one("#vivado-part", Input).value.strip() or os.getenv("VIVADO_PART", "xc7a35tcpg236-1")

        if not trial_id:
            vivado_status.update("[red]Enter a trial ID.[/red]")
            vivado_log.write("[red]No trial ID entered.[/red]")
            return

        vivado_log.clear()
        result = self._build_vivado_project(trial_id, top_module, part, vivado_log)
        proj_dir, rtl_found, tb_found, top_module, project_name = result
        if not proj_dir:
            vivado_status.update("[red]Failed to create project.[/red]")
            return

        wsl_mode = _is_wsl()
        def win(f: Path) -> str:
            return _to_win_path(f) if wsl_mode else str(f)

        proj_dir_win = win(proj_dir)
        rtl_win = [win(proj_dir / f.name) for f in rtl_found]
        tb_win  = [win(proj_dir / f.name) for f in tb_found]

        rtl_cmds = "\n".join(f"add_files -norecurse -fileset sources_1 {{{s}}}" for s in rtl_win)
        tb_cmds  = "\n".join(f"add_files -norecurse -fileset sim_1 {{{s}}}" for s in tb_win)
        user_top = self.query_one("#vivado-top",  Input).value.strip()
        top_cmd = f"set_property top {user_top} [current_fileset]\n" if user_top else ""
        tcl_content = (
            f"create_project {project_name} {{{proj_dir_win}}} -part {part} -force\n"
            + (f"{rtl_cmds}\n" if rtl_cmds else "")
            + (f"{tb_cmds}\n" if tb_cmds else "")
            + top_cmd
            + f"update_compile_order -fileset sources_1\n"
            + (f"update_compile_order -fileset sim_1\n"
               f"launch_simulation\n" if tb_found else "")
            + "close_project\n"
        )
        tcl_content = tcl_content.encode("ascii", errors="ignore").decode("ascii")
        tcl_path = proj_dir / "create_project.tcl"
        tcl_path.write_text(tcl_content, encoding="ascii")
        tcl_win = _to_win_path(tcl_path) if wsl_mode else str(tcl_path)

        import subprocess as _sp
        from src.agents.vivado_ppa import find_vivado_wsl, _build_cmd
        vbin = os.getenv("VIVADO_BIN") or find_vivado_wsl() or "vivado"
        if vbin == "vivado":
            vivado_status.update("[yellow]Vivado not found, TCL only.[/yellow]")
            vivado_log.write("[yellow]Set VIVADO_BIN in .env to auto-create project structure.[/yellow]")
            vivado_log.write(f"[green]TCL: {tcl_win}[/green]")
            return

        vivado_status.update("Creating Vivado project...")
        vivado_log.write("[dim]Running vivado -mode batch...[/dim]")
        try:
            cmd = _build_cmd(vbin, tcl_win, wsl_mode)
            proc = _sp.run(
                cmd, capture_output=True, text=False, timeout=120,
            )
            # Smart decode function for local Chinese encoding (CP950/GBK)
            def decode_bytes(data: bytes) -> str:
                if not data:
                    return ""
                for enc in ("utf-8", "cp950", "gbk"):
                    try:
                        return data.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return data.decode("utf-8", errors="replace")

            stdout_str = decode_bytes(proc.stdout)
            stderr_str = decode_bytes(proc.stderr)

            if proc.returncode == 0:
                vivado_status.update("Project ready.")
                vivado_log.write(f"[green]Project created: {proj_dir_win}[/green]")
                vivado_log.write(f"[green]  {project_name}.xpr[/green]")
                vivado_log.write(f"[green]  {project_name}.srcs/sources_1/imports/[/green]")
                if tb_found:
                    vivado_log.write(f"[green]  {project_name}.srcs/sim_1/imports/[/green]")
            else:
                vivado_status.update("[red]Vivado batch failed.[/red]")
                vivado_log.write(f"[red]{stderr_str[:1000]}[/red]")
        except _sp.TimeoutExpired:
            vivado_status.update("[red]Vivado timed out.[/red]")
            vivado_log.write("[red]Vivado batch timed out (120s)[/red]")
        except FileNotFoundError:
            vivado_status.update("[red]Vivado not found.[/red]")
            vivado_log.write(f"[red]Binary not found: {vbin}[/red]")

    def _launch_vivado_with_tcl(self, tcl_win: str | None = None) -> None:
        import subprocess
        from src.agents.vivado_ppa import find_vivado_wsl, _to_win_path, _is_wsl

        vivado_log    = self.query_one("#vivado-log", RichLog)
        vivado_status = self.query_one("#vivado-status", Label)

        vbin = os.getenv("VIVADO_BIN") or find_vivado_wsl() or "vivado"
        if vbin == "vivado":
            vivado_status.update("[red]Vivado not found.[/red]")
            vivado_log.write("[red]Set VIVADO_BIN in .env to point to vivado.bat[/red]")
            return

        wsl_mode = _is_wsl() and vbin.startswith("/mnt/")

        source_args = ["-source", tcl_win, "-nojournal", "-nolog"] if tcl_win else []
        if wsl_mode:
            win_bat = _to_win_path(vbin)
            cmd = ["cmd.exe", "/c", "start", '""', win_bat,
                   "-mode", "gui"] + source_args
        else:
            cmd = [vbin, "-mode", "gui"] + source_args

        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            vivado_status.update("Vivado GUI launched.")
            vivado_log.write("[green]Vivado GUI launched with project.[/green]")
        except Exception as e:
            vivado_status.update("[red]Failed to launch GUI.[/red]")
            vivado_log.write(f"[red]Error: {e}[/red]")

    def _launch_vivado_gui(self) -> None:
        import subprocess
        from src.agents.vivado_ppa import find_vivado_wsl, _to_win_path, _is_wsl

        vivado_log    = self.query_one("#vivado-log", RichLog)
        vivado_status = self.query_one("#vivado-status", Label)
        trial_id   = self.query_one("#vivado-trial-id", Input).value.strip()
        top_module = self.query_one("#vivado-top",  Input).value.strip() or None
        part       = self.query_one("#vivado-part", Input).value.strip() or os.getenv("VIVADO_PART", "xc7a35tcpg236-1")

        if trial_id:
            vivado_log.clear()
            result = self._build_vivado_project(trial_id, top_module, part, vivado_log)
            proj_dir, rtl_found, tb_found, top_module, project_name = result
            if not proj_dir:
                return

            wsl_mode = _is_wsl()
            def win(f: Path) -> str:
                return _to_win_path(f) if wsl_mode else str(f)

            proj_dir_win = win(proj_dir)
            rtl_win = [win(proj_dir / f.name) for f in rtl_found]
            tb_win  = [win(proj_dir / f.name) for f in tb_found]

            rtl_cmds = "\n".join(f"add_files -norecurse -fileset sources_1 {{{s}}}" for s in rtl_win)
            tb_cmds  = "\n".join(f"add_files -norecurse -fileset sim_1 {{{s}}}" for s in tb_win)
            user_top = self.query_one("#vivado-top",  Input).value.strip()
            top_cmd = f"set_property top {user_top} [current_fileset]\n" if user_top else ""
            tcl_content = (
                f"create_project {project_name} {{{proj_dir_win}}} -part {part} -force\n"
                + (f"{rtl_cmds}\n" if rtl_cmds else "")
                + (f"{tb_cmds}\n" if tb_cmds else "")
                + top_cmd
                + f"update_compile_order -fileset sources_1\n"
                + (f"update_compile_order -fileset sim_1\n"
                   f"launch_simulation\n" if tb_found else "")
            )
            tcl_content = tcl_content.encode("ascii", errors="ignore").decode("ascii")
            tcl_path = proj_dir / "open_project.tcl"
            tcl_path.write_text(tcl_content, encoding="ascii")
            tcl_win = _to_win_path(tcl_path) if wsl_mode else str(tcl_path)

            self._launch_vivado_with_tcl(tcl_win)
        else:
            self._launch_vivado_with_tcl()

    _PANEL_IDS = ["panel-logs", "panel-sim", "panel-code", "panel-todo"]

    def _switch_panel(self, panel_id: str) -> None:
        self.query_one("#panel-switcher", ContentSwitcher).current = panel_id
        for pid in self._PANEL_IDS:
            btn_id = f"btn-{pid}"
            try:
                btn = self.query_one(f"#{btn_id}", Button)
                if pid == panel_id:
                    btn.add_class("panel-tab-active")
                else:
                    btn.remove_class("panel-tab-active")
            except Exception:
                pass

    def _refresh_todo_panel(self, trial_id: str) -> None:
        from src.mcp.server import MCPServer
        todo_log = self.query_one("#todo-log", RichLog)
        todo_log.clear()
        try:
            todos = MCPServer().read_todos(trial_id)
        except Exception:
            return
        if not todos:
            todo_log.write("[dim]No TODOs yet.[/dim]")
            return
        STATUS_COLOR = {
            "done": "green", "active": "yellow",
            "failed": "red", "pending": "white",
        }
        todo_log.write(f"[bold]Trial {trial_id} — {len(todos)} item(s)[/bold]\n")
        for t in todos:
            status = t.get("status", "pending")
            color = STATUS_COLOR.get(status, "white")
            tid = t.get("id", "?")
            loc = t.get("location", {})
            line_info = f" line {loc.get('line_start', '?')}" if loc.get("line_start") else ""
            todo_log.write(
                f"[{color}][{status.upper():7}][/{color}] "
                f"[bold]{tid}[/bold]{line_info}"
            )
            if t.get("bug"):
                todo_log.write(f"  [dim]BUG:[/dim]  {t['bug']}")
            if t.get("fix"):
                todo_log.write(f"  [dim]FIX:[/dim]  {t['fix']}")
            todo_log.write("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-docker-retry":
            self.check_docker_worker = self._check_and_start_docker()
            return
        for pid in self._PANEL_IDS:
            if button_id == f"btn-{pid}":
                self._switch_panel(pid)
                return
        if button_id == "btn-kill-sim":
            if hasattr(self, "current_cancel_token") and self.current_cancel_token:
                self.current_cancel_token.kill_processes()
            return
        if button_id == "btn-run":
            self.query_one("#btn-run", Button).disabled = True
            self.query_one("#btn-stop", Button).disabled = False
            self.generator_worker = self.run_generation_task()
        elif button_id == "btn-stop":
            if hasattr(self, "current_cancel_token") and self.current_cancel_token:
                self.current_cancel_token.cancel()
            if hasattr(self, "generator_worker"):
                self.generator_worker.cancel()
            if hasattr(self, "current_backend") and self.current_backend:
                self.current_backend.abort()
            self.query_one("#btn-run", Button).disabled = False
            self.query_one("#btn-stop", Button).disabled = True
            for step in ["gen", "todo", "syntax", "sim", "db"]:
                self.set_step_status(step, "inactive")
        elif button_id == "btn-copy-logs":
            log_text = "\n".join(getattr(self, "run_log_buffer", []))
            self.copy_to_clipboard(log_text)
            self.query_one("#run-log", RichLog).write(
                "Copied pipeline logs to clipboard." if log_text else "No pipeline logs to copy."
            )
        elif button_id == "btn-copy-sim-log":
            sim_text = "\n".join(getattr(self, "sim_log_buffer", []))
            self.copy_to_clipboard(sim_text)
            self.query_one("#run-log", RichLog).write(
                "Copied SIM log to clipboard." if sim_text else "No SIM log to copy."
            )
        elif button_id == "btn-to-vivado":
            self.query_one(TabbedContent).active = "tab-vivado"
        elif button_id == "btn-vivado":
            trial_id = self.query_one("#vivado-trial-id", Input).value.strip()
            top_module = self.query_one("#vivado-top", Input).value.strip() or None
            part = self.query_one("#vivado-part", Input).value.strip() or os.getenv("VIVADO_PART", "xc7a35tcpg236-1")
            vivado_status = self.query_one("#vivado-status", Label)
            vivado_log = self.query_one("#vivado-log", RichLog)
            self.query_one("#btn-vivado", Button).disabled = True
            self.query_one("#btn-vivado-stop", Button).disabled = False
            self.vivado_worker = self.run_vivado_task(trial_id, top_module, part, vivado_status, vivado_log)
        elif button_id == "btn-vivado-stop":
            if hasattr(self, "vivado_worker"):
                self.vivado_worker.cancel()
            self.query_one("#vivado-status", Label).update("Vivado stopped.")
            self.query_one("#btn-vivado", Button).disabled = False
            self.query_one("#btn-vivado-stop", Button).disabled = True
        elif button_id == "btn-vivado-mkproj":
            self._make_vivado_project()
        elif button_id == "btn-vivado-gui":
            self._launch_vivado_gui()
        elif button_id == "btn-db-refresh":
            self.refresh_database_table()
        elif button_id == "btn-db-clear":
            mcp = MCPServer()
            count = mcp.clear_all_trials()
            self.refresh_database_table()
            self.query_one("#db-code", RichLog).clear()
            self.query_one("#db-search", Input).value = ""
            self.query_one("#db-code", RichLog).write(
                f"Cleared {count} trial record(s)." if count else "Database already empty."
            )
        elif button_id == "btn-db-to-vivado":
            trial_id = getattr(self, "_selected_db_trial_id", "")
            if trial_id:
                self.query_one(TabbedContent).active = "tab-vivado"
                def fill_id():
                    try:
                        self.query_one("#vivado-trial-id", Input).value = trial_id
                    except Exception:
                        pass
                self.set_timer(0.1, fill_id)
            else:
                self.query_one("#db-code", RichLog).write("[yellow]Select a trial row first.[/yellow]")


if __name__ == "__main__":
    app = VeriGenTUI()
    app.run()

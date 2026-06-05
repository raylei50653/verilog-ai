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
)
from textual import work

from src.llm import create_backend
from src.pipeline import TrialRunner
from src.cvdp.loader import CVDPDataset
from src.mcp.server import MCPServer
from src.optimizer.optuna_runner import OptunaRunner


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
    #btn-run {
        width: 60%;
        background: #50fa7b;
        color: #282a36;
    }
    #btn-run:hover {
        background: #85fa9b;
    }
    #btn-stop {
        width: 40%;
        background: #ff5555;
        color: #f8f8f2;
    }
    #btn-stop:hover {
        background: #ff6e6e;
    }
    .log-view {
        height: 30%;
        border: solid #44475a;
        margin-top: 1;
        background: #282a36;
    }
    .code-view {
        height: 40%;
        border: solid #44475a;
        margin-top: 1;
        background: #282a36;
    }
    .status-view {
        height: 12%;
        background: #282a36;
        border: solid #44475a;
        padding: 1;
        color: #f8f8f2;
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

    TITLE = "VeriGen - RTL Generator & Optimizer"
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
                        yield Checkbox("Enable RTL Reuse", value=False, id="run-reuse")
                        yield Checkbox("Show AI Thinking", value=False, id="run-show-thinking")
                        with Horizontal(classes="btn-group"):
                            yield Button("Generate RTL", variant="success", id="btn-run")
                            yield Button("Stop", variant="error", id="btn-stop", disabled=True)

                    with Vertical(classes="right-panel"):
                        yield Label("Pipeline Progress Status:")
                        with Horizontal(id="runner-status-bar"):
                            yield Label("PLAN", id="step-plan", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("GEN", id="step-gen", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("SYNTAX", id="step-syntax", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("SIM", id="step-sim", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("PPA", id="step-ppa", classes="status-step inactive")
                            yield Label(" -> ", classes="status-arrow")
                            yield Label("DB", id="step-db", classes="status-step inactive")
                        yield Label("Pipeline Logs:")
                        yield RichLog(highlight=True, markup=True, classes="log-view", id="run-log")
                        yield Label("Generated Verilog Code:")
                        yield TextArea(read_only=True, language="verilog", classes="code-view", id="run-code")
                        yield Label("Status Summary:")
                        yield Label("Ready to run.", classes="status-view", id="run-status")

            with TabPane("Optimizer", id="tab-opt"):
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Label("CVDP Problem ID:", classes="field-label")
                        opt_options = [
                            (f"[{CVDP_PROBLEM_DIFFS.get(pid, 'unknown').upper()}] {pid}", pid)
                            for pid in CVDP_PROBLEM_IDS
                        ]
                        default_opt_val = "cvdp_copilot_16qam_mapper_0001" if "cvdp_copilot_16qam_mapper_0001" in CVDP_PROBLEM_IDS else CVDP_PROBLEM_IDS[0]
                        yield Select(opt_options, value=default_opt_val, id="opt-problem-id")
                        yield Label("Objective Metric:", classes="field-label")
                        yield Input(value="area", id="opt-objective")
                        yield Label("Trials count:", classes="field-label")
                        yield Input(value="5", id="opt-trials")
                        yield Label("Model Name Override:", classes="field-label")
                        yield Input(placeholder="e.g. deepseek-coder-v2", id="opt-model")
                        yield Checkbox("Enable RTL Reuse", value=True, id="opt-reuse")
                        yield Button("Start Optimization", variant="success", id="btn-optimize", classes="run-btn")

                    with Vertical(classes="right-panel"):
                        yield Label("Best Score Summary:")
                        yield Label("No runs yet.", classes="status-view", id="opt-status")
                        yield Label("Optuna Trial History:")
                        yield DataTable(classes="opt-trials-view", id="opt-table")

            with TabPane("Database History", id="tab-db"):
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Label("Search Pattern:", classes="field-label")
                        yield Input(placeholder="e.g. counter, fifo", id="db-search")
                        yield Button("Refresh", variant="primary", id="btn-db-refresh", classes="run-btn")
                    with Vertical(classes="right-panel"):
                        yield Label("Saved Trial Records:")
                        yield DataTable(classes="opt-trials-view", id="db-table")
                        yield Label("Selected Code:")
                        yield RichLog(highlight=True, markup=True, classes="code-view", id="db-code")

        yield Footer()

    def on_mount(self) -> None:
        # Initialize tables
        opt_table = self.query_one("#opt-table", DataTable)
        opt_table.add_columns("Trial #", "Parameters", "Metric Value", "State")

        db_table = self.query_one("#db-table", DataTable)
        db_table.add_columns("Trial ID", "Problem ID", "Parameters", "Passed", "Duration")
        self.refresh_database_table()

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
            if val and val != "custom" and val != Select.BLANK:
                try:
                    cvdp = CVDPDataset()
                    prob = cvdp.get_by_id(val)
                    if prob:
                        self.query_one("#run-spec", Input).value = prob.prompt
                except Exception:
                    pass
            elif val == "custom":
                self.query_one("#run-spec", Input).value = "Implement a simple 2-to-1 multiplexer"

    def set_step_status(self, step_name: str, status: str) -> None:
        try:
            label = self.query_one(f"#step-{step_name}", Label)
            label.remove_class("inactive", "active", "success", "fail")
            label.add_class(status)
        except Exception:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = event.control
        if table.id == "db-table":
            row_data = table.get_row(event.row_key)
            trial_id = row_data[0]
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
        run_code = self.query_one("#run-code", TextArea)
        run_status = self.query_one("#run-status", Label)

        self.call_from_thread(run_log.clear)
        def clear_code():
            run_code.text = ""
        self.call_from_thread(clear_code)
        self.call_from_thread(run_status.update, "Running RTL generation pipeline...")

        spec = self.query_one("#run-spec", Input).value.strip()
        problem_id = self.query_one("#run-problem-id", Select).value
        params_str = self.query_one("#run-params", Input).value.strip()
        model = self.query_one("#run-model", Input).value.strip() or None
        max_retries = int(self.query_one("#run-max-retries", Input).value)
        reuse_rtl = self.query_one("#run-reuse", Checkbox).value
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
                self.call_from_thread(run_status.update, "Failed loading CVDP problem.")
                return

        try:
            params = json.loads(params_str)
        except Exception as e:
            self.call_from_thread(run_log.write, f"[red]Invalid JSON params: {e}[/red]")
            self.call_from_thread(run_status.update, "Failed due to invalid parameters.")
            return

        backend = create_backend(model=model)
        self.current_backend = backend
        runner = TrialRunner(backend)

        def write_to_log(s):
            self.call_from_thread(run_log.write, s.strip())

        def handle_step_change(step, status):
            if hasattr(self, "generator_worker") and self.generator_worker.is_cancelled:
                raise RuntimeError("cancelled by user")
            self.call_from_thread(self.set_step_status, step, status)

        accumulated_text = ""
        def handle_token(token: str):
            if hasattr(self, "generator_worker") and self.generator_worker.is_cancelled:
                raise RuntimeError("cancelled by user")
            nonlocal accumulated_text
            if token == "":
                accumulated_text = ""
            else:
                accumulated_text += token
            def update_ui():
                run_code.text = accumulated_text
                # Scroll to the bottom by placing cursor at the end of text
                lines = accumulated_text.splitlines()
                if lines:
                    run_code.cursor_location = (len(lines) - 1, len(lines[-1]))
            self.call_from_thread(update_ui)

        try:
            with StdoutRedirector(write_to_log):
                code, score = runner.run_trial(
                    spec=spec,
                    problem_id=problem_id or "custom",
                    params=params,
                    context_files=context_files,
                    testbench_files=tb_files,
                    max_retries=max_retries,
                    enable_rtl_reuse=reuse_rtl,
                    verbose=True,
                    on_step_change=handle_step_change,
                    on_token=handle_token,
                    enable_thinking=show_thinking,
                )
            
            def set_final_code():
                run_code.text = code
            self.call_from_thread(set_final_code)
            status_text = (
                f"Result: {'PASS' if score.pass_ else 'FAIL'} ({score.duration_ms:.0f}ms)\n"
                f"  Syntax:     {'PASS' if score.syntax_pass else 'FAIL'}\n"
                f"  Simulation: {'PASS' if score.simulation_pass else 'FAIL'}\n"
                f"  PPA Score:  {json.dumps(score.ppa_metrics or {})}"
            )
            self.call_from_thread(run_status.update, status_text)
            self.call_from_thread(self.refresh_database_table)
        except Exception as e:
            # If worker was cancelled, silently exit to avoid overriding a new run's UI state
            if hasattr(self, "generator_worker") and self.generator_worker.is_cancelled:
                return

            self.call_from_thread(run_log.write, f"[red]Execution error: {e}[/red]")
            self.call_from_thread(run_status.update, "Pipeline execution error.")
            for step in ["plan", "gen", "syntax", "sim", "ppa", "db"]:
                self.call_from_thread(self.set_step_status, step, "fail")
        finally:
            if hasattr(self, "generator_worker") and self.generator_worker.is_cancelled:
                return
            def restore_buttons():
                self.query_one("#btn-run", Button).disabled = False
                self.query_one("#btn-stop", Button).disabled = True
            self.call_from_thread(restore_buttons)

    @work(thread=True)
    def run_optimization_task(self) -> None:
        opt_status = self.query_one("#opt-status", Label)
        opt_table = self.query_one("#opt-table", DataTable)

        self.call_from_thread(opt_status.update, "Initializing Optuna Study...")
        self.call_from_thread(opt_table.clear)

        problem_id = self.query_one("#opt-problem-id", Select).value
        objective_metric = self.query_one("#opt-objective", Input).value.strip()
        trials_count = int(self.query_one("#opt-trials", Input).value)
        model = self.query_one("#opt-model", Input).value.strip() or None
        reuse_rtl = self.query_one("#opt-reuse", Checkbox).value

        # Fetch problem
        try:
            cvdp = CVDPDataset()
            problem = cvdp.get_by_id(problem_id)
            if not problem:
                self.call_from_thread(opt_status.update, f"[red]Problem not found: {problem_id}[/red]")
                return
        except Exception as e:
            self.call_from_thread(opt_status.update, f"[red]Error: {e}[/red]")
            return

        storage_uri = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna/optuna.db")
        backend = create_backend(model=model)
        runner = TrialRunner(backend)
        opt_runner = OptunaRunner(trial_runner=runner, storage_uri=storage_uri)

        def optuna_callback(study, trial):
            state_str = str(trial.state).replace("TrialState.", "")
            val_str = f"{trial.value}" if trial.value is not None and trial.value < 1e8 else "Penalty"
            self.call_from_thread(
                opt_table.add_row,
                f"{trial.number}",
                json.dumps(trial.params),
                val_str,
                state_str,
            )
            # Update summary
            best_val = f"{study.best_value}" if study.best_value < 1e8 else "No passing runs"
            self.call_from_thread(
                opt_status.update,
                f"Study: {study.study_name}\n"
                f"Best Objective: {best_val}\n"
                f"Best Parameters: {json.dumps(study.best_params)}"
            )

        try:
            # Modify study optimization call temporarily to plug callback
            study_name = f"study_{problem.id}_{objective_metric}"
            import optuna
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_uri,
                direction="minimize",
                load_if_exists=True,
            )

            parameter_space = opt_runner.get_space_for_problem(problem)

            def objective(trial):
                trial_params = {}
                for param_name, param_cfg in parameter_space.items():
                    ptype = param_cfg.get("type")
                    if ptype == "categorical":
                        trial_params[param_name] = trial.suggest_categorical(param_name, param_cfg["choices"])
                    elif ptype == "int":
                        low = param_cfg["low"]
                        high = param_cfg["high"]
                        step = param_cfg.get("step", 1)
                        trial_params[param_name] = trial.suggest_int(param_name, low, high, step=step)
                    elif ptype == "float":
                        low = param_cfg["low"]
                        high = param_cfg["high"]
                        step = param_cfg.get("step")
                        log = param_cfg.get("log", False)
                        trial_params[param_name] = trial.suggest_float(param_name, low, high, step=step, log=log)

                code, score = runner.run_trial(
                    spec=problem.prompt,
                    problem_id=problem.id,
                    params=trial_params,
                    context_files=dict(problem.context) if problem.context else None,
                    testbench_files=problem.get_testbench_files() if problem.has_testbench() else None,
                    enable_rtl_reuse=reuse_rtl,
                    verbose=False,
                )
                if not score.pass_:
                    return 1e9
                val = (score.ppa_metrics or {}).get(objective_metric)
                if val is None:
                    if objective_metric == "area":
                        val = (score.ppa_metrics or {}).get("num_cells")
                    elif objective_metric == "wires":
                        val = (score.ppa_metrics or {}).get("num_wires")
                return float(val) if val is not None else 1e8

            study.optimize(objective, n_trials=trials_count, callbacks=[optuna_callback])
        except Exception as e:
            self.call_from_thread(opt_status.update, f"[red]Optimization error: {e}[/red]")
            return

        self.call_from_thread(self.refresh_database_table)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-run":
            self.query_one("#btn-run", Button).disabled = True
            self.query_one("#btn-stop", Button).disabled = False
            self.generator_worker = self.run_generation_task()
        elif button_id == "btn-stop":
            if hasattr(self, "generator_worker"):
                self.generator_worker.cancel()
            if hasattr(self, "current_backend") and self.current_backend:
                self.current_backend.abort()
            self.query_one("#run-status", Label).update("Pipeline cancelled by user.")
            self.query_one("#btn-run", Button).disabled = False
            self.query_one("#btn-stop", Button).disabled = True
            for step in ["plan", "gen", "syntax", "sim", "ppa", "db"]:
                self.set_step_status(step, "inactive")
        elif button_id == "btn-optimize":
            self.run_optimization_task()
        elif button_id == "btn-db-refresh":
            self.refresh_database_table()


if __name__ == "__main__":
    app = VeriGenTUI()
    app.run()

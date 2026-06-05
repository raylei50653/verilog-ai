import json
import optuna
from pathlib import Path
from typing import Any
from src.pipeline import TrialRunner
from src.cvdp.loader import CVDPDataset, CVDPProblem


class OptunaRunner:
    def __init__(
        self,
        trial_runner: TrialRunner,
        config_path: str | Path | None = None,
        storage_uri: str | None = None,
    ):
        self.trial_runner = trial_runner
        self.config_path = Path(config_path) if config_path else Path("config/optuna_params.json")
        self.storage_uri = storage_uri
        self.params_config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except Exception:
                pass
        # Fallback default hardcoded config
        return {
            "default": {
                "parameters": {
                    "data_width": {"type": "categorical", "choices": [8, 16, 32, 64]}
                }
            },
            "fifo": {
                "parameters": {
                    "depth": {"type": "categorical", "choices": [4, 8, 16, 32, 64, 128]},
                    "data_width": {"type": "categorical", "choices": [8, 16, 32, 64]}
                }
            },
            "arbiter": {
                "parameters": {
                    "num_requestors": {"type": "categorical", "choices": [2, 4, 8, 16]},
                    "policy": {"type": "categorical", "choices": ["fixed_priority", "round_robin"]}
                }
            },
            "counter": {
                "parameters": {
                    "width": {"type": "categorical", "choices": [4, 8, 16, 32, 64]},
                    "direction": {"type": "categorical", "choices": ["up", "down", "updown"]}
                }
            },
            "fsm": {
                "parameters": {
                    "encoding": {"type": "categorical", "choices": ["binary", "onehot", "gray"]}
                }
            },
            "pipeline": {
                "parameters": {
                    "stages": {"type": "int", "low": 1, "high": 8},
                    "data_width": {"type": "categorical", "choices": [8, 16, 32, 64]}
                }
            }
        }

    def get_space_for_problem(self, problem: CVDPProblem) -> dict:
        # Match by problem ID first
        for key, value in self.params_config.items():
            if key != "default" and key in problem.id:
                return value.get("parameters", {})
        # Match by categories
        for cat in problem.categories:
            for key, value in self.params_config.items():
                if key != "default" and key in cat:
                    return value.get("parameters", {})
        # Fallback to default
        return self.params_config.get("default", {}).get("parameters", {})

    def optimize_problem(
        self,
        problem: CVDPProblem,
        n_trials: int = 50,
        objective_metric: str = "area",
        enable_rtl_reuse: bool = False,
        verbose: bool = False,
    ) -> optuna.Study:
        # Define study name
        study_name = f"study_{problem.id}_{objective_metric}"

        # If SQLite storage is configured, ensure parent directories exist
        if self.storage_uri and self.storage_uri.startswith("sqlite:///"):
            db_path = Path(self.storage_uri.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Optuna Study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_uri,
            direction="minimize",
            load_if_exists=True,
        )

        parameter_space = self.get_space_for_problem(problem)

        def objective(trial: optuna.Trial) -> float:
            # 1. Suggest parameters
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

            if verbose:
                print(f"\n[Trial {trial.number}] Suggested parameters: {trial_params}")

            # 2. Run the trial via pipeline
            try:
                code, score = self.trial_runner.run_trial(
                    spec=problem.prompt,
                    problem_id=problem.id,
                    params=trial_params,
                    context_files=dict(problem.context) if problem.context else None,
                    testbench_files=problem.get_testbench_files() if problem.has_testbench() else None,
                    enable_rtl_reuse=enable_rtl_reuse,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"[Trial {trial.number}] Run trial threw exception: {e}")
                return 1e9  # Penalty for system failure

            # 3. Check for pass/fail
            if not score.pass_:
                if verbose:
                    print(f"[Trial {trial.number}] Functional verification failed. Returning penalty.")
                return 1e9  # Penalty for verification failure

            # 4. Extract target metric
            ppa = score.ppa_metrics or {}
            val = ppa.get(objective_metric)
            if val is None:
                # If metric not directly present, try mapping area/wires
                if objective_metric == "area":
                    val = ppa.get("num_cells")
                elif objective_metric == "wires":
                    val = ppa.get("num_wires")

            if val is None:
                if verbose:
                    print(f"[Trial {trial.number}] Metric '{objective_metric}' not found in PPA metrics: {ppa}")
                return 1e8  # Minor penalty for missing metric on passing design

            try:
                return float(val)
            except (ValueError, TypeError):
                return 1e8

        study.optimize(objective, n_trials=n_trials)
        return study

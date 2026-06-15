import uuid
from pathlib import Path
from typing import Any

from src.workbench.models import ArtifactRef, RunState, Task
from src.workbench.store import WorkbenchStore


class WorkbenchManager:
    def __init__(self, root_dir: str | Path = "data/workbench"):
        self.store = WorkbenchStore(root_dir)

    def init_trial(self, trial_id: str, problem_id: str, spec: str) -> RunState:
        trial_dir = self.store.ensure_trial_dirs(trial_id)
        spec_path = trial_dir / "artifacts" / "spec.md"
        self.store.write_text(spec_path, spec)

        state = RunState(
            trial_id=trial_id,
            problem_id=problem_id,
            current_phase="planning",
        )
        artifacts = [
            ArtifactRef(
                id="spec",
                kind="spec",
                path=str(spec_path.relative_to(trial_dir)),
                metadata={"problem_id": problem_id},
            ).to_dict()
        ]
        self._write_state(trial_id, state)
        self._write_artifacts(trial_id, artifacts)
        self._write_tasks(trial_id, [])
        return state

    def publish_artifact(
        self,
        trial_id: str,
        kind: str,
        content: str,
        *,
        revision: int = 0,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> ArtifactRef:
        trial_dir = self.store.ensure_trial_dirs(trial_id)
        rel_path = self._artifact_relpath(kind, revision, filename)
        abs_path = trial_dir / rel_path
        self.store.write_text(abs_path, content)

        artifact = ArtifactRef(
            id=f"{kind}:{revision}:{uuid.uuid4().hex[:8]}",
            kind=kind,
            path=str(rel_path),
            revision=revision,
            metadata=metadata or {},
        )
        artifacts = self._read_artifacts(trial_id)
        artifacts.append(artifact.to_dict())
        self._write_artifacts(trial_id, artifacts)

        state = self.get_run_state(trial_id)
        state.latest_artifacts[kind] = artifact.id
        if revision > state.current_revision:
            state.current_revision = revision
        self._write_state(trial_id, state)
        return artifact

    def create_task(
        self,
        trial_id: str,
        task_type: str,
        owner_agent: str,
        title: str,
        *,
        input_artifacts: list[str] | None = None,
        depends_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        task = Task(
            id=f"task-{uuid.uuid4().hex[:8]}",
            type=task_type,
            status="pending",
            owner_agent=owner_agent,
            title=title,
            input_artifacts=input_artifacts or [],
            depends_on=depends_on or [],
            metadata=metadata or {},
        )
        tasks = self._read_tasks(trial_id)
        tasks.append(task.to_dict())
        self._write_tasks(trial_id, tasks)
        return task

    def update_task(
        self,
        trial_id: str,
        task_id: str,
        *,
        status: str | None = None,
        output_artifacts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        tasks = self._read_tasks(trial_id)
        for task in tasks:
            if task["id"] != task_id:
                continue
            if status is not None:
                task["status"] = status
            if output_artifacts:
                task["output_artifacts"].extend(output_artifacts)
            if metadata:
                task["metadata"].update(metadata)
            break
        self._write_tasks(trial_id, tasks)

    def set_phase(self, trial_id: str, phase: str) -> None:
        state = self.get_run_state(trial_id)
        state.current_phase = phase
        self._write_state(trial_id, state)

    def record_retry(self, trial_id: str, retry_count: int) -> None:
        state = self.get_run_state(trial_id)
        state.retry_count = retry_count
        self._write_state(trial_id, state)

    def finalize(self, trial_id: str, final_status: str, retry_count: int) -> None:
        state = self.get_run_state(trial_id)
        state.final_status = final_status
        state.retry_count = retry_count
        self._write_state(trial_id, state)

    def get_run_state(self, trial_id: str) -> RunState:
        trial_dir = self.store.trial_dir(trial_id)
        state = self.store.read_json(trial_dir / "run_state.json", None)
        if state is None:
            raise FileNotFoundError(f"Run state not initialized for trial {trial_id}")
        return RunState(**state)

    def list_tasks(self, trial_id: str) -> list[dict[str, Any]]:
        return self._read_tasks(trial_id)

    def list_artifacts(self, trial_id: str) -> list[dict[str, Any]]:
        return self._read_artifacts(trial_id)

    def get_latest_artifact(self, trial_id: str, kind: str) -> dict[str, Any] | None:
        state = self.get_run_state(trial_id)
        artifact_id = state.latest_artifacts.get(kind)
        if not artifact_id:
            return None
        for artifact in self._read_artifacts(trial_id):
            if artifact["id"] == artifact_id:
                return artifact
        return None

    def _artifact_relpath(self, kind: str, revision: int, filename: str | None) -> Path:
        if kind == "architecture":
            return Path("artifacts/architecture.md")
        if kind == "rtl":
            if revision > 0:
                name = filename or f"rev_{revision:03d}.sv"
                return Path("artifacts/rtl/revisions") / name
            return Path("artifacts/rtl/current.sv")
        if kind.startswith("diagnostic:"):
            diag_name = kind.split(":", 1)[1]
            name = filename or f"{diag_name}_rev_{revision:03d}.json"
            return Path("artifacts/diagnostics") / name
        if kind == "repair_plan":
            name = filename or f"repair_rev_{revision:03d}.md"
            return Path("artifacts/diagnostics") / name
        if kind == "ppa":
            name = filename or f"ppa_rev_{revision:03d}.json"
            return Path("artifacts/reports") / name
        name = filename or f"{kind}.txt"
        return Path("artifacts") / name

    def _write_state(self, trial_id: str, state: RunState) -> None:
        self.store.write_json(self.store.trial_dir(trial_id) / "run_state.json", state.to_dict())

    def _write_tasks(self, trial_id: str, tasks: list[dict[str, Any]]) -> None:
        self.store.write_json(self.store.trial_dir(trial_id) / "tasks.json", tasks)

    def _write_artifacts(self, trial_id: str, artifacts: list[dict[str, Any]]) -> None:
        self.store.write_json(self.store.trial_dir(trial_id) / "artifacts.json", artifacts)

    def _read_tasks(self, trial_id: str) -> list[dict[str, Any]]:
        return self.store.read_json(self.store.trial_dir(trial_id) / "tasks.json", [])

    def _read_artifacts(self, trial_id: str) -> list[dict[str, Any]]:
        return self.store.read_json(self.store.trial_dir(trial_id) / "artifacts.json", [])

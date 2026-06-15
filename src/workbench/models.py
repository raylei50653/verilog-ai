from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskStatus = Literal["pending", "active", "blocked", "done", "failed"]
TaskType = Literal["plan", "generate", "syntax_check", "sim_check", "diagnose", "repair", "ppa"]


@dataclass
class ArtifactRef:
    id: str
    kind: str
    path: str
    revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    id: str
    type: TaskType
    status: TaskStatus
    owner_agent: str
    title: str
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    trial_id: str
    problem_id: str
    current_phase: str
    current_revision: int = 0
    retry_count: int = 0
    final_status: str = "running"
    latest_artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

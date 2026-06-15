from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class AgentResult:
    pass_: bool
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    raw_output: str = ""
    duration_ms: float = 0.0
    is_infra_failure: bool = False


class SubAgent(ABC):
    @abstractmethod
    def execute(
        self,
        rtl_files: dict[str, str],
        testbench_files: dict[str, str] | None = None,
        config: dict | None = None,
    ) -> AgentResult: ...

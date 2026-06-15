import json
from pathlib import Path
from typing import Any


class WorkbenchStore:
    def __init__(self, root_dir: str | Path = "data/workbench"):
        self.root_dir = Path(root_dir)

    def trial_dir(self, trial_id: str) -> Path:
        return self.root_dir / trial_id

    def ensure_trial_dirs(self, trial_id: str) -> Path:
        trial_dir = self.trial_dir(trial_id)
        for rel in [
            "artifacts",
            "artifacts/rtl",
            "artifacts/rtl/revisions",
            "artifacts/diagnostics",
            "artifacts/reports",
        ]:
            (trial_dir / rel).mkdir(parents=True, exist_ok=True)
        return trial_dir

    def write_json(self, path: Path, data: dict[str, Any] | list[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except Exception:
            return default

    def write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

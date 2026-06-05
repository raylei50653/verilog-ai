import json
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset, Dataset

CVDP_HF_ID = "nvidia/cvdp-benchmark-dataset"
AVAILABLE_SUBSETS = [
    "cvdp_nonagentic_code_generation_no_commercial",
    "cvdp_nonagentic_code_generation_commercial",
    "cvdp_agentic_code_generation_no_commercial",
    "cvdp_agentic_code_generation_commercial",
]

ALIASES = {
    "nonagentic_no_commercial": "cvdp_nonagentic_code_generation_no_commercial",
    "nonagentic_commercial": "cvdp_nonagentic_code_generation_commercial",
    "agentic_no_commercial": "cvdp_agentic_code_generation_no_commercial",
    "agentic_commercial": "cvdp_agentic_code_generation_commercial",
}


def _row_to_dict(row: dict | Any) -> dict:
    return row


class CVDPProblem:
    __slots__ = ("_row",)

    def __init__(self, row: dict):
        self._row = row

    @property
    def id(self) -> str:
        return self._row["id"]

    @property
    def categories(self) -> list[str]:
        return self._row.get("categories", [])

    @property
    def difficulty(self) -> str:
        for c in self.categories:
            if c in ("easy", "medium", "hard"):
                return c
        return "unknown"

    @property
    def cids(self) -> list[str]:
        return [c for c in self.categories if c.startswith("cid")]

    @property
    def prompt(self) -> str:
        return self._row.get("input", {}).get("prompt", "")

    @property
    def context(self) -> dict[str, str]:
        return self._row.get("input", {}).get("context", {})

    @property
    def output_files(self) -> list[str]:
        return list(self._row.get("output", {}).get("context", {}).keys())

    @property
    def harness_files(self) -> dict[str, str]:
        return self._row.get("harness", {}).get("files", {})

    def get_testbench_files(self) -> dict[str, str]:
        tb: dict[str, str] = {}
        for fname, content in self.harness_files.items():
            if "test" in fname.lower() or "tb" in fname.lower():
                tb[fname] = content
            elif fname == "src/harness_library.py":
                tb[fname] = content
            elif fname == "src/test_runner.py":
                tb[fname] = content
        return tb

    def has_testbench(self) -> bool:
        return len(self.get_testbench_files()) > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "categories": self.categories,
            "prompt": self.prompt,
            "context": self.context,
            "output_files": self.output_files,
            "has_testbench": self.has_testbench(),
        }

    def __repr__(self) -> str:
        return f"CVDPProblem(id={self.id!r}, cats={self.categories})"


class CVDPDataset:
    def __init__(self, subset: str = "nonagentic_no_commercial", data_dir: str = "data/cvdp_dataset"):
        subset_name = ALIASES.get(subset, subset)
        if subset_name not in AVAILABLE_SUBSETS:
            raise ValueError(
                f"Unknown subset '{subset}'. Available: {list(ALIASES.keys())} or {AVAILABLE_SUBSETS}"
            )
        self.subset = subset_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._dataset: Dataset | None = None

    def _ensure_loaded(self) -> Dataset:
        if self._dataset is None:
            saved_path = self.data_dir / self.subset
            if saved_path.exists():
                self._dataset = Dataset.load_from_disk(str(saved_path))
            else:
                self._dataset = load_dataset(CVDP_HF_ID, self.subset, split="eval")
                self._dataset.save_to_disk(str(saved_path))
        return self._dataset

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __getitem__(self, idx: int) -> CVDPProblem:
        row = self._ensure_loaded()[idx]
        return CVDPProblem(dict(row))

    def __iter__(self) -> Iterator[CVDPProblem]:
        for i in range(len(self)):
            yield self[i]

    def get_by_id(self, problem_id: str) -> CVDPProblem | None:
        ds = self._ensure_loaded()
        for row in ds:
            if row["id"] == problem_id:
                return CVDPProblem(dict(row))
        return None

    def get_categories(self) -> list[str]:
        ds = self._ensure_loaded()
        cats: set[str] = set()
        for row in ds:
            for c in row.get("categories", []):
                cats.add(c)
        return sorted(cats)

    def filter_by_category(self, cid: str) -> list[CVDPProblem]:
        ds = self._ensure_loaded()
        return [CVDPProblem(dict(row)) for row in ds if cid in row.get("categories", [])]

    def filter_by_difficulty(self, level: str) -> list[CVDPProblem]:
        ds = self._ensure_loaded()
        return [CVDPProblem(dict(row)) for row in ds if level in row.get("categories", [])]

    def export_index(self, output_path: str | None = None) -> list[dict]:
        index: list[dict] = []
        for problem in self:
            index.append(problem.to_dict())

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(index, indent=2))

        return index

    def export_sample_code(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for problem in self:
            problem_dir = out / problem.id
            problem_dir.mkdir(exist_ok=True)

            (problem_dir / "prompt.txt").write_text(problem.prompt)

            rtl_dir = problem_dir / "rtl"
            rtl_dir.mkdir(exist_ok=True)
            for fname, content in problem.context.items():
                (problem_dir / fname).parent.mkdir(parents=True, exist_ok=True)
                (problem_dir / fname).write_text(content)

            tb_dir = problem_dir / "verif"
            tb_dir.mkdir(exist_ok=True)
            for fname, content in problem.get_testbench_files().items():
                dest = problem_dir / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)


def download_dataset(subset: str, data_dir: str = "data/cvdp_dataset") -> None:
    cvdp = CVDPDataset(subset=subset, data_dir=data_dir)
    cvdp._ensure_loaded()
    print(f"Downloaded {len(cvdp)} problems from {cvdp.subset}")

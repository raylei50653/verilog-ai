import json
import pytest
from pathlib import Path
from unittest.mock import patch


class MockDataset:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def save_to_disk(self, path):
        pass

    @classmethod
    def load_from_disk(cls, path):
        return None


def make_cvdp_row(
    pid: str,
    categories: list[str],
    prompt: str = "",
    context: dict | None = None,
    output_context: dict | None = None,
    harness: dict | None = None,
) -> dict:
    return {
        "id": pid,
        "categories": categories,
        "input": {
            "prompt": prompt,
            "context": context or {},
        },
        "output": {
            "response": "",
            "context": output_context or {},
        },
        "harness": {
            "files": harness or {},
        },
    }


class TestCVDPDataset:
    @patch("src.cvdp.loader.load_dataset")
    def test_empty_dataset(self, mock_load):
        mock_load.return_value = MockDataset([])
        from src.cvdp.loader import CVDPDataset

        cvdp = CVDPDataset(subset="nonagentic_no_commercial")
        cvdp._dataset = mock_load.return_value

        assert len(cvdp) == 0
        assert cvdp.get_by_id("nonexistent") is None
        assert cvdp.get_categories() == []

    @patch("src.cvdp.loader.load_dataset")
    def test_dataset_with_data(self, mock_load):
        rows = [
            make_cvdp_row("cvdp_counter_0001", ["cid001", "easy"], "Implement a counter"),
            make_cvdp_row("cvdp_fifo_0001", ["cid005", "medium"], "Implement a FIFO"),
            make_cvdp_row("cvdp_arbiter_0001", ["cid001", "hard"], "Implement an arbiter"),
        ]
        mock_load.return_value = MockDataset(rows)

        from src.cvdp.loader import CVDPDataset, CVDPProblem

        cvdp = CVDPDataset(subset="nonagentic_no_commercial")
        cvdp._dataset = mock_load.return_value

        assert len(cvdp) == 3

        problem = cvdp.get_by_id("cvdp_fifo_0001")
        assert isinstance(problem, CVDPProblem)
        assert problem.id == "cvdp_fifo_0001"
        assert problem.categories == ["cid005", "medium"]
        assert problem.difficulty == "medium"
        assert problem.prompt == "Implement a FIFO"
        assert cvdp.get_by_id("nonexistent") is None

        first = cvdp[0]
        assert isinstance(first, CVDPProblem)
        assert first.id == "cvdp_counter_0001"

    @patch("src.cvdp.loader.load_dataset")
    def test_filter_by_category(self, mock_load):
        rows = [
            make_cvdp_row("a", ["cid001"]),
            make_cvdp_row("b", ["cid005"]),
            make_cvdp_row("c", ["cid001"]),
        ]
        mock_load.return_value = MockDataset(rows)
        from src.cvdp.loader import CVDPDataset

        cvdp = CVDPDataset()
        cvdp._dataset = mock_load.return_value

        cid001 = cvdp.filter_by_category("cid001")
        assert len(cid001) == 2
        assert all("cid001" in p.cids for p in cid001)

    @patch("src.cvdp.loader.load_dataset")
    def test_get_categories(self, mock_load):
        rows = [
            make_cvdp_row("a", ["cid001", "easy"]),
            make_cvdp_row("b", ["cid001", "medium"]),
            make_cvdp_row("c", ["cid005", "hard"]),
        ]
        mock_load.return_value = MockDataset(rows)
        from src.cvdp.loader import CVDPDataset

        cvdp = CVDPDataset()
        cvdp._dataset = mock_load.return_value

        cats = cvdp.get_categories()
        assert "cid001" in cats
        assert "easy" in cats
        assert "hard" in cats

    @patch("src.cvdp.loader.load_dataset")
    def test_export_index(self, mock_load, tmp_path):
        rows = [
            make_cvdp_row("cvdp_counter_0001", ["cid001"], "test" * 50),
        ]
        mock_load.return_value = MockDataset(rows)
        from src.cvdp.loader import CVDPDataset

        cvdp = CVDPDataset()
        cvdp._dataset = mock_load.return_value

        output_path = tmp_path / "index.json"
        index = cvdp.export_index(str(output_path))
        assert len(index) == 1
        assert index[0]["id"] == "cvdp_counter_0001"
        assert "prompt" in index[0]
        assert output_path.exists()

    def test_invalid_subset(self):
        from src.cvdp.loader import CVDPDataset
        with pytest.raises(ValueError, match="Unknown subset"):
            CVDPDataset(subset="nonexistent_subset")


class TestCVDPProblem:
    def test_problem_properties(self):
        row = make_cvdp_row(
            "cvdp_async_fifo_0001",
            ["cid008", "hard"],
            prompt="Design an async FIFO",
            context={"docs/fifo.md": "# FIFO spec"},
            output_context={"rtl/async_fifo.sv": ""},
            harness={
                "docker-compose.yml": "services:",
                "src/test_fifo.py": "import cocotb",
            },
        )

        from src.cvdp.loader import CVDPProblem

        p = CVDPProblem(row)
        assert p.id == "cvdp_async_fifo_0001"
        assert p.difficulty == "hard"
        assert p.cids == ["cid008"]
        assert p.prompt == "Design an async FIFO"
        assert p.context == {"docs/fifo.md": "# FIFO spec"}
        assert p.output_files == ["rtl/async_fifo.sv"]
        assert "src/test_fifo.py" in p.harness_files
        assert p.has_testbench()

    def test_no_testbench(self):
        row = make_cvdp_row("test", ["cid001"], "counter")
        from src.cvdp.loader import CVDPProblem

        p = CVDPProblem(row)
        assert not p.has_testbench()
        assert p.get_testbench_files() == {}

# CVDP Integration Plan

## 1. Dataset Acquisition

### Source
- HuggingFace: `nvidia/cvdp-benchmark-dataset`
- GitHub: `NVlabs/cvdp_benchmark` (scoring infrastructure)

### Subsets Used

| Subset | Rows | Description | Priority |
|---|---|---|---|
| `cvdp_nonagentic_code_generation_no_commercial` | 302 | Pure RTL generation, open-source tools | **Primary** |
| `cvdp_nonagentic_code_generation_commercial` | 187 | RTL generation, needs Cadence | Phase 2 |
| `cvdp_agentic_code_generation_no_commercial` | 92 | Agent-based generation | Phase 3 |
| `cvdp_agentic_code_generation_commercial` | 68 | Agent-based, needs Cadence | Phase 3 |

**Initial focus**: `nonagentic_no_commercial` (302 problems) — covers RTL generation with open-source verification.

### Dataset Format (per row)

```json
{
  "id": "cvdp_nonagentic_fixed_arbiter_0001",
  "categories": ["cid001", "easy"],
  "input": {
    "prompt": "Implement a fixed-priority arbiter with 4 requestors...",
    "context": {
      "docs/spec.md": "# Arbiter Specification..."
    }
  },
  "output": {
    "response": "",
    "context": {
      "rtl/fixed_arbiter.sv": ""
    }
  },
  "harness": {
    "files": {
      "docker-compose.yml": "services:\n  direct:\n    image: __OSS_SIM_IMAGE__...",
      "src/test_arbiter.py": "import cocotb\n...",
      "src/test_runner.py": "...",
      "src/harness_library.py": "..."
    }
  }
}
```

- `input.prompt`: the problem specification text
- `input.context`: supporting files (spec docs, RTL stubs) mapped by file path
- `output.response`: reference solution (empty — withheld from public release)
- `output.context`: expected output file paths (values empty — solutions withheld)
- `harness.files`: test infrastructure (docker-compose, cocotb testbench, runner)

## 2. CVDP Problem Categories

CVDP uses 13 category IDs. Key ones for initial focus:

| CID | Name | Example | Difficulty |
|---|---|---|---|
| cid001 | Combinational Logic | Priority arbiter, decoder | Easy |
| cid002 | Sequential Logic | Counters, shift registers | Easy |
| cid003 | Finite State Machines | Protocol FSMs | Medium |
| cid004 | Arithmetic | ALU, multipliers | Medium |
| cid005 | Memory | FIFO, RAM controllers | Medium |
| cid006 | Protocol Interfaces | SPI, I2C, UART | Hard |
| cid007 | Bus Interfaces | AXI-lite, APB, Wishbone | Hard |
| cid008 | Clock Domain Crossing | Async FIFO, synchronizers | Hard |
| cid009 | Pipelined Designs | Pipeline stages | Medium |
| cid010 | Error Correction | Hamming, CRC | Medium |
| cid011 | DSP | Filters, FFT blocks | Hard |
| cid012 | CPU IP | Simple CPU components | Hard |
| cid013 | Mixed | Multi-component designs | Medium |

## 3. Scoring Integration

### 3.1 Pass@k Calculation

CVDP uses pass@k as the primary metric:

```
pass@k = E[1 - C(n-c, k) / C(n, k)]
```

Where:
- `n` = total samples generated per problem
- `c` = number of correct samples
- `k` = the k in pass@k

Implementation: Adopt the unbiased estimator from the CVDP benchmark repo (`run_samples.py`).

### 3.2 Pass Criterion

For **code generation** tasks:
1. `iverilog` compilation succeeds (no syntax errors).
2. `cocotb` simulation passes all test cases.
3. Both conditions must be met — partial credit philosophy TBD.

For **code comprehension** tasks (future):
- BLEU/ROUGE for reference answer matching.
- LLM-based subjective scoring (requires API key).

### 3.3 Scoring Adapter

```python
# Conceptual interface
class CVDPScorer:
    def score_single(self, problem_id: str, generated_code: str, testbench: str) -> TrialResult
    def compute_pass_at_k(self, results: List[TrialResult], k: int) -> float
    def generate_report(self, results: List[TrialResult]) -> dict
```

Scoring pipeline:
1. Write generated RTL to temp file.
2. Run through Syntax Agent (iverilog).
3. If syntax pass → run Simulation Agent (cocotb + testbench from CVDP).
4. Return pass/fail + details.

Note: CVDP testbenches are embedded in the dataset harness field (docker-compose + cocotb config). The adapter must extract and execute them, without requiring the full CVDP Docker orchestration.

## 4. Testbench Adapter

CVDP provides testbenches via cocotb (Python) or SystemVerilog. The adapter must handle both:

### Type A: cocotb Testbench
```yaml
# From CVDP harness:
docker-compose.yml:
  services:
    direct:
      image: __OSS_SIM_IMAGE__
      command: pytest -s --log-cli-level=INFO /src/test_runner.py -v
```

The adapter must:
1. Extract cocotb test file from dataset.
2. Place RTL and testbench in correct directory structure.
3. Run in CVDP's OSS_SIM_IMAGE Docker container.
4. Parse pytest output for pass/fail.

### Type B: SystemVerilog Testbench
```verilog
// Direct SV testbench
module tb_top;
  // ... test logic
endmodule
```

The adapter must:
1. Compile with iverilog (RTL + testbench).
2. Run `vvp` simulation.
3. Parse `$display`/`$fatal` output for pass/fail.

## 5. Execution Environment

### Docker Simulation Image

Reuse CVDP's `Dockerfile.sim`:
```dockerfile
# Tools included:
# - Icarus Verilog v13_0 (iverilog, vvp)
# - Verilator v5.038
# - Yosys yosys-0.40
# - cocotb 2.0.1
# - pytest 8.3.2
```

Build command:
```bash
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
```

### Running in Docker

Trial execution wrapper:
```python
def run_in_sim_docker(rtl_files: dict, test_files: dict, work_dir: str) -> dict:
    """Mount RTL and testbench, run simulation, return results."""
    # 1. Create temp directory with RTL and testbench
    # 2. Run docker with volume mount
    # 3. Execute cocotb/iverilog inside container
    # 4. Parse and return output
```

## 6. Dataset Preprocessing

### Step 1: Download
```bash
# Using huggingface datasets library via uv
uv run python -c "
from datasets import load_dataset
ds = load_dataset('nvidia/cvdp-benchmark-dataset',
                   'cvdp_nonagentic_code_generation_no_commercial')
ds.save_to_disk('data/cvdp_dataset/nonagentic_no_commercial')
"
```

### Step 2: Index
Generate `benchmark_index.json`:
```json
[
  {
    "id": "cvdp_nonagentic_fixed_arbiter_0001",
    "categories": ["cid001", "easy"],
    "difficulty": "easy",
    "has_testbench": true,
    "output_files": ["rtl/fixed_arbiter.sv"]
  }
]
```

### Step 3: Extract Testbenches
Parse each problem's `harness.files` field to extract testbench files (`src/test_*.py`, `src/harness_library.py`, `src/test_runner.py`), and write them alongside the problem data for fast access.

## 7. Baseline Evaluation Plan

### 7.1 Models to Benchmark

| Model | Source | Format | Notes |
|---|---|---|---|
| DeepSeek-Coder-V2-Instruct | HuggingFace | GGUF Q4_K_M | Primary target |
| Qwen3-Coder-30B | HuggingFace | GGUF Q4_K_M | Comparison point |
| CodeLlama-34B-Instruct | HuggingFace | GGUF Q4_K_M | Open-source baseline |

All models served via `llama-server` with OpenAI-compatible API.

### 7.2 Baseline Configuration

- **Mode**: Non-agentic (direct API call per problem).
- **Temperature**: 0.2 (low for reproducibility).
- **Samples per problem**: 10 (for pass@1 and pass@5).
- **Max tokens**: 4096 per generation.
- **Context size**: 8192 (llama-server `--ctx-size`).
- **Prompt**: Systematic prompt = system_message + problem prompt from CVDP.

### 7.3 Metrics to Track

| Metric | Tool | Target |
|---|---|---|
| Pass@1 | CVDP scoring | Baseline > 20% |
| Pass@5 | CVDP scoring | Baseline > 33% |
| Syntax pass rate | iverilog | > 70% of raw outputs |
| Simulation pass rate | cocotb | > 40% of syntax-pass outputs |
| Avg tokens per generation | API response | < 2000 |
| Avg wall time per problem | Timer | < 60s |
| llama-server throughput | llama.cpp metrics | > 10 tok/s |

## 8. Customization Points

### Adding a New Problem

1. Create spec file in `data/specs/{problem_id}.md`.
2. Add testbench to `data/testbenches/{problem_id}/`.
3. Add entry to `config/benchmark_index.json`.
4. Define tunable parameters in `config/optuna_params.json`.

### Adding a New Sub-Agent

1. Implement in `src/agents/{agent_name}.py`.
2. The agent must implement:
   ```python
   class SubAgent(ABC):
       @abstractmethod
       def execute(self, rtl: str, testbench: str, config: dict) -> AgentResult: ...
   ```
3. Register in `src/agents/__init__.py`.

### Swapping LLM Backend

1. Implement the `LLMBackend` interface.
2. Set `LLM_BACKEND` in config/env.
3. Main model references backend via dependency injection.

## 9. CVDP Leaderboard (Future)

When ready to submit to CVDP leaderboard:

1. Wrap generator as CVDP agent (implement `agent.py` per CVDP agentic spec).
2. The agent internally calls the VeriGen pipeline.
3. Output format matches CVDP's expected patch format.
4. Submit through CVDP's `run_samples.py` with `-g` flag.

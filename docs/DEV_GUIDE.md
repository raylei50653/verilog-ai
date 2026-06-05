# VeriGen Developer Guide

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package and project manager)
- [llama-server](https://github.com/ggerganov/llama.cpp) (local LLM inference server)
- Docker CE (with user in docker group)
- At least 20GB free disk space (for Docker image + dataset)

## Setup

### 1. Clone and Environment

```bash
cd verilog-ai
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Build Docker Simulation Image

```bash
# Build from CVDP's Dockerfile
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .

# Verify tools are available
docker run --rm nvidia/cvdp-sim:v1.0.0 iverilog -V
docker run --rm nvidia/cvdp-sim:v1.0.0 yosys -V
docker run --rm nvidia/cvdp-sim:v1.0.0 verilator --version
```

### 3. Download CVDP Dataset

```bash
python -m src.cvdp.loader --download --subset nonagentic_no_commercial
```

### 4. Initialize MCP Database

```bash
python -m src.mcp.server --init
```

This creates:
- `data/trials/trials.db` — SQLite database for trial history
- `config/constraints.json` — Starter constraint templates
- `config/interfaces.json` — Starter interface templates

### 5. Start llama-server

```bash
# Download a code-capable model (e.g., DeepSeek-Coder-V2 or Qwen3-Coder)
# See https://huggingface.co/models for GGUF quantized versions

# Start the server (OpenAI-compatible API on localhost:8080)
llama-server \
  --model ./models/deepseek-coder-v2-instruct.Q4_K_M.gguf \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --host 0.0.0.0 \
  --port 8080
```

### 6. Configure LLM Backend

Copy and edit the environment file:
```bash
cp .env.example .env
# Edit: set LLAMA_SERVER_BASE_URL and model name
```

## Development Workflow

### Run a Single Trial (Core Loop)

For debugging the full pipeline end-to-end:
```bash
python -m src.cli run \
  --spec "Implement a 4-bit synchronous counter with enable and reset" \
  --params '{"width": 4}' \
  --verbose
```

This will:
1. Generate RTL via Main Model
2. Run Syntax Agent
3. Run Simulation Agent (creates a minimal testbench if none provided)
4. Run PPA Agent
5. Print all intermediate results

### Run on a CVDP Problem

```bash
python -m src.cli run \
  --problem-id cvdp_nonagentic_fixed_arbiter_0001 \
  --params '{"num_requestors": 4}' \
  --verbose
```

### Run Baseline Benchmark

```bash
python -m src.cli benchmark \
  --dataset nonagentic_no_commercial \
  --model deepseek-coder-v2 \
  --samples 10 \
  --pass-k 1,5 \
  --output results/baseline.json
```

### Run Optuna Optimization

```bash
python -m src.cli optimize \
  --problem-id cvdp_nonagentic_pipeline_mult_0001 \
  --trials 50 \
  --objective area_and_delay \
  --output results/optuna/
```

## Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific component tests
pytest tests/agents/ -v
pytest tests/main_model/ -v
pytest tests/mcp/ -v
```

### Integration Tests

```bash
# Requires Docker simulation image
pytest tests/integration/ -v --docker

# Test the full pipeline on a single problem
pytest tests/integration/test_pipeline.py -v
```

### CVDP Scoring Tests

```bash
# Verify scoring matches CVDP reference implementation
pytest tests/cvdp/test_scoring.py -v
```

## Configuration Reference

### constraints.json

```json
{
  "default": {
    "clock_freq_mhz": 100,
    "setup_time_ns": 1.0,
    "hold_time_ns": 0.5
  },
  "async_fifo": {
    "write_clk_mhz": 200,
    "read_clk_mhz": 100,
    "depth": [4, 8, 16, 32, 64, 128],
    "data_width": [8, 16, 32, 64]
  },
  "axi4_lite": {
    "data_width": 32,
    "addr_width": 32,
    "max_burst_length": 1
  }
}
```

### interfaces.json

```json
{
  "axi4_lite": {
    "signals": {
      "awvalid": "output logic",
      "awready": "input logic",
      "awaddr": "output logic [ADDR_WIDTH-1:0]",
      "wvalid": "output logic",
      "wready": "input logic",
      "wdata": "output logic [DATA_WIDTH-1:0]",
      "wstrb": "output logic [DATA_WIDTH/8-1:0]",
      "bvalid": "input logic",
      "bready": "output logic",
      "arvalid": "output logic",
      "arready": "input logic",
      "araddr": "output logic [ADDR_WIDTH-1:0]",
      "rvalid": "input logic",
      "rready": "output logic",
      "rdata": "input logic [DATA_WIDTH-1:0]"
    },
    "handshake": "valid-ready",
    "channels": ["write_address", "write_data", "write_response", "read_address", "read_data"]
  }
}
```

### Optuna Parameter Config

```json
{
  "cvdp_nonagentic_pipeline_mult_0001": {
    "parameters": {
      "pipeline_stages": {"type": "int", "low": 1, "high": 8},
      "data_width": {"type": "categorical", "choices": [8, 16, 32, 64]},
      "use_dsp": {"type": "categorical", "choices": [true, false]}
    },
    "objective": "minimize_area",
    "constraints": {
      "max_delay_ns": 10.0
    }
  }
}
```

### .env.example

```bash
# LLM Backend (llama-server with OpenAI-compatible API)
LLM_BACKEND=llama_server
LLAMA_SERVER_BASE_URL=http://localhost:8080/v1
LLAMA_SERVER_API_KEY=not-needed

# Model Settings
DEFAULT_MODEL=deepseek-coder-v2
MAX_RETRIES=3
TEMPERATURE=0.2
MAX_TOKENS=4096

# Docker
SIM_IMAGE=nvidia/cvdp-sim:v1.0.0

# Paths
DATA_DIR=./data
CVDP_DATASET_DIR=./data/cvdp_dataset
TRIAL_DB_PATH=./data/trials/trials.db
CONSTRAINTS_PATH=./config/constraints.json
INTERFACES_PATH=./config/interfaces.json

# Optuna
OPTUNA_STORAGE=sqlite:///data/optuna/optuna.db
OPTUNA_N_TRIALS=100
OPTUNA_SAMPLER=TPE
```

## Key Interfaces

### SubAgent (Abstract Base)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AgentResult:
    pass_: bool
    errors: list[dict]      # [{line, message, ...}]
    warnings: list[dict]
    metrics: dict            # agent-specific metrics
    raw_output: str
    duration_ms: float

class SubAgent(ABC):
    @abstractmethod
    def execute(self, rtl_files: dict[str, str],
                testbench_files: dict[str, str] | None = None,
                config: dict | None = None) -> AgentResult:
        ...
```

### MCP Server

```python
class MCPServer:
    def get_constraints(self, module_type: str) -> dict: ...
    def get_interface(self, protocol: str) -> dict: ...
    def get_history(self, problem_id: str, limit: int = 5) -> list[dict]: ...
    def write_trial(self, trial: dict) -> str: ...
    def get_successful_trials(self, problem_id: str) -> list[dict]: ...
```

### LLM Backend

```python
class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2,
                 max_tokens: int = 4096) -> str:
        ...

class LlamaServerBackend(LLMBackend):
    """OpenAI-compatible client pointing to llama-server."""
    def __init__(self, base_url: str = "http://localhost:8080/v1"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
```

### CVDP Loader

```python
class CVDPDataset:
    def __init__(self, subset: str): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...
    def get_by_id(self, problem_id: str) -> dict: ...
    def get_categories(self) -> list[str]: ...
    def filter_by_category(self, cid: str) -> list[dict]: ...
    def filter_by_difficulty(self, level: str) -> list[dict]: ...
```

## Phase Plan

### Phase 1: Foundation (Weeks 1-2)
- [x] Project scaffolding (directories, configs, pyproject.toml)
- [x] Docker simulation image build
- [x] CVDP dataset download and indexing
- [x] Syntax Agent (iverilog wrapper)
- [x] MCP Server (SQLite + JSON)
- [x] LLM backend abstraction (llama-server via OpenAI-compatible client)
- [x] CLI skeleton
- [x] Unit tests for all components

### Phase 2: Core Pipeline (Weeks 3-4)
- [x] Main Model with basic prompt engineering
- [x] Simulation Agent (cocotb + Verilator)
- [x] PPA Agent (Yosys synthesis)
- [x] Single-trial end-to-end flow
- [x] Retry logic (compile-fix loop)
- [x] CVDP dataset integration (load problem, extract testbench)
- [x] Integration tests

### Phase 3: Optimization (Weeks 5-6)
- [x] Optuna outer-loop integration
- [x] Multi-problem benchmark runner
- [x] Pass@k scoring adapter
- [x] Baseline evaluation (llama-server with DeepSeek-Coder-V2, Qwen3-Coder)
- [x] Optimization on parameterized problems
- [x] Results dashboard / reporting

### Phase 4: Advanced (Weeks 7-8)
- [x] Multi-model comparison (different GGUF models)
- [x] Advanced prompt strategies (few-shot, chain-of-thought)
- [x] RTL reuse across problems (spec-to-existing-module matching)
- [x] Full CVDP agentic mode adapter (leaderboard submission)


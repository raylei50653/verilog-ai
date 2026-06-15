# VeriGen

LLM-based Verilog RTL generator powered by llama-server, with multi-agent verification, Yosys synthesis, and optional Vivado analysis.

## Approach

```
Spec → LLM (generate RTL) → Sub-Agent Pipeline → Result
         ↕ query/write          ├─ Syntax check (iverilog)
      MCP Server                ├─ Simulation (Verilator + cocotb)
  (constraints, interfaces,     ├─ PPA analysis (Yosys + abc)
   trial history in SQLite)     └─ Vivado analysis (optional)
```

Evaluated on the [CVDP benchmark](https://github.com/NVlabs/cvdp_benchmark) (783 real-world Verilog design problems).

## Stack

| Category | Tools |
|---|---|
| LLM inference | llama-server (OpenAI-compatible API, GGUF models) |
| Package manager | uv |
| CLI | Click |
| TUI | Textual + Rich |
| Data validation | Pydantic |
| Verification (Docker) | Icarus Verilog, Verilator, cocotb |
| Synthesis (Docker) | Yosys + abc |
| Synthesis (optional) | Vivado |
| Dataset | HuggingFace `datasets` (CVDP benchmark) |
| Storage | SQLite3 |
| Testing | pytest |
| Build | Hatchling |

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture, design decisions, component specs |
| [CVDP_INTEGRATION.md](docs/CVDP_INTEGRATION.md) | CVDP dataset usage, scoring, testbench adaptation |
| [DEV_GUIDE.md](docs/DEV_GUIDE.md) | Setup, development workflow, CLI reference, phase plan |

## Quick Start

```bash
git clone git@github.com:raylei50653/verilog-ai.git && cd verilog-ai
uv venv && uv sync
docker build --network=host -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
cp .env.example .env   # edit to match your setup
```

Launch the TUI:

```bash
uv run verigen tui
```

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker CE](https://docs.docker.com/engine/install/) | User in `docker` group |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Provides `llama-server`; needs `build-essential cmake git` if compiling from source |
| GGUF model file | Code-capable model (e.g. Qwen3-Coder, DeepSeek-Coder-V2) |
| ~20 GB free disk | Docker image ~5 GB + dataset ~2 GB |

## Setup

### 1. Install Python dependencies

```bash
uv venv && uv sync
```

### 2. Build the Docker simulation image

```bash
docker build --network=host -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
```

Verify tools:

```bash
docker run --rm nvidia/cvdp-sim:v1.0.0 iverilog -V
docker run --rm nvidia/cvdp-sim:v1.0.0 yosys -V
docker run --rm nvidia/cvdp-sim:v1.0.0 verilator --version
```

### 3. Configure environment

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server API endpoint |
| `DEFAULT_MODEL` | `Qwen3.5-9B-UD-Q4_K_XL` | Model name reported to server |
| `VIVADO_BIN` | (auto-detect) | Path to vivado binary (optional) |
| `VIVADO_PART` | `xc7a35tcpg236-1` | Xilinx part number |
| `VIVADO_PROJECT_DIR` | (system temp) | Vivado project output directory |

### 4. Start the LLM server

```bash
llama-server \
  --model ./models/model.gguf \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --host 0.0.0.0 \
  --port 8080
```

### 5. Download the CVDP benchmark dataset

```bash
uv run verigen download --subset nonagentic_no_commercial
```

### 6. Verify installation

```bash
uv run verigen info
uv run verigen run --spec "Implement a 4-bit synchronous counter" --verbose
```

### 7. (Optional) Dev dependencies

```bash
uv sync --dev
pytest tests/ -v
```

### 8. (Optional) Vivado support

```bash
uv run verigen vivado-detect
```

## Usage Examples

```bash
uv run verigen run --spec "cvdp:cvdp_nonagentic_fixed_arbiter_0001" --vivado -v
uv run verigen benchmark --dataset nonagentic_no_commercial --samples 10
uv run verigen vivado-analyze --trial-id <trial_id>
uv run verigen tui
```

## Status

**Complete**. All phases have been fully implemented, optimized, and verified.

* **Phase 1 (Foundation)**: Project scaffolding, database, and backend abstraction.
* **Phase 2 (Core Pipeline)**: Unified compilation-simulation retry loops.
* **Phase 3 (Advanced)**: Spec-to-module matching for RTL reuse, chain-of-thought support, and custom model comparative benchmarking.
* **Phase 4 (Vivado)**: Vivado synthesis analysis integration for pass/fail validation.

## License

MIT

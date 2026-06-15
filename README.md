# VeriGen

LLM-based Verilog RTL generator powered by llama-server, with multi-agent verification and Vivado synthesis analysis.

## Approach

```
Spec → Main Model (generate RTL) → Sub-Agent Pipeline (verify) → Vivado (analyze)
         ↕ query/write
      MCP Server (constraints, interfaces, history)
```

Evaluated on the [CVDP benchmark](https://github.com/NVlabs/cvdp_benchmark) (783 real-world Verilog design problems).

## Stack

- **LLM**: llama-server (OpenAI-compatible API, GGUF models)
- **Package manager**: uv
- **Verification**: iverilog, cocotb + Verilator
- **Synthesis analysis**: Vivado (optional)
- **Simulation environment**: Docker (CVDP OSS_SIM_IMAGE)

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture, design decisions, component specs |
| [CVDP_INTEGRATION.md](docs/CVDP_INTEGRATION.md) | CVDP dataset usage, scoring, testbench adaptation |
| [DEV_GUIDE.md](docs/DEV_GUIDE.md) | Setup, development workflow, CLI reference, phase plan |

## Initialization

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| [uv](https://docs.astral.sh/uv/) | `pip install uv` |
| Docker CE | User in `docker` group |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Provides `llama-server` |
| GGUF model file | Code-capable model (e.g. DeepSeek-Coder-V2, Qwen3-Coder) |
| ~20 GB free disk | Docker image ~5 GB + dataset ~2 GB |

### Quick Start

```bash
git clone git@github.com:raylei50653/verilog-ai.git && \
cd verilog-ai && \
uv venv && uv sync && \
docker build --network=host -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
```

Launch the GUI:

```bash
uv run verigen tui
```

### 0. Install system packages

```bash
# Docker CE
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build tools for llama.cpp (if compiling from source)
sudo apt install -y build-essential cmake git

# EDA tools are bundled in the Docker simulation image — no need to install locally

# Reload group membership
newgrp docker
```

### 1. Install Python dependencies

```bash
uv venv && uv sync
```

### 2. Build the Docker simulation image

```bash
docker build --network=host -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .

# Verify tools
docker run --rm nvidia/cvdp-sim:v1.0.0 iverilog -V
docker run --rm nvidia/cvdp-sim:v1.0.0 yosys -V
docker run --rm nvidia/cvdp-sim:v1.0.0 verilator --version
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env to match your setup (model name, paths, etc.)
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server API endpoint |
| `DEFAULT_MODEL` | `Qwen3.5-9B-UD-Q4_K_XL` | Model name reported to server |
| `VIVADO_BIN` | (auto-detect) | Path to vivado binary |
| `VIVADO_PART` | `xc7a35tcpg236-1` | Xilinx part number |
| `VIVADO_PROJECT_DIR` | (system temp) | Vivado project output directory |

### 4. Start the LLM server

Download a GGUF model and start llama-server:

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
# Check dataset info
uv run verigen info

# Run a quick single trial
uv run verigen run --spec "Implement a 4-bit synchronous counter" --verbose

# Or use a CVDP problem
uv run verigen run --spec "cvdp:cvdp_nonagentic_fixed_arbiter_0001" --params '{"num_requestors": 4}' --verbose
```

### 7. (Optional) Install dev dependencies

```bash
uv sync --dev
pytest tests/ -v
```

### 8. (Optional) Vivado support (WSL2 / native Linux)

```bash
uv run verigen vivado-detect
# Set VIVADO_BIN and VIVADO_PART in .env accordingly
```

## Usage Examples

```bash
# Run a single trial with Vivado synthesis analysis
uv run verigen run --spec "cvdp:cvdp_nonagentic_fixed_arbiter_0001" --vivado -v

# Baseline evaluation with pass@k scoring
uv run verigen benchmark --dataset nonagentic_no_commercial --samples 10

# Standalone Vivado analysis on a trial
uv run verigen vivado-analyze --trial-id <trial_id>

# Launch TUI
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

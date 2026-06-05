# VeriGen

LLM-based Verilog RTL generator powered by llama-server, with multi-agent verification and Optuna-driven design space exploration.

## Approach

```
Spec → Main Model (generate RTL) → Sub-Agent Pipeline (verify) → Optuna (optimize params)
         ↕ query/write
      MCP Server (constraints, interfaces, history)
```

Evaluated on the [CVDP benchmark](https://github.com/NVlabs/cvdp_benchmark) (783 real-world Verilog design problems).

## Stack

- **LLM**: llama-server (OpenAI-compatible API, GGUF models)
- **Package manager**: uv
- **Verification**: iverilog, cocotb + Verilator, Yosys
- **Optimization**: Optuna TPE
- **Simulation environment**: Docker (CVDP OSS_SIM_IMAGE)

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture, design decisions, component specs |
| [CVDP_INTEGRATION.md](docs/CVDP_INTEGRATION.md) | CVDP dataset usage, scoring, testbench adaptation |
| [DEV_GUIDE.md](docs/DEV_GUIDE.md) | Setup, development workflow, CLI reference, phase plan |

## Quick Start

```bash
# Install dependencies
uv venv && source .venv/bin/activate && uv pip install -e .

# Build simulation environment
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .

# Start llama-server (download a GGUF model first)
llama-server --model ./models/model.gguf --ctx-size 8192 --n-gpu-layers 99 --port 8080

# Download benchmark dataset
python -m src.cvdp.loader --download --subset nonagentic_no_commercial

# Run baseline evaluation
python -m src.cli benchmark --dataset nonagentic_no_commercial --samples 10

# Run Optuna design space parameter optimization for a single problem
python -m src.cli optimize --problem-id cvdp_nonagentic_pipeline_mult_0001 --trials 50 --objective area --reuse-rtl

# Run Optuna optimization across all problems in a dataset
python -m src.cli optimize-all --dataset nonagentic_no_commercial --trials-per-problem 10 --objective area
```

## Status

**Complete**. All phases (Phase 1 to Phase 4) have been fully implemented, optimized, and verified.
* **Phase 1 (Foundation)**: Project scaffolding, database, and backend abstraction.
* **Phase 2 (Core Pipeline)**: Unified compilation-simulation retry loops and PPA metrics parsing.
* **Phase 3 (Optimization)**: Optuna integration with declarative parameter configs and objective constraints.
* **Phase 4 (Advanced)**: Spec-to-module matching for RTL reuse, chain-of-thought support, and custom model comparative benchmarking.

## License

MIT

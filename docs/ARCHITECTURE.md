# VeriGen Architecture

## 1. Overview

VeriGen is a multi-agent LLM-based Verilog RTL generator with Optuna-driven design space exploration, evaluated on the CVDP benchmark.

```
                   ┌──────────────────────────┐
                   │        Optuna            │
                   │  (TPE hyperparameter     │
                   │   search outer-loop)     │
                   └────────────┬─────────────┘
                                │ suggest params
                                ▼
┌─────────────────────────────────────────────────────┐
│                   Main Model                        │
│  - Parse specification + parameters                 │
│  - Query MCP for constraints/interfaces/history     │
│  - Generate RTL code                                │
│  - Receive Sub-Agent feedback → decide retry/done   │
└──┬────────────────────────────────────────────┬─────┘
   │ query/write                                │ submit RTL
   ▼                                            ▼
┌──────────────┐              ┌──────────────────────────────┐
│  MCP Server  │              │    Sub-Agent Pipeline        │
│  ─────────── │              │                              │
│  constraints │              │  ① Syntax Agent              │
│  interfaces  │              │     iverilog compile          │
│  history     │              │     → pass/fail + errors      │
│              │              │                              │
└──────────────┘              │  ② Simulation Agent          │
                              │     cocotb + Verilator        │
                              │     → pass/fail + failures    │
                              │                              │
                              │  ③ PPA Agent                 │
                              │     Yosys synthesis           │
                              │     → area/delay/power        │
                              └──────────────────────────────┘
```

## 2. Design Decisions

### 2.1 CVDP Integration: Mode B (Standalone Framework)

**Decision**: Use CVDP dataset as input source and CVDP scoring as evaluation, but keep the generator pipeline fully independent.

**Rationale**:
- Optuna outer-loop requires free control over iteration cycles — CVDP's agentic Docker framework would constrain this.
- The pipeline (spec → generate → verify → decide) maps naturally to a standalone architecture.
- A thin adapter layer can later bridge to CVDP agentic mode for leaderboard submission.

**CVDP dataset usage**:
- Source: HuggingFace `nvidia/cvdp-benchmark-dataset`
- Subset: Primarily `cvdp_nonagentic_code_generation_no_commercial` (302 problems)
- Scoring reuse: Pass@k calculation logic from `cvdp_benchmark/src/`

### 2.2 MCP: Structured JSON/SQLite, No RAG

**Decision**: Use static JSON for design rules/timing constraints and SQLite for trial history. No vector retrieval.

**Rationale**:
- Timing constraints and interface specifications are exact knowledge — fuzzy matching introduces error.
- Trial history is per-problem structured data, not free-text retrieval.
- RAG (DeepCircuitX) reserved for Phase 3.

### 2.3 Sub-Agent: Pure Tool Invocation, No LLM

**Decision**: Each Sub-Agent is a deterministic tool that invokes an EDA binary, parses output, and returns structured JSON. No LLM in the Sub-Agent layer.

**Rationale**:
- Clean separation of concerns: verification is deterministic, diagnosis is the Main Model's job.
- Avoids compounding LLM errors (wrong RTL → wrong diagnosis → wrong fix).
- Sub-Agents are fast, cheap, and reproducible.

### 2.4 Optuna: Outer-Loop Design Parameter Search

**Decision**: Optuna TPE sampler explores design parameters (bus width, pipeline depth, etc.), treating the RTL generator as a black-box function.

**Rationale**:
- Design space exploration is fundamentally a hyperparameter optimization problem.
- PPA metrics from synthesis are the natural objective function.
- TPE handles mixed integer/continuous parameters well.

## 3. Component Specifications

### 3.1 Main Model

- **Role**: Parse specification, integrate context, generate RTL, decide retry/done.
- **Input**: Natural language or structured specification + Optuna trial parameters.
- **Context sources**: MCP constraints, MCP interfaces, MCP trial history.
- **Output**: Verilog/SystemVerilog RTL code.
- **Retry logic**: Receives structured feedback from Sub-Agent pipeline; decides whether to regenerate with error context.
- **LLM backend**: Configurable (llama-server via OpenAI-compatible API, or any OpenAI-compatible endpoint).

### 3.2 MCP Server

- **Role**: Centralized shared memory for the generator system.
- **Storage**:
  - `constraints.json`: Static timing constraints keyed by module type.
  - `interfaces.json`: Interface specifications (AXI, APB, AHB, custom) keyed by protocol name.
  - `trials.db` (SQLite): Per-trial records.
  - `benchmark_index.json`: CVDP problem metadata index.
- **API**: Simple query interface (not MCP protocol — naming is conceptual):
  - `get_constraints(module_type) → dict`
  - `get_interface(protocol_name) → dict`
  - `get_history(problem_id) → List[TrialRecord]`
  - `write_trial(trial) → trial_id`
- **History record schema**:
  ```json
  {
    "trial_id": "uuid",
    "problem_id": "cvdp_...",
    "params": {"width": 32, "pipeline_stages": 3},
    "spec_prompt": "...",
    "generated_code": "...",
    "syntax_pass": true,
    "simulation_pass": false,
    "simulation_failures": ["signal_x_mismatch_at_t=100ns"],
    "ppa_score": {"area": 1234, "area_unit": "cells", "num_modules": 1, "num_wires": 5, "num_cells": 1234},
    "pass": false,
    "retry_count": 1,
    "duration_ms": 1500.0,
    "diagnosis_report": "...",
    "timestamp": "2026-..."
  }
  ```

### 3.3 Syntax Agent

- **Role**: Compile-check generated Verilog, return errors with location.
- **Tool**: `iverilog` (Icarus Verilog).
- **Input**: Verilog/SystemVerilog source file(s).
- **Output**:
  ```json
  {
    "pass": false,
    "errors": [
      {"line": 42, "column": 5, "message": "syntax error: unexpected ';'"}
    ],
    "warnings": []
  }
  ```
- **Execution**: Subprocess call, parse stderr with regex.
- **Timeout**: Per-compilation configurable (default 30s).

### 3.4 Simulation Agent

- **Role**: Run functional simulation with provided testbench, report failures.
- **Tool**: `cocotb` + `Verilator` (or `Icarus` as fallback).
- **Input**: RTL source + testbench file.
- **Output**:
  ```json
  {
    "pass": false,
    "total_tests": 10,
    "passed": 7,
    "failed": 3,
    "failures": [
      {"test": "test_burst_write", "signal": "data_out", "expected": "0xDEAD", "got": "0xBEEF", "time_ns": 250}
    ],
    "coverage_summary": "line: 85%, branch: 62%"
  }
  ```
- **Execution**: Docker container with cocotb + Verilator (reuse CVDP's `OSS_SIM_IMAGE`).
- **Timeout**: Per-simulation configurable (default 300s).

### 3.5 PPA Agent

- **Role**: Synthesize RTL with Yosys, extract PPA metrics.
- **Tool**: `Yosys` synthesis + `abc` for mapping.
- **Input**: RTL source(s) + target technology library (default: generic).
- **Output**:
  ```json
  {
    "pass": true,
    "area": 1234,
    "area_unit": "cells",
    "delay_ns": 2.1,
    "power_uw": 450,
    "target_met": {"area": false, "delay": true},
    "raw_yosys_output": "..."
  }
  ```
- **Execution**: Subprocess call, regex parse `stat` output.
- **Note**: Power estimation is approximate with open-source flow. Accuracy improves with commercial tools (optional future path).

### 3.6 Optuna Optimizer

- **Role**: Drive design parameter exploration across multiple CVDP problems.
- **Sampler**: TPE (Tree-structured Parzen Estimator).
- **Objective**: Minimize area/delay/power subject to functional correctness constraints.
- **Parameters** (per-problem, defined in problem config):
  - Discrete: bus width, pipeline stages, FIFO depth.
  - Categorical: arbitration policy, encoding scheme.
  - Continuous: clock frequency target.
- **Study lifecycle**: Per problem or per problem category.

## 4. Data Flow

### 4.1 Single Trial Flow

```
1. Optuna.suggest() → trial_params
2. Main Model:
   a. Load spec + trial_params
   b. MCP.get_constraints(module_type)
   c. MCP.get_interfaces(protocol)
   d. MCP.get_history(problem_id)  [for in-context examples]
   e. Generate RTL code
3. Syntax Agent: compile → pass/fail
   a. If FAIL → return errors to Main Model → go to 2e (max N retries)
4. Simulation Agent: cocotb run → pass/fail
   a. If FAIL → return failures to Main Model → go to 2e (max N retries)
5. PPA Agent: yosys synth → metrics
6. MCP.write_trial(trial_record)
7. Optuna.tell(trial_params, ppa_score)
```

### 4.2 Retry Decision Logic (Main Model)

```
─ Syntax fail ∧ retry_count < MAX → "Fix these syntax errors: {errors}"
─ Simulation fail ∧ retry_count < MAX → "Fix these test failures: {failures}"
─ PPA fail (target not met) → Optuna handles via objective function
─ All pass → done
─ retry_count ≥ MAX → mark as failed, move to next trial
```

## 5. Directory Structure

```
verilog-ai/
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   ├── CVDP_INTEGRATION.md  # CVDP dataset usage and scoring
│   └── DEV_GUIDE.md         # Developer setup and workflow
├── src/
│   ├── main_model/
│   │   ├── __init__.py
│   │   ├── generator.py     # RTL generation logic
│   │   └── prompt.py        # Prompt templates
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── syntax.py        # Syntax Agent (iverilog)
│   │   ├── simulation.py    # Simulation Agent (cocotb)
│   │   └── ppa.py           # PPA Agent (yosys)
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py        # SQLite Database & Constraint management (constraints.json, interfaces.json)
│   ├── optimizer/
│   │   ├── __init__.py
│   │   └── optuna_runner.py # Optuna TPE search loop & parameter suggest/tell
│   ├── cvdp/
│   │   ├── __init__.py
│   │   ├── loader.py        # CVDP dataset loader
│   │   └── scoring.py       # Pass@k scoring & benchmark report generator
│   └── cli.py               # CLI entry point (run, benchmark, optimize, optimize-all)
├── config/
│   ├── constraints.json     # Static timing/IP constraints
│   └── interfaces.json      # Interface specifications
├── data/
│   ├── cvdp_dataset/        # Local copy of CVDP dataset
│   ├── trials/              # SQLite trial database
│   └── outputs/             # Generated RTL per trial
├── tests/
│   └── ...
├── docker/
│   └── Dockerfile.sim       # Simulation Docker image
├── requirements.txt → pyproject.toml (uv-managed)
├── pyproject.toml
```

## 6. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| LLM backend | llama-server (OpenAI-compatible API) | Local inference, no API costs, GGUF model ecosystem |
| Syntax check | Icarus Verilog (iverilog) | Fast, free, handles SV subset |
| Simulation | cocotb + Verilator | Open-source, Python testbenches |
| Synthesis/PPA | Yosys + abc | Open-source, fast, adequate for relative comparison |
| MCP storage | SQLite + JSON files | Zero-dependency, human-readable, version-controllable |
| Optimization | Optuna (TPE) | Battle-tested, handles mixed params |
| Benchmark | CVDP (NVlabs/cvdp_benchmark) | SOTA benchmark, realistic problems |
| Containerization | Docker (CVDP OSS_SIM_IMAGE) | Reproducible simulation environment |
| Package manager | uv | Fast, unified pip+venv replacement

## 7. Execution Modes

### 7.1 Single Trial (Debug)

```bash
verigen run --spec cvdp:cvdp_async_fifo_0001 --params '{"width": 8, "depth": 16}' --verbose
```

Runs one trial end-to-end, prints all intermediate results. For development and debugging.

### 7.2 Benchmark Run

```bash
verigen benchmark --dataset nonagentic_no_commercial --pass-k 1,5 --samples 10
```

Runs multiple trials per CVDP problem, computes Pass@k. For baseline evaluation.

### 7.3 Optuna Study

```bash
verigen optimize --problem-id cvdp_async_fifo_0001 --trials 100 --objective area
```

Runs Optuna TPE search over design parameters for a single problem.

### 7.4 Full Optimization Run

```bash
verigen optimize-all --dataset nonagentic_no_commercial --trials-per-problem 50
```

Runs Optuna optimization across all problems in a CVDP subset.

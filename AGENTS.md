# Repository Guidelines

## Project Structure & Module Organization

Expert Kit is primarily a Rust workspace (edition 2024) with `ek-cli` as its default member. Core crates cover shared utilities (`ek-base`), controller/worker runtime (`ek-computation`), metadata and weight serving (`ek-db`), protobuf definitions (`ek-proto`), GGML integration (`ek-ggml`), and micro-benchmarks (`ek-benchmark`). `ek-integration/expertkit_torch` and `ek-integration/expertkit_vllm` contain Python clients, model adapters, and tests. Deployment material is in `ek-solution` and `container`; tutorials and diagrams are under `doc`.

## Build, Test, and Development Commands

Run these from the repository root:

```bash
cargo build --release             # Build release binaries
cargo test --workspace            # Run Rust tests across all crates
cargo fmt --all -- --check        # Verify Rust formatting
uv sync                           # Install Python dependencies
uv run ruff check .               # Lint Python code
uv run pytest ek-integration/expertkit_torch/tests/test_torch.py -k test_torch
```

Use `uv sync --extra cu130` for the CUDA-oriented Python environment. Integration tests may require model files, services, and hardware; follow `doc/tutorial/standalone` first. Use `cargo run -p ek-cli -- --help` to inspect CLI commands.

## Coding Style & Naming Conventions

Use `cargo fmt` for Rust and four spaces for Python. Keep workspace dependencies alphabetized where practical. Follow Rust naming conventions (`snake_case` functions/modules, `PascalCase` types) and Python conventions (`snake_case` functions/modules, `PascalCase` classes). Keep protobuf changes in `ek-proto`; regenerate bindings with the integration package’s `gen_proto_py.sh`. Run formatters and linters before submitting.

## Testing Guidelines

Rust tests use Cargo’s built-in test harness; add unit tests near the implementation and run focused tests with `cargo test -p <crate> <filter>`. Python tests use pytest and are located in each integration package’s `tests/` directory. Name Python tests `test_*.py` and test functions `test_*`. Add regression coverage for behavior changes; hardware- or service-dependent tests should document prerequisites.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, often with prefixes such as `fix:` and `chore:` (for example, `fix: update print data size`). Keep commits focused and explain the user-visible or architectural impact. Pull requests should include a concise description, linked issue when applicable, validation commands and results, configuration or migration notes, and screenshots/log excerpts for CLI or documentation changes. Call out hardware, model, database, or network prerequisites explicitly.

## Security & Configuration Tips

Do not commit credentials, private model files, or generated runtime output. Configuration is loaded through `EK_CONFIG` or `--config`, with environment overrides using the `EK_` prefix; verify worker channels (`grpc`, `shm`, or `rdma`) and controller ports match across configuration, inventory, and database records.

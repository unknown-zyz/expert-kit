---
name: expert-kit-startup
description: Start and verify the Expert-Kit weight server, Controller, and Worker from this repository using run.md. Use when a user asks to bring up the local Expert-Kit service chain, diagnose startup prerequisites, check its configured ports, or prepare it for vLLM integration.
---

# Expert-Kit Startup

Use this project skill when launching the local Expert-Kit service chain or checking why it is not ready. Work from the repository root and preserve the user's existing configuration and startup files.

## Preflight

1. Enter `/home/zhangyz/expert-kit` (or the repository root discovered from the request) and activate `.venv`:

   ```bash
   cd /home/zhangyz/expert-kit
   source .venv/bin/activate
   ```

2. Check that `QWEN3_30B_A3B_ROOT` is set and is a readable model directory. Check `EK_CONFIG`; if it is unset, choose the intended config explicitly with `--config <path>` or export it. Inspect the selected YAML before starting it. The config must provide:
   - PostgreSQL DSN (`db.db_dsn`),
   - a weight cache and optional weight-server address,
   - worker device/channel and worker port,
   - Controller listen/broadcast addresses and both Controller ports.

3. Confirm PostgreSQL is reachable and migrations have been applied. For the repository's local configuration the expected database endpoint is `localhost:5432`; run the embedded migration command when needed:

   ```bash
   cargo run --release --bin ek-cli -- db migrate
   ```

   Do not start services against a missing database or silently switch to another database.

4. Check for conflicts before binding ports. The usual local ports are:

   | Component | Port | Source |
   |---|---:|---|
   | PostgreSQL | 5432 | `db.db_dsn` |
   | weight-server | 6543 | `weight-server` default |
   | Controller intra | 5001 | `controller.ports.intra` |
   | Controller inter / vLLM endpoint | 5002 | `controller.ports.inter` |
   | Worker main | 51234 | `worker.ports.main` |

   Use `ss -ltn` (or an equivalent platform tool) and report the owning process for any conflict. Do not kill an unrelated process without explicit authorization.

## Start the services

Use three independent terminals, keeping each process in the foreground so its logs remain visible. Start the weight server first, then Controller, then Worker:

```bash
# Terminal 1
cd /home/zhangyz/expert-kit
source .venv/bin/activate
cargo run --release --bin ek-cli weight-server --model "${QWEN3_30B_A3B_ROOT}"
```

```bash
# Terminal 2
cd /home/zhangyz/expert-kit
source .venv/bin/activate
cargo run --release --bin ek-cli controller
```

```bash
# Terminal 3
cd /home/zhangyz/expert-kit
source .venv/bin/activate
cargo run --release --bin ek-cli worker
```

If `EK_CONFIG` is not exported, add `--config /path/to/config.yaml` to each command. The weight-server command may use `--host` and `--port` when the defaults do not match the selected config. The `run.md` file labels terminals with the typo `ternminal`; interpret those labels as Terminal 1/2/3 and do not modify that user-owned file merely to correct the spelling.

## Readiness and operation

- Weight server: verify it reports the model roots loaded and is listening on `6543` (or the explicitly selected port).
- Controller: verify `expert kit controller started`, with intra/inter listeners on the configured `5001`/`5002` ports.
- Worker: verify registration with Controller, model/expert loading, and the configured worker port (`51234` by default).
- For vLLM, use `EK_ENABLE=1`, `EK_MODE=expert_mode`, `EK_ADDR=localhost:5002`, and an `EK_MODEL_NAME` that exactly matches `inference.model_name` and the registered model (for the local Qwen3 config, `qwen3-30b-a3b`); confirm Controller logs show incoming Expert requests and Worker logs show forwarding/activation.
- Before inference, register the model and schedule experts when the deployment tutorial requires it:

  ```bash
  cargo run --release --bin ek-cli model upsert --name qwen3-30b-a3b
  cargo run --release --bin ek-cli schedule static --inventory ./dev/local.inventory.yaml
  ```

  Verify the model name and inventory match the selected config; do not claim readiness from open ports alone.

## Stop and troubleshoot

Press `Ctrl-C` in each service terminal, stopping Worker and Controller before the weight server. Do not use broad `kill` commands. If startup fails, record the command, selected config, first error, and port/process state. Distinguish missing model/database/GPU prerequisites from configuration errors and application/plugin errors. A `ternminal` spelling issue in `run.md` is harmless; missing prerequisites are not.

# Architecture map

This file owns the component map. [docs/DECISION_ENGINE.md](docs/DECISION_ENGINE.md)
owns detailed runtime interfaces and examples; [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)
owns API stability and serialized-format rules.

| Component | Responsibility |
|---|---|
| `serpentos/runtime/models.py` | Context, decision, and outcome values |
| `serpentos/runtime/policy.py` | Pure decision interface and policy identity |
| `serpentos/runtime/engine.py` | Invoke policy, identify decision, validate, audit, and handle explicit fallback |
| `serpentos/runtime/validation.py` | Host-configured decision guardrails |
| `serpentos/runtime/audit.py` | Versioned audit records, redaction, memory and JSONL sinks |
| `serpentos/runtime/replay.py` | Reevaluate recorded contexts and report matches |
| `serpentos/runtime/comparison.py` | Compare candidate decisions and available outcomes; does not declare a universal winner |
| `serpentos/policies/` | Rule, weighted, and read-only Q-learning adapters |
| `serpentos/environments/snake/` | Convert game state into context and apply returned actions in the reference environment |
| `serpentos/core.py` | Snake rules, learning, checkpoints, and benchmark |
| `serpentos/bot.py` | Headless training lifecycle and CLI |
| `serpentos/serpentos.py`, `serpentos/theme.py` | Terminal rendering and theme roles |
| `serpentos/__main__.py` | CLI dispatch |

## Data flow and ownership

The host builds a context and calls the engine. A policy proposes a decision;
the engine applies the optional validator, emits a record to the optional sink,
and returns an accepted decision or raises. A configured fallback is separately
invoked and validated after primary rejection. The host executes actions and
provides outcome information; the runtime does not perform execution.

Policies must be pure. Engine clocks and ID factories provide record metadata
and are injectable for reproducible tests. Deterministic policy replay does not
imply that timestamps or generated identifiers are identical on every run.

## State and concurrency

Policy contexts are immutable. Audit state belongs to sinks; game training state
belongs to the learning core and its data directory. Built-in sinks synchronize
threads within one process. Custom policies, validators, sinks, clocks, and ID
factories must meet the host's concurrency requirements. JSONL thread safety is
not a guarantee for concurrent writers in separate processes.

## Deployment boundary

The primary product is an installed Python package, with a headless bot and terminal
UI as reference consumers. CI exercises source tests, installed-wheel behavior,
packaging, and benchmark reproducibility. The release workflow targets TestPyPI;
its presence does not prove publication succeeded. No hosted service, database,
Kubernetes cluster, or cloud deployment is required by this architecture.

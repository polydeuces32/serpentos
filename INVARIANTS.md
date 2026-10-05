# Protected contracts

These are engineering requirements. Test coverage supports them within tested cases;
it does not prove arbitrary plugins or deployments safe.

| Contract | Evidence to inspect |
|---|---|
| Importing the public runtime does not load Snake or curses; runtime code stays independent of game code. | `tests/test_public_api.py`, installed-wheel CI |
| Core game logic remains independent of terminal rendering. | `serpentos/core.py`, `tests/test_core.py`, `tests/test_compatibility.py` |
| Contexts are deeply immutable; decisions, replay, and comparison do not mutate supplied contexts. | `tests/test_runtime_models.py`, `tests/test_runtime_invariants.py` |
| Policies propose decisions without executing actions. Policy implementations honor the purity contract. | `serpentos/runtime/policy.py`, `tests/test_sdk_contract.py` |
| Configured validators reject disallowed actions. Fallback is explicit, validated, and records both attempts when an audit sink is configured. | `tests/test_runtime_engine.py`, `tests/test_runtime_invariants.py` |
| Omitting a validator is explicitly unvalidated behavior, not a security guarantee; omitting a sink does not persist audit records. | `serpentos/runtime/engine.py`, `tests/test_runtime_engine.py` |
| Serialized policies are data, with a closed operator vocabulary; runtime does not execute serialized expressions. | `tests/test_public_api.py`, `tests/test_runtime_invariants.py`, `tests/test_policies_rules.py` |
| Unknown audit schema versions are rejected; redaction applies to configured keys. | `tests/test_runtime_audit.py`, `tests/test_runtime_invariants.py` |
| Nondeterministic policies cannot receive a guaranteed deterministic replay claim. | `tests/test_runtime_replay.py` |
| Public exports and persisted contracts follow the stability tiers and version rules. | `tests/test_api_surface.py`, [compatibility policy](docs/COMPATIBILITY.md) |
| Published benchmark meaning and Q-table state encoding must not change silently. | `tests/test_core.py`, [contributor rules](CONTRIBUTING.md) |

If a proposed change conflicts with a contract, document the conflict and required
compatibility migration before implementation. Do not delete enforcing tests to make
the change appear successful.

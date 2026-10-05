# Project intent

Provide an embedded Python runtime that makes application decision logic explicit,
validatable, auditable, replayable, and comparable without requiring an external service.
The host remains responsible for execution, permissions, and real-world consequences.

## Goals

- Separate context, policy, validation, audit, and host execution behind explicit interfaces.
- Support rule-based, weighted, and learned policies through the same policy contract.
- Make policy changes assessable against recorded cases before a host adopts them.
- Preserve a small standard-library runtime and documented compatibility guarantees.
- Maintain Snake as a complete reference integration and learning demonstration.

## Non-goals

- A chatbot, operating system, hosted SaaS API, or general autonomous action executor.
- Requiring an LLM, GPU, network connection, or cloud account to make policy decisions.
- Treating a matching replay or higher Snake score as proof of business value or safety.
- Promising universal determinism for exploratory or third-party policies.

## Success criteria

A developer can embed a policy, configure validation and audit, inspect decisions,
and assess changes with replay or comparison. Tests and installed-package CI enforce
the contracts in [INVARIANTS.md](INVARIANTS.md); readiness claims follow
[DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md). Product expansions require an explicit
design decision rather than being inferred from the project name.

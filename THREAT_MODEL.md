# Threat model

Scope: embedded runtime, serialized policy/audit input, configured Python plugins,
and host-managed audit storage. This is a design review, not a penetration-test result.

## Assets and trust boundaries

Protect action permissions, context confidentiality, audit integrity, and host
availability. Contexts and serialized files can cross an untrusted-data boundary.
The validator is configured by the host, separately from policy proposals.
Custom Python policies are trusted executable code: the runtime does not sandbox them.
The host's action executor is outside SerpentOS's security boundary.

| Threat | Existing control or required host control | Residual risk |
|---|---|---|
| Executable expressions in imported rules | Closed operator vocabulary and data-only serialization; contract tests | Arbitrary Python plugins still execute with host privileges |
| Policy proposes an unauthorized action | Host-configured validator; explicit audited fallback | No validator means unvalidated acceptance; host must authorize execution |
| Policy changes context seen by another candidate | Deeply immutable context models and invariant tests | A plugin can still access external mutable state |
| Sensitive fields leak to audit | Configurable recursive key redaction and context omission | Unconfigured keys, free text, or returned records can still contain secrets |
| Tampered or incompatible audit records | Schema/version validation | Files are not cryptographically authenticated; filesystem access can alter them |
| Replay presented as safety proof | Determinism reporting and explicit match results | Matching an action does not prove safe execution or business value |
| Expensive input or plugin exhausts resources | Host must bound input size, depth, policy count, time, and storage | No general plugin isolation or host resource governor |
| Concurrent audit writers corrupt storage | Built-in sinks lock within one process | Multiple processes need separate files or host coordination |
| Dependency or release compromise | Minimal runtime dependencies and reviewed CI publishing permissions | Build tools, UI dependency, and release infrastructure remain supply-chain boundaries |

Review this document when adding operators, formats, plugins, sinks, dependencies,
or external execution. Add targeted tests for new controls and record remaining risks.

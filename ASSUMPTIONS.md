# Assumption register

Verified implementation details belong in architecture or status. This register
tracks premises that need validation before they can support a decision.

| Premise | State | Required validation |
|---|---|---|
| Every custom policy is pure and reports determinism honestly. | Host responsibility; unverified for third-party code | Review implementation, test repeated decisions, and mark exploratory policies nondeterministic. |
| Custom runtime components can be shared safely across threads. | Unverified per integration | Exercise the actual shared components under the intended concurrency. |
| Audit redaction covers every sensitive field. | Unverified per integration | Review field names, decision metadata, nested data, and retained context; verify representative records. |
| A replay match predicts a beneficial production outcome. | Unsupported inference | Evaluate representative cases and application-specific outcomes separately. |
| Package metadata and a release workflow prove index availability. | Unsupported inference | Inspect the completed publish run and install the exact version from the intended index. |
| Current performance is adequate for a future host workload. | Unverified | Benchmark representative payloads, policy sizes, sink configuration, and concurrency on target hardware. |

Update entries when evidence changes. Add an owner and review trigger for
integration-specific assumptions; do not convert a proposal into a verified fact.

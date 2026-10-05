# Proposed next work

This is a proposal list, not a claim of implemented functionality or a release commitment.
The current verified snapshot is [STATUS.md](STATUS.md).

1. Adopt the documentation map through review and passing CI.
2. Verify the current TestPyPI publication and installed-package smoke test;
   record the exact revision and run evidence before making availability claims.
3. Before the next version, remove the release verification step's hard-coded
   `serpentos==2.0.0` assumption through a separate tested release-workflow change.
4. Evaluate one real host integration using representative contexts, explicit
   validation, redacted audit, and application-specific outcome metrics.
5. Add performance or recovery requirements only after the host workload is known.

Record consequential choices in `docs/decisions/`. Hosted services, new model
dependencies, and broader execution capabilities require a scope decision against
[INTENT.md](INTENT.md).

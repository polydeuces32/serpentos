# Verified status

Snapshot date: 2026-10-05. Implementation examined at
`28acf1d583f96275de1bd671cc9c3b4ac2d290bf` (main at inspection).
This is a dated observation, not a claim about every future checkout.
Evidence: [verification record](docs/evidence/2026-10-05-governance.md).

| Item | State | Basis |
|---|---|---|
| Package version and minimum Python | Verified metadata | `pyproject.toml`: 2.0.0, Python >=3.9 |
| Embedded runtime, policy adapters, audit, replay, comparison | Verified source inspection and local tests | Existing package and test suite |
| Local source test suite | PASS | 493 tests on Linux, Python 3.12.14 |
| Cross-platform CI coverage | Configuration verified; current run not claimed here | Ubuntu/macOS/Windows, Python 3.9 and 3.13 |
| Installed-wheel and benchmark checks | Configuration verified; not run locally for this documentation change | `.github/workflows/ci.yml` |
| Release publication | Workflow configured; index availability unverified in this snapshot | TestPyPI trusted-publishing workflow |
| Hosted production service, SLA, production workload performance | Unverified and outside current product scope | No deployment evidence collected |

## Follow-up

- Require passing PR CI before merging this documentation change.
- Verify publication with an exact-version install before advertising availability.
- Release verification currently pins 2.0.0; address before publishing another version.
- Review host-specific assumptions and security controls before production integration.

Update this snapshot when new evidence materially changes the state. Keep historical
records under `docs/evidence/`; never silently replace an old observation with a plan.

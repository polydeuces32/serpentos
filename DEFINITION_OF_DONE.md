# Definition of done

Completion is scoped to the change. Report passed, failed, skipped, and blocked
checks honestly; a configured CI job is not an observed passing result.

## Every change

- The change addresses the stated task and preserves unrelated behavior.
- Relevant contracts and compatibility tiers were reviewed.
- `python -m unittest discover -s tests` passes.
- Behavioral fixes have regression coverage; documentation-only changes require
  accurate source references and valid relative links, not artificial unit tests.
- Changed documentation has one authoritative owner per fact and distinguishes
  observed behavior from requirements and plans.
- Review the final diff for unintended changes and sensitive data.
- Report the revision, environment, executed checks, results, and limitations.

## Applicable additional checks

| Change | Additional validation |
|---|---|
| Public API or serialized formats | Export/contract tests, stability-tier review, migration/version implications |
| Runtime or policy behavior | Validation, audit, replay, comparison, and relevant concurrency/invariant tests |
| Packaging | `python -m build`, `python -m twine check dist/*`, wheel installation and smoke test outside the source tree |
| Benchmark or state encoding | Contributor benchmark/version process and reproducibility CI |
| Terminal rendering | Relevant UI tests and manual 8-color/monochrome checks where supported |
| Release | Successful target-index publication and exact-version install; workflow dispatch requires release authorization |

Build and metadata checks require development tooling (`build` and `twine`),
not new runtime dependencies. No lint or type-check gate is currently configured;
do not claim those checks passed. Required CI must succeed before merging.

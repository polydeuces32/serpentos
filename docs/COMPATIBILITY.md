# Compatibility policy

What SerpentOS promises not to break, what it reserves the right to change, and
how you will find out. Short, because a policy nobody reads protects nobody.

SerpentOS follows [semantic versioning](https://semver.org/). The current
release is **2.0.0**.

---

## What counts as public API

Exactly three things:

1. **Names exported from `serpentos`** — everything in `serpentos.__all__`.
2. **Names exported from `serpentos.policies`** — everything in its `__all__`.
3. **The persisted audit record format**, identified by its `schema_version`.

Anything else is internal. In particular:

- **Module paths are not API.** `serpentos.runtime.engine.DecisionEngine` works
  today, but the guarantee is `from serpentos import DecisionEngine`. Modules
  may be split or merged in a minor release; the top-level names will not move.
- **Anything with a leading underscore is private**, at any depth. So is
  anything reached through one.
- **Repr strings, exception message wording, log text and docstrings are not
  API.** Match on exception *types*, never on their text.
- **`serpentos.core`, `serpentos.bot`, `serpentos.serpentos` and
  `serpentos.theme` are the game**, not the library. They are public in the
  sense that the CLI depends on them, but they carry no library-stability
  promise.

The public surface is pinned by a test (`tests/test_api_surface.py`). Removing
or renaming an exported name fails CI, which makes this policy something the
build enforces rather than something a document asserts.

---

## Stability tiers

### Stable

Breaking changes require a major version bump.

| | |
|---|---|
| Models | `DecisionContext`, `Decision`, `Outcome` |
| Interfaces | `Policy`, `BasePolicy`, `DecisionValidator` |
| Engine | `DecisionEngine`, and its decide-validate-audit lifecycle |
| Guardrails | `ActionValidator`, `ValidationResult` |
| Policies | `RulePolicy`, `WeightedPolicy`, and the `Rule` / `Condition` vocabulary |
| Audit | `AuditRecord`, `InMemoryAuditLog`, `JsonlAuditLog`, `NullAuditSink`, `read_jsonl` |
| Errors | The whole `SerpentOSError` hierarchy |

For these, within a major version:

- Exported names keep working and keep their meaning.
- Existing parameters keep their names, order and defaults. New parameters are
  added keyword-only-by-convention, at the end, with a default that preserves
  today's behaviour.
- `to_dict()` may gain keys. It will not remove or repurpose one.
- `from_dict()` keeps accepting anything an earlier release in the same major
  version produced.

### Experimental

Working and tested, but not yet shaped by enough real use to freeze. **These may
change in a minor release**, with the change described in the changelog.

- `replay`, `replay_all`, `ReplayResult`, `ReplayReport`
- `compare`, `ComparisonReport`, `PolicyReport`, `OutcomeSummary`,
  `MetricSummary`, `DisagreementSummary`, `Disagreement`, `PairDisagreement`
- `QLearningPolicy`
- Everything in `serpentos.environments.snake`

The most likely change is the shape of the comparison reports. If you persist
them, persist `to_dict()` output rather than pickling the objects, and be
prepared for new keys.

`Outcome` deserves a specific note. It is **stable**, and the optional `score`
alongside named `metrics` is a deliberate decision rather than an accident: most
real outcomes have no honest scalar summary, so the model refuses to demand one.
If that split turns out to be wrong, fixing it is a 3.0 change.

### Not API at all

The game (`serpentos.core` and friends), the CLI's exact output text, the
`~/.serpentos` on-disk layout, and every module path.

---

## Serialized formats

Anything written to disk carries a version, and readers check it.

| Format | Version field | Current |
|--------|---------------|---------|
| Audit records (JSONL) | `schema_version` | 1 |
| Exported Q-learning policies | `serpentos.policy/1` type tag | 1 |
| Benchmark results | `benchmark_version` | 1 |

The rules:

- **Every persisted record states its version.** No guessing from shape.
- **A reader that meets a version it does not know fails loudly.** Reading an
  audit record from a newer SerpentOS raises `AuditError` naming the version,
  rather than parsing what it recognises and silently dropping the rest. A
  misread audit log is worse than an unreadable one.
- **Additive changes do not bump the version.** A new optional key that older
  readers can ignore is not a new schema.
- **A version bump means a shape change**, and the new reader will still accept
  the old version for at least one major release.

Audit records written before `schema_version` existed have exactly the version-1
shape and are read as version 1. That is intentional, not tolerance.

---

## Deprecation

Nothing is removed without warning:

1. The replacement ships first, alongside the old thing.
2. The old thing keeps working, emits a `DeprecationWarning`, and is documented
   as deprecated with its replacement named.
3. It is removed no earlier than the next major version.

There are no deprecations in 2.0.0.

---

## What a change to this repository should not do

For contributors, the practical version:

- Do not move a name out of `serpentos.__all__` without a major bump.
- Do not reorder or rename an existing parameter. Add at the end, with a default.
- Do not change what `to_dict()` emits for an existing key.
- Do not change an audit record's shape without bumping `AUDIT_SCHEMA_VERSION`
  and teaching the reader both versions.
- Do not promote something out of Experimental casually. Promotion is a promise.
- Do add to `tests/test_api_surface.py` when you add a public name — the test
  will tell you.

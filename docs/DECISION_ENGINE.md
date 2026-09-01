# SerpentOS as a decision engine

A small kernel for decision logic: define a policy, run it under guardrails,
record what it decided, and prove later that it still decides the same way. It
has no dependencies beyond the Python standard library, makes no network calls,
starts no services and knows nothing about Snake, Q-learning or machine
learning.

**What SerpentOS is now.** The repository began as a Snake game with a
Q-learning agent, and that still works exactly as it did — see the
[README](../README.md). But the interesting part turned out to be the shape
underneath it: something observes a situation, something chooses an action, and
something else has to live with the result. That shape is not about Snake, so it
has been extracted into a library you can use for retry strategies, queue
prioritisation or routing. Q-learning is now one policy implementation among
several, reached through the same interface as everything else. Snake is the
reference environment that proves the interface is honest.

This document covers the architecture, the interfaces, the built-in policies,
what is and is not guaranteed, the security model, and the things it
deliberately does not do.

**Maturity.** The core — context, policy, decision, validation, audit — is
settled and heavily tested. Replay, comparison and the Q-learning adapter are
newer and marked *experimental* below: they work and are tested, but their
signatures may change in response to real use. Nothing here has been through a
long production soak. Treat version numbers accordingly.

---

## Why this exists

Most systems have decision logic that nobody is comfortable changing. Which
requests to retry. Which queue to drain first. Which replica to route a read to.
Which users get the new checkout flow. It is usually a handful of nested `if`
statements somewhere in a service, with three properties in common:

- **It is not tested in isolation**, because the conditions are tangled up with
  the effects. You cannot ask "what would this decide?" without also doing the
  thing.
- **It is not explainable after the fact.** When something goes wrong, the logs
  say what happened, not why it was chosen.
- **It cannot be changed safely.** There is no way to ask "would this edit have
  altered any of yesterday's decisions?" short of shipping it.

None of that needs a platform to fix. It needs one structural discipline —
separating the choice from the act — and a small amount of machinery around it.
That is all this is.

---

## Architecture

```
        DecisionContext            what the policy is allowed to see
               |
               v
            Policy                 pure function: context -> decision
               |
               v
        DecisionEngine
         /     |     \
   validate  identify  audit
         \     |     /
               v
           Decision                a proposal, not an action
               |
               v
      [ your application ]         the only thing that executes anything
               |
               v
            Outcome                what actually happened, reported back
```

And offline, over the recorded decisions:

```
    audit records ---> replay()   does this policy still decide the same way?
    context dataset -> compare()  how do these policies differ?
```

### Modules

| Module | Responsibility |
|--------|----------------|
| `serpentos.runtime.models` | `DecisionContext`, `Decision`, `Outcome` — immutable, JSON-only |
| `serpentos.runtime.policy` | The `Policy` interface and the `BasePolicy` helper |
| `serpentos.runtime.engine` | `DecisionEngine` — invoke, validate, identify, audit |
| `serpentos.runtime.validation` | `ActionValidator`, `ValidationResult` |
| `serpentos.runtime.audit` | `AuditRecord` and the in-memory / JSONL sinks |
| `serpentos.runtime.replay` | `replay`, `replay_all` |
| `serpentos.runtime.comparison` | `compare` |
| `serpentos.runtime.errors` | The typed exception hierarchy |
| `serpentos.policies` | `RulePolicy`, `WeightedPolicy`, `QLearningPolicy` |
| `serpentos.environments.snake` | The reference integration |

`serpentos.runtime` imports nothing from `serpentos.core`, `serpentos.bot` or
`serpentos.serpentos`, and `import serpentos` pulls in only the runtime. There
is a test that asserts this, because the boundary is easy to erode by accident.

---

## Interfaces

### DecisionContext

```python
DecisionContext(values: Mapping[str, Any], request_id: Optional[str] = None)
```

The information supplied to a policy. `values` is arbitrary JSON chosen by you.
`request_id` is an optional correlation identifier copied onto audit records; it
is not the decision identifier, which the engine assigns.

Contexts are deeply immutable. Nested dictionaries become read-only mappings and
lists become tuples, recursively, at construction time. Values that JSON cannot
represent — objects, callables, `NaN`, non-string keys, structures nested more
than 32 levels deep — raise `ConfigurationError` immediately rather than at
serialisation time.

### Policy

```python
class Policy(Protocol):
    name: str
    version: str
    def decide(self, context: DecisionContext) -> Decision: ...
```

Anything with that shape is a policy. `BasePolicy` is a convenience base class
that handles identity and stamps decisions correctly, but subclassing is not
required.

An optional `deterministic` attribute (assumed `True` when absent) declares
whether the same context always produces the same decision. Set it to `False`
if your policy consults an unseeded random source, the clock or anything else
outside the context.

**Policies must not perform side effects.** No writes, no network, no clock
reads, no mutating the context. This is not a style preference — it is the
property the rest of the runtime is built on. A policy that reaches outside its
context cannot be replayed, cannot be compared fairly offline, and cannot be
trusted to behave the same way twice.

### Decision

```python
Decision(action: str, policy_name: str, policy_version: str,
         metadata: Mapping[str, Any] = {}, decision_id: Optional[str] = None)
```

What a policy proposes. Inert by construction: producing one changes nothing.
The engine assigns `decision_id` if the policy did not, and propagates it if it
did. `metadata` is free-form JSON for explaining the choice — which rule fired,
what the scores were. It is data. Nothing ever executes it.

### Outcome

```python
Outcome(success: bool, score: Optional[float] = None,
        metrics: Mapping[str, float] = {},
        metadata: Mapping[str, Any] = {}, decision_id: Optional[str] = None)
```

What happened after your application executed a decision, in three registers:

- `success` — did it work? Means whatever you say it means; the runtime never
  infers it.
- `score` — one number, when one honestly exists: a reward signal, a utility, a
  realised profit. **Optional on purpose.** Most real decisions have no truthful
  scalar summary, and forcing one is how you end up optimising the wrong thing.
- `metrics` — named numbers: latency, cost, retries. Usually the honest
  representation. `compare()` aggregates each separately rather than collapsing
  them into a total.

Outcomes are reported by you, not produced by the runtime. Nothing consumes them
automatically — they exist so `compare()` can aggregate real results instead of
guessing.

### DecisionEngine

```python
DecisionEngine(policy, *, validator=None, audit_sink=None,
               fallback_policy=None, clock=utc_now, id_factory=uuid4hex)
```

The lifecycle of `engine.decide(context)`:

1. Check the argument is a `DecisionContext`.
2. Invoke the policy. Any exception it raises is re-typed as `PolicyError` with
   the original on `__cause__`. A non-`Decision` return is also a `PolicyError`.
3. Assign a decision identifier, or keep the one the policy set.
4. Validate. No validator means an explicit "no validator configured" verdict on
   the record, not a silent pass.
5. Build an audit record — including for rejected decisions — and hand it to the
   sink.
6. Return the decision, or raise `DecisionValidationError`.

`clock` and `id_factory` are injectable so tests can be deterministic.

The engine holds no mutable state, so one instance can serve a whole process.
Whether it is safe across threads depends on the sink; the built-in sinks are
not synchronised.

**The engine never executes the action.** Nothing in SerpentOS does.

### ActionValidator

```python
ActionValidator(allowed_actions: Iterable[str], *, name="action-allowlist")
```

The allow-list is frozen at construction, so a policy holding a reference to the
validator cannot extend it. Validators return a `ValidationResult` rather than
raising — the engine turns a failing result into an exception, which keeps
validators usable in offline comparison where a rejection is a data point rather
than a failure.

Rejection is loud. The engine raises. The only way an action gets substituted is
an explicitly configured `fallback_policy`, and then both the rejection and the
replacement appear in the audit log.

### Audit

`AuditRecord` carries `decision_id`, `timestamp`, `policy_name`,
`policy_version`, `action`, `context`, `decision_metadata`, `validation_result`
and `request_id`. It is plain JSON with sorted keys, so equal records serialise
byte-identically and can be diffed or hashed.

Two sinks ship:

- `InMemoryAuditLog(max_records=10_000, redact=(), include_context=True)` —
  bounded; the oldest records are dropped at the cap.
- `JsonlAuditLog(path, max_bytes=5MB, redact=(), include_context=True,
  fsync=False)` — one JSON object per line, rotated to `path.1` at the size
  threshold so a long-running service cannot fill a disk.

`NullAuditSink` is the explicit way to say "do not persist decisions".

**Nothing is persisted unless you attach a sink.** The runtime does not assume
your context is safe to keep — applications routinely put tokens and personal
data in the values a policy reads. Pass `redact=["authorization", "ssn"]` to mask
values by key name at any depth, in both the context and the decision metadata,
or `include_context=False` to drop the context entirely.

---

## The built-in policies

Three ship in the box. None of them is privileged: the engine cannot tell them
apart, and neither can `replay()` or `compare()`.

### RulePolicy — ordered, declarative logic

The one to reach for first. An ordered list of rules; the first whose condition
matches wins, and a `default_action` catches everything else. Reading the rules
top to bottom *is* reading the logic.

```python
from serpentos.policies import Rule, RulePolicy, AllOf, when

policy = RulePolicy(
    name="retry-rules",
    version="1.0",
    rules=[
        Rule("fail", when("attempt", "ge", 3), name="attempts-exhausted"),
        Rule("wait", when("status_code", "eq", 429), name="rate-limited"),
        Rule("retry", when("status_code", "ge", 500), name="transient"),
    ],
    default_action="fail",
)
```

The decision's metadata names the rule that fired, so an audit record says
*why*, not just *what*. Conditions compose with `AllOf`, `AnyOf` and `Not`, and
the whole rule set round-trips through `to_dict()` / `from_dict()` as plain JSON
— which is the point. A rule set is **data**, so a config file cannot smuggle in
executable code. See [the security model](#security-model).

`Predicate` wraps an arbitrary Python callable for logic the closed operator set
cannot express. It is deliberately not serialisable; that is the trade.

### WeightedPolicy — quantitative scoring

When the answer is a trade-off rather than a branch. Each candidate action gets
a score, highest wins, ties break by declaration order.

```python
from serpentos.policies import WeightedPolicy

carrier = WeightedPolicy.from_linear("carrier-choice", "1.0", {
    name: {
        "weights": {"cost_usd": -1.0, "days": -4.0, "reliability": 30.0},
        "source": name,
    }
    for name in ("ups", "fedex", "usps")
})
```

`source` tells a scorer to read its factors from that key's nested object, so
each candidate is scored on its own attributes:

```python
decision = carrier.decide(DecisionContext({
    "ups":   {"cost_usd": 12.50, "days": 2, "reliability": 0.97},
    "fedex": {"cost_usd": 18.00, "days": 1, "reliability": 0.99},
    "usps":  {"cost_usd":  8.25, "days": 4, "reliability": 0.91},
}))
# decision.action == "ups"
# decision.metadata["scores"]  == {"ups": 8.6, "fedex": 7.7, "usps": 3.05}
# decision.metadata["factors"]["ups"]
#     == {"cost_usd": -12.5, "days": -8.0, "reliability": 29.1}
```

The `factors` breakdown is the part worth caring about. "UPS scored 8.6" is not
an explanation; "cheapest-but-one, penalised two days in transit, carried by
reliability" is. Contributions are derived *from* the breakdown rather than
computed alongside it, so the explanation and the score cannot drift apart.

**This is not machine learning.** Nothing is fitted and nothing learns. The
weights are numbers a human chose and can defend in review, which for most
routing problems is a feature rather than a limitation.

### QLearningPolicy — the learned agent, as one option *(experimental)*

The Snake agent, wrapped to fit the same interface.

```python
from serpentos.policies import QLearningPolicy

policy = QLearningPolicy.from_policy_file("policy.json")
```

Strictly read-only: it looks up states with `QAgent.peek`, so serving decisions
never grows the Q-table, never changes its fingerprint and never perturbs the
benchmark. The version defaults to the table's fingerprint, so replaying against
a differently-trained table is caught rather than silently tolerated. Set
`explore=True` and the policy correctly declares itself non-deterministic, which
propagates into replay as `guaranteed=False`.

It is one policy. The runtime has no opinion about learning.

---

## Deterministic guarantees

These hold, and are covered by tests:

- **Serialisation is canonical.** Equal models produce byte-identical JSON,
  regardless of the order keys were inserted in.
- **Contexts are never mutated** by deciding, replaying or comparing.
- **A policy that declares `deterministic = True` and honours the contract
  replays to a match.** `replay()` reports `original_action`, `replayed_action`
  and `match`.
- **Comparison never mutates the policies it evaluates.**
- **`ActionValidator` never accepts an action outside its allow-list.** This is
  checked against thousands of generated actions, not one example.
- **Read-only Q-table access.** `QLearningPolicy` uses `QAgent.peek`, so serving
  decisions never grows the table and never changes its fingerprint or its
  benchmark score.

These do **not** hold, and the runtime says so rather than pretending:

- **Replay does not certify a nondeterministic policy.** If the policy declares
  `deterministic = False` — an exploring Q-learning agent, anything consulting an
  unseeded RNG — every `ReplayResult` is flagged `guaranteed=False`. A match is
  then evidence, not proof.
- **Replay cannot verify configuration.** It checks the policy's name and
  version against the record and refuses a mismatch in strict mode. It has no way
  to know whether you have edited the rules behind an unchanged version string.
  Bump your versions.
- **Timestamps are not deterministic** unless you inject a `clock`. Neither are
  decision identifiers unless you inject an `id_factory`.
- **A policy that breaks the purity contract breaks everything downstream.** The
  runtime cannot detect a policy that reads a database. It can only be clear that
  such a policy is outside the contract.

---

## Security model

The threat this design takes seriously: **a policy definition arriving from
somewhere you do not fully trust** — a config file, another team, a pull request
you skimmed. The guarantee is that loading and running one cannot execute
arbitrary code.

What that means concretely:

- **No `eval`, no `exec`, no `pickle`, no `marshal`, no `__import__`, no name
  resolution.** There is a test that greps the runtime and policy modules for
  these tokens.
- **Rule conditions are a closed operator set.** `eq`, `ne`, `lt`, `le`, `gt`,
  `ge`, `in`, `not_in`, `contains`, `startswith`, `endswith`, `exists`,
  `missing`. Adding one is a code change and a review, not a config change. There
  is no expression parser and there will not be one.
- **`condition_from_dict` constructs only built-in condition types.** An unknown
  `type` raises. Nesting beyond 16 levels raises.
- **`Predicate` — the Python-callable escape hatch — is deliberately not
  serialisable.** `to_dict()` raises, so a rule set with a predicate cannot round
  trip through a config file, and a config file cannot conjure one into being.
- **Q-learning policies are pure data.** A `serpentos.policy/1` file is a table
  of state keys to three floats, with a SHA-256 fingerprint. Importing one runs
  nothing the author wrote.
- **Audit parsing validates every field.** Malformed JSON, wrong types, missing
  fields and lines over 1 MB all raise `AuditError` naming the line. Nothing is
  reconstructed by type name.
- **Writes are bounded.** In-memory logs have a record cap; JSONL logs rotate.
- **Audit paths are trusted configuration.** They are never derived from context
  data — that is how a policy input becomes a path traversal. Files are created
  mode `0600`. Records are appended with a single `write` to an `O_APPEND`
  descriptor, so concurrent writers interleave whole lines rather than corrupting
  each other. Durability is separate: pass `fsync=True` if you need it, at a real
  cost in throughput.
- **Secrets are not logged by accident, but they are not detected either.** The
  runtime cannot tell which of your fields is sensitive. Redaction is opt-in and
  by key name; if you do not configure it, the context is persisted verbatim.

Residual risks worth naming: a hostile rule set can still describe comparisons
you did not intend, and can be large enough to be slow. Validate what you load,
and cap what you accept.

---

## Failure behaviour

Everything raises a typed exception descending from `SerpentOSError`. Nothing is
swallowed and nothing returns `None` where it should fail.

| Exception | Raised when |
|-----------|-------------|
| `ConfigurationError` | A component was built with unusable arguments, or a value cannot be represented as JSON |
| `PolicyError` | A policy raised, or returned something that is not a `Decision` |
| `DecisionValidationError` | A validator rejected the proposal. Carries the `ValidationResult` on `.result` |
| `ReplayError` | A record cannot be replayed: no context, identity mismatch in strict mode, or the policy failed |
| `AuditError` | A record could not be written, read or parsed |

Audit failure fails the decision. A decision nobody can account for is worse
than no decision.

---

## Comparison

*Experimental: the reports are useful and tested, but their shape may change.*

```python
compare(policies, cases, *, validator=None, outcome_fn=None,
        max_error_samples=5, max_disagreement_samples=5)
```

Runs every policy over every context and reports, per policy: decision count,
action distribution, error count with sampled messages, and validation failures.
If you pass an `outcome_fn`, it also aggregates success rate, the optional
score, and each metric's count, total, mean, min and max.

A policy that raises is isolated — the exception is counted against that policy
and the run continues, because finding out which policies are broken is the
point.

### Disagreement is the measurement that matters

Action distributions are easy to over-read. Consider two retry policies, each of
which retries exactly half the time:

| Policy | retry | fail |
|--------|-------|------|
| A      | 50    | 50   |
| B      | 50    | 50   |

Identical. Interchangeable, surely. Except they might not agree on a single
request between them — A retrying precisely the ones B gives up on. The
distributions cannot show you that, so `compare()` measures it directly:

```python
report = compare([policy_a, policy_b], cases)
summary = report.disagreements

summary.compared        # cases where every policy managed to decide
summary.agreed          # ... and chose the same action
summary.disagreed
summary.agreement_rate
summary.pairs           # per pair: compared, disagreed, rate
summary.examples        # worked examples, capped; counts are always complete
```

Pairwise counts matter once you have three or more candidates, where "they
mostly agree" can hide one outlier disagreeing with everyone. A case that any
policy errored on is excluded from the overall tally — a crash is not an
opinion — though pairs that both decided it still count it.

`disagreements` is `None` when you compare a single policy, which has nobody to
disagree with.

**There is no overall score and there will not be one.** A policy that retries
twice as often may be better or catastrophically worse depending on what a retry
costs you, and the runtime does not know. `outcome_fn` is where you encode what
success means.

---

## A worked example, with no Snake in it

[`examples/retry_policy.py`](../examples/retry_policy.py) is the shortest honest
demonstration that this is a decision engine rather than a game. A service call
failed; something has to choose between retrying, waiting and giving up.

```bash
python examples/retry_policy.py
```

No network, no server, no database, no model, no API key. It runs in about a
tenth of a second and walks through the whole lifecycle:

1. **Deciding.** A rule policy proposes an action for six situations, behind an
   allow-list. Each one prints its decision id and the rule that fired.
2. **Auditing.** The full JSON record for one decision.
3. **Replaying.** The recorded decisions re-run against the unchanged policy
   (6/6 match, guaranteed), and then against a policy with the retry budget
   tightened from three attempts to two — which reports exactly the two
   decisions that would have changed, before you ship it.
4. **Comparing.** The rules against a weighted-scoring model over the same six
   situations, including where they disagree.

That third step is the one worth stealing. "Would this edit have altered any of
yesterday's decisions?" is normally answered by deploying and watching. Here it
is answered offline, exactly, in milliseconds.

---

## Snake, and what it is doing here

Snake is the reference environment. It is a real application with state,
actions, rewards, persistence, a trained policy and a frozen reproducible
benchmark — which makes it a far more honest demonstration than a toy example
would be.

`serpentos.environments.snake` shows the three things every integration does:

1. `context_from_state()` builds the context. Note that it publishes *derived*
   features (`food_left`, `food_ahead`) next to the raw ones. Deciding what a
   policy may see, and pre-computing anything awkward, is the host application's
   job.
2. `action_index()` maps the returned action back onto something the game can
   execute. The runtime never touches the environment.
3. `outcome_from_episode()` reports what happened, using this environment's
   definition of success.

The demonstration that matters is `survival_policy()`: eight hand-written rules,
no training, no Q-table, playing through the same engine, validator and audit log
as the learned agent. On the standard 22×78 grid it averages **36.9** food per
episode. An untrained Q-table averages **0.05**, and a Q-table trained for 1,500
episodes averages **11.9**.

That last comparison is not a claim that rules beat learning in general — it is a
small tabular agent on a hard credit-assignment problem, and
[ECOSYSTEM.md](ECOSYSTEM.md) is candid about that ceiling. It is a claim that the
interface is not secretly shaped around reinforcement learning, which is exactly
what a runtime meant for retry strategies and queue prioritisation needs to
prove.

---

## Limitations

Known, and deliberate for this phase:

- **Single-process, in-memory or file-backed.** No registry, no server, no
  distribution.
- **The built-in sinks are not thread-safe.** Concurrent JSONL writers append
  whole lines safely at the OS level, but a single sink object shared across
  threads is not synchronised.
- **No schema for contexts.** The runtime checks that values are JSON; it does
  not check that `attempts` is present or that it is an integer. A policy that
  needs a field should say so when it is missing.
- **Rule conditions cannot express relationships between two context fields.**
  Every comparison is one key against a fixed operand. Use `Predicate` in-process
  when you need more, and accept losing serialisability.
- **`compare()` buffers the whole case list in memory.**
- **Replay cannot detect a configuration change behind an unchanged version
  string.**
- **No migration story for audit records.** The format is unversioned. It has not
  changed yet, and when it does, that will need addressing.

Explicitly deferred: shadow traffic, automatic rollout and rollback, a REST or
gRPC server, remote policy registries, dashboards, authentication, distributed
execution, and any AI or LLM adapter.

---

## API reference

```python
from serpentos import (
    # models
    DecisionContext, Decision, Outcome,
    # policy interface
    Policy, BasePolicy, is_deterministic,
    # engine
    DecisionEngine,
    # validation
    ActionValidator, DecisionValidator, ValidationResult,
    # audit
    AuditRecord, AuditSink, InMemoryAuditLog, JsonlAuditLog, NullAuditSink,
    read_jsonl, REDACTED,
    # replay
    replay, replay_all, ReplayResult, ReplayReport,
    # comparison
    compare, ComparisonReport, PolicyReport, OutcomeSummary, MetricSummary,
    DisagreementSummary, Disagreement, PairDisagreement,
    # errors
    SerpentOSError, ConfigurationError, PolicyError,
    DecisionValidationError, ReplayError, AuditError,
)

from serpentos.policies import (
    RulePolicy, Rule, when, condition_from_dict,
    Condition, Comparison, AllOf, AnyOf, Not, Always, Predicate,
    WeightedPolicy, LinearScorer,
    QLearningPolicy,
)

from serpentos.environments.snake import (
    SNAKE_ACTIONS, context_from_state, action_index,
    survival_policy, run_policy_episode, evaluate_policy, outcome_from_episode,
)
```

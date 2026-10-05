# Engineering SerpentOS

SerpentOS is an embedded Python policy runtime. Snake is its reference environment.
Keep changes small, explicit, compatible, and supported by observed evidence.

## Read before changing code

- [INTENT.md](INTENT.md): product purpose and scope.
- [INVARIANTS.md](INVARIANTS.md): protected behavior and enforcing tests.
- [ARCHITECTURE.md](ARCHITECTURE.md): component ownership and data flow.
- [STATUS.md](STATUS.md): dated verification, never a substitute for fresh checks.
- [DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md): applicable completion checks.

Consult [ASSUMPTIONS.md](ASSUMPTIONS.md) before relying on an unverified claim,
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for API or format changes,
and [THREAT_MODEL.md](THREAT_MODEL.md) for trust-boundary changes.
[ROADMAP.md](ROADMAP.md) contains proposals, not implemented capabilities.
[CONTRIBUTING.md](CONTRIBUTING.md) owns contributor style and benchmark rules.
[docs/AGENT.md](docs/AGENT.md) describes running the training bot, not coding-agent instructions.

## Workflow

1. Inspect Git state, relevant code, existing tests, and CI before editing.
2. Identify the affected contract and smallest correct change. Preserve unrelated work.
3. Implement within the existing package structure; add regression coverage for behavioral fixes.
4. Run `python -m unittest discover -s tests` from the repository root.
5. For packaging or release changes, also run the applicable checks in the definition of done.
6. Report changed behavior, commands and results, limitations, and remaining blockers.

## Boundaries

- Runtime must not import the game or curses; policies propose actions, hosts execute them.
- Preserve pure policy decisions, immutable contexts, explicit validation and fallback behavior.
- Runtime and learning core use the standard library. The Windows UI dependency is already declared.
  Follow the existing contributor review process before adding dependencies.
- Never introduce executable serialized policies, hide audit failures, or weaken format validation.
- Do not rewrite exports, benchmark definitions, or persisted formats without the compatibility process.
- Do not publish packages or merge a pull request unless the task authorizes that action.

## Documentation and evidence

Give each fact one authoritative home and link to it elsewhere. Keep this file a map.
Update purpose or invariants only when the intended contract actually changes.
Record consequential design changes in `docs/decisions/` with context, alternatives,
decision, consequences, and verification; create an ADR when there is a real decision.
Status claims need a date, exact revision, environment, command, result, and evidence reference.
Separate verified facts, assumptions, proposals, and blocked work. Never invent checks,
deployment state, benchmark results, release availability, or completion percentages.

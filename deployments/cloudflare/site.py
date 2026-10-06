"""Static public site content and host configuration for SerpentOS."""

from __future__ import annotations

PUBLIC_HOSTS = frozenset({"serpentos.dev", "www.serpentos.dev"})
API_HOSTS = frozenset({"api.serpentos.dev"})


def landing_page() -> str:
    """Return the production landing page for the public SerpentOS site."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="SerpentOS is a deterministic decision runtime for reliable software systems.">
  <title>SerpentOS</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070a0d;
      --panel: #0d1217;
      --text: #f2f5f7;
      --muted: #9aa7b2;
      --line: #202a33;
      --accent: #8df0c0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 0%, #10251f 0, transparent 32rem),
        var(--bg);
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    main {
      width: min(980px, calc(100% - 40px));
      margin: 0 auto;
      padding: 96px 0 72px;
    }
    .eyebrow {
      color: var(--accent);
      font-size: 13px;
      letter-spacing: .14em;
      text-transform: uppercase;
    }
    h1 {
      margin: 18px 0 18px;
      max-width: 820px;
      font-size: clamp(48px, 8vw, 92px);
      line-height: .95;
      letter-spacing: -.055em;
    }
    .lead {
      max-width: 720px;
      margin: 0 0 44px;
      color: var(--muted);
      font-size: clamp(18px, 2.4vw, 24px);
      line-height: 1.55;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-top: 54px;
    }
    .card {
      min-height: 170px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: color-mix(in srgb, var(--panel) 92%, transparent);
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--accent);
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
    }
    code {
      color: var(--text);
      background: #111820;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 3px 7px;
    }
    footer {
      margin-top: 70px;
      padding-top: 22px;
      border-top: 1px solid var(--line);
      color: #6f7c86;
      font-size: 12px;
    }
    @media (max-width: 760px) {
      main { padding-top: 64px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Deterministic systems runtime</div>
    <h1>SerpentOS</h1>
    <p class="lead">
      A compact decision runtime for software that must behave predictably,
      validate every decision, and leave an auditable trail.
    </p>

    <div class="grid">
      <section class="card">
        <h2>Deterministic</h2>
        <p>Explicit policies produce bounded, inspectable outcomes instead of hidden behavior.</p>
      </section>
      <section class="card">
        <h2>Validated</h2>
        <p>Decisions are checked against strict action constraints before they are returned.</p>
      </section>
      <section class="card">
        <h2>Auditable</h2>
        <p>Every decision includes structured metadata and an audit record for downstream systems.</p>
      </section>
    </div>

    <footer>
      API: <code>api.serpentos.dev</code>
    </footer>
  </main>
</body>
</html>
"""


__all__ = ["API_HOSTS", "PUBLIC_HOSTS", "landing_page"]

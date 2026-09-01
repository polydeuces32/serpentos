"""Reference environments that consume the SerpentOS runtime.

An environment is a worked example, not part of the kernel. It shows how a real
application turns its own state into a
:class:`~serpentos.runtime.models.DecisionContext`, hands that to an engine, maps
the returned action back onto something it can execute, and reports an
:class:`~serpentos.runtime.models.Outcome`.

Only :mod:`serpentos.environments.snake` ships today. Nothing in
:mod:`serpentos.runtime` imports it, and nothing in it is required to use the
runtime.
"""

from __future__ import annotations

__all__ = ["snake"]

"""Cloudflare Python Worker entrypoint for SerpentOS."""

from __future__ import annotations

from urllib.parse import urlsplit

from workers import Response, WorkerEntrypoint

from deployments.cloudflare.api import handle_decision, health_payload


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        path = urlsplit(str(request.url)).path
        method = str(request.method).upper()

        if method == "GET" and path == "/health":
            return Response.json(health_payload())

        if method == "POST" and path == "/v1/decide":
            try:
                payload = await request.json()
            except Exception:
                return Response.json(
                    {"error": "request body must contain valid JSON"},
                    status=400,
                )

            status, body = handle_decision(payload)
            return Response.json(body, status=status)

        return Response.json({"error": "not found"}, status=404)

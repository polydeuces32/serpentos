"""Cloudflare Python Worker entrypoint for SerpentOS."""

from __future__ import annotations

from urllib.parse import urlsplit

from workers import Response, WorkerEntrypoint

from deployments.cloudflare.api import handle_decision, health_payload
from deployments.cloudflare.site import API_HOSTS, PUBLIC_HOSTS, landing_page


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        parsed = urlsplit(str(request.url))
        host = (parsed.hostname or "").lower()
        path = parsed.path
        method = str(request.method).upper()

        if method == "GET" and path == "/" and host in PUBLIC_HOSTS:
            return Response(
                landing_page(),
                headers={"content-type": "text/html; charset=utf-8"},
            )

        if method == "GET" and path == "/health" and host in API_HOSTS | PUBLIC_HOSTS:
            return Response.json(health_payload())

        if method == "POST" and path == "/v1/decide" and host in API_HOSTS | PUBLIC_HOSTS:
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

"""
Load test with Locust.

  pip install -r requirements-dev.txt
  locust -f stress/locustfile.py --host=https://YOUR-SERVICE.run.app

Open http://localhost:8089 to drive users/spawn rate.
Set API key via environment (optional):

  set LOCUST_API_KEY=your-key
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task


class RAGUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.api_key = os.environ.get("LOCUST_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    @task(3)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def query_light(self) -> None:
        self.client.post(
            "/v1/query",
            json={"question": "What is in the documents?", "top_k": 3},
            headers={**self._headers(), "Content-Type": "application/json"},
            name="/v1/query",
        )

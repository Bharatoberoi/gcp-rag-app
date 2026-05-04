"""
Async stress harness (no Locust UI).

  python stress/async_stress.py --url https://YOUR.run.app --concurrency 30 --requests 300

Uses only /health by default to avoid burning Vertex quota; add --hit-query to POST /v1/query.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import httpx


async def worker(
    base: str,
    client: httpx.AsyncClient,
    n: int,
    hit_query: bool,
    api_key: str | None,
) -> tuple[int, list[float]]:
    ok = 0
    lat: list[float] = []
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = await client.get(f"{base}/health", headers=headers)
            if r.status_code < 400:
                ok += 1
            if hit_query:
                rq = await client.post(
                    f"{base}/v1/query",
                    json={"question": "ping", "top_k": 2},
                    headers={**headers, "Content-Type": "application/json"},
                )
                if rq.status_code < 400:
                    ok += 1
        except Exception:
            pass
        lat.append((time.perf_counter() - t0) * 1000)
    return ok, lat


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("STRESS_BASE_URL", "http://127.0.0.1:8765"))
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--requests", type=int, default=200, help="Per worker")
    p.add_argument("--hit-query", action="store_true", help="Also POST /v1/query (uses Vertex)")
    p.add_argument("--api-key", default=os.environ.get("STRESS_API_KEY", ""))
    args = p.parse_args()
    base = args.url.rstrip("/")

    limits = httpx.Limits(max_connections=args.concurrency + 5, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        t0 = time.perf_counter()
        tasks = [
            worker(base, client, args.requests, args.hit_query, args.api_key or None)
            for _ in range(args.concurrency)
        ]
        results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    all_lat: list[float] = []
    total_ok = 0
    for ok, lat in results:
        total_ok += ok
        all_lat.extend(lat)

    all_lat.sort()
    def pct(p: float) -> float:
        if not all_lat:
            return 0.0
        i = int(len(all_lat) * p / 100.0)
        return all_lat[min(i, len(all_lat) - 1)]

    print(f"base_url={base}")
    print(f"concurrency={args.concurrency} per_worker_requests={args.requests} hit_query={args.hit_query}")
    print(f"elapsed_s={elapsed:.2f} success_count={total_ok}")
    if all_lat:
        print(f"latency_ms p50={pct(50):.1f} p95={pct(95):.1f} p99={pct(99):.1f} max={max(all_lat):.1f}")


if __name__ == "__main__":
    asyncio.run(main())

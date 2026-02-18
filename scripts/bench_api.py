"""
bench_api.py – Latency benchmark for the power-forecast API.

Usage:
    python scripts/bench_api.py [--url URL] [--n N]

Defaults:
    --url  http://127.0.0.1:8000/forecast?hours=24
    --n    50
"""

import argparse
import json
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description="Benchmark the /forecast endpoint.")
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:8000/forecast?hours=24",
        help="Full URL to benchmark (default: %(default)s)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of requests to send (default: %(default)s)",
    )
    args = ap.parse_args()

    times = []
    for i in range(args.n):
        t0 = time.time()
        try:
            with urllib.request.urlopen(args.url) as r:
                _ = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"HTTP {e.code} on iter {i}: {body}")
            raise
        except urllib.error.URLError as e:
            print(
                f"Connection error on iter {i}: {e.reason}\n"
                "Is the API server running?  Try:\n"
                "  uvicorn src.service.api:app --reload"
            )
            raise SystemExit(1)
        times.append((time.time() - t0) * 1000)

    times.sort()
    n = len(times)
    p50 = times[int(0.50 * (n - 1))]
    p90 = times[int(0.90 * (n - 1))]
    p99 = times[int(0.99 * (n - 1))]

    print(f"N={n}")
    print(f"p50={p50:.2f} ms")
    print(f"p90={p90:.2f} ms")
    print(f"p99={p99:.2f} ms")
    print(f"mean={sum(times)/n:.2f} ms")


if __name__ == "__main__":
    main()

import time
import json
import urllib.request
import urllib.error

URL = "http://127.0.0.1:8000/forecast?hours=24"
N = 50

times = []
for i in range(N):
    t0 = time.time()
    try:
        with urllib.request.urlopen(URL) as r:
            _ = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} on iter {i}: {body}")
        raise
    times.append((time.time() - t0) * 1000)

times.sort()
p50 = times[int(0.50 * (N - 1))]
p90 = times[int(0.90 * (N - 1))]
p99 = times[int(0.99 * (N - 1))]

print(f"N={N}")
print(f"p50={p50:.2f} ms")
print(f"p90={p90:.2f} ms")
print(f"p99={p99:.2f} ms")
print(f"mean={sum(times)/len(times):.2f} ms")

"""Local latency check (not part of pytest): start the service on this machine, send ~1,000 requests per case over
loopback, print p50 / p95 / max. These are LOCAL-MACHINE numbers (this laptop, loopback, one worker, no network),
not production numbers. Payload values are invented (1.0 for numbers, the first known label for categories).

    .venv/bin/python scripts/latency.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PORT, N, WARM = 8123, 1000, 50
spec = json.loads((ROOT / "models" / "service" / "spec.json").read_text())
sparse = {"TransactionAmt": 149.5, "ProductCD": "W", "card4": "visa", "card6": "debit", "C1": 2.0, "C2": 1.0}
wide = {f["name"]: (f["levels"][0] if f["kind"] == "category" else 1.0) for f in spec["features"]}
wide["TransactionAmt"] = 149.5
cases = [("single, 6 fields", "/score", sparse), ("single, all 431 fields", "/score", wide),
         ("batch of 10, 431 fields each", "/score_batch", [wide] * 10)]

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.app:create_app", "--factory", "--port", str(PORT),
                           "--no-access-log"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=10) as c:
        for _ in range(100):
            try:
                c.get("/health").raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        print(f"{'case':32}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}")
        for name, path, body in cases:
            data = json.dumps(body)
            ms = []
            for i in range(WARM + N):
                t = time.perf_counter()
                r = c.post(path, content=data, headers={"content-type": "application/json"})
                dt = 1000 * (time.perf_counter() - t)
                assert r.status_code == 200
                if i >= WARM:
                    ms.append(dt)
            print(f"{name:32}{np.percentile(ms, 50):9.1f}{np.percentile(ms, 95):9.1f}{max(ms):9.1f}")
finally:
    server.terminate()

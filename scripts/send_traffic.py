"""Send synthetic prediction traffic to a running service (stdlib only).

    python scripts/send_traffic.py --url http://localhost:8000 --rows 400 --scenario heatwave

``--scenario none`` matches the training distribution, ``heatwave`` is shifted so the
drift job and the Grafana drift panels have something to show. Stdlib only so CI can run it
without installing the project.
"""

import argparse
import json
import math
import random
import urllib.request


def make_row(rng: random.Random, scenario: str) -> dict:
    temp = rng.gauss(10 + 9 * math.sin(rng.random() * 2 * math.pi), 6)
    wind = rng.weibullvariate(6.0, 2.0)
    cloud = rng.betavariate(2, 2)
    if scenario == "heatwave":
        temp, wind, cloud = temp + 9, wind * 0.5, cloud * 0.4
    return {
        "temperature_c": round(max(-40, min(50, temp)), 2),
        "hour_of_day": rng.randrange(24),
        "day_of_week": rng.randrange(7),
        "is_holiday": int(rng.random() < 0.03),
        "wind_speed_ms": round(min(wind, 60), 2),
        "cloud_cover": round(cloud, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--scenario", choices=["none", "heatwave"], default="none")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    sent = 0
    while sent < args.rows:
        n = min(args.batch, args.rows - sent)
        body = json.dumps({"instances": [make_row(rng, args.scenario) for _ in range(n)]}).encode()
        req = urllib.request.Request(
            args.url + "/predict", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.load(resp)
        sent += n
    print(f"sent {sent} rows, scenario={args.scenario}, last model_version={out['model_version']}")


if __name__ == "__main__":
    main()

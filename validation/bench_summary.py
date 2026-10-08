#!/usr/bin/env python3
"""Resume validation/bench_fernandina.sh: mediana e IQR de pared, CPU y RAM pico."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])


def parse(p):
    s = p.read_text()
    w = re.search(r"Elapsed \(wall clock\) time.*: (.+)", s).group(1).split(":")
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(w)))
    cpu = (float(re.search(r"User time.*: (.+)", s).group(1))
           + float(re.search(r"System time.*: (.+)", s).group(1)))
    rss = int(re.search(r"Maximum resident set size.*: (\d+)", s).group(1)) / 1024
    return wall, cpu, rss


runs = defaultdict(lambda: defaultdict(list))  # cfg -> rep -> [(wall,cpu,rss)]
for p in sorted(out.glob("*.time")):
    parts = p.stem.split("_")
    cfg, rep = f"{parts[0]}_{parts[1]}", parts[2]
    if rep == "warm":
        continue
    runs[cfg][rep].append((parts[3] if len(parts) > 3 else "all", *parse(p)))

summary = {}
for cfg, reps in runs.items():
    tot = np.array([[sum(r[1] for r in st), sum(r[2] for r in st), max(r[3] for r in st)]
                    for st in reps.values()])
    stages = defaultdict(list)
    for st in reps.values():
        for name, w, c, m in st:
            stages[name].append((w, c, m))
    q = lambda a: dict(median=float(np.median(a)), q1=float(np.percentile(a, 25)),
                       q3=float(np.percentile(a, 75)), n=len(a))
    summary[cfg] = dict(wall_s=q(tot[:, 0]), cpu_s=q(tot[:, 1]), rss_mb=q(tot[:, 2]),
                        stages={k: dict(wall_s=q([x[0] for x in v]),
                                        rss_mb=q([x[2] for x in v]))
                                for k, v in stages.items()})
    w = summary[cfg]["wall_s"]
    print(f"{cfg:8s} wall {w['median']:7.2f} s [IQR {w['q1']:.2f}–{w['q3']:.2f}]  "
          f"CPU {summary[cfg]['cpu_s']['median']:7.1f} s  "
          f"RSS {summary[cfg]['rss_mb']['median']:7.0f} MB  (n={w['n']})")
    for k, v in sorted(summary[cfg]["stages"].items()):
        if k != "all":
            print(f"    {k:6s} {v['wall_s']['median']:7.2f} s  RSS {v['rss_mb']['median']:6.0f} MB")
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("→", out / "summary.json")

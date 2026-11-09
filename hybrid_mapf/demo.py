"""Local visual replay of the real controllers. Run: python3.12 -m hybrid_mapf.demo."""
import argparse
import json
import statistics
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import numpy as np
import torch

from .gate import LogisticGate
from .grid import make_instance, sum_dist
from .lacam import lacam
from .pibt import is_valid_step
from .policy import ObsBuilder, Policy
from .sim import Controller, flowtime

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "demo"
LOCK = Lock()
POLICY = None


@lru_cache(maxsize=24)
def simulate(seed, count, mode):
    inst = make_instance(seed, count)
    settings = json.loads((ROOT / "artifacts/gate.json").read_text())
    gate = LogisticGate.from_dict(settings)
    frames = [dict(positions=inst.starts.tolist(), method="Start", q=None,
                   arrived=int((inst.starts == inst.goals).sum()), remaining=sum_dist(inst, inst.starts),
                   search_calls=0, lacam_calls=0, lacam_nodes=0, compute_seconds=0)]
    if mode == "lacam":
        import time
        start = time.perf_counter()
        path, nodes = lacam(inst.grid, inst.starts, inst.goals, inst.dist,
                            max_nodes=20000, time_limit=10, rng=np.random.default_rng(seed))
        elapsed = time.perf_counter() - start
        if path is not None:
            frames.extend(dict(positions=q.tolist(), method="LaCAM", q=None,
                               arrived=int((q == inst.goals).sum()), remaining=sum_dist(inst, q),
                               search_calls=0, lacam_calls=1, lacam_nodes=nodes, compute_seconds=elapsed)
                          for q in path[1:])
        ft, mk = flowtime([np.array(f["positions"]) for f in frames], inst.goals)
        result = dict(success=path is not None, runtime=elapsed, n_cspibt=0, n_lacam=1,
                      lacam_nodes=nodes, flowtime=ft, makespan=mk, steps=len(frames)-1)
    else:
        controller = Controller(POLICY, "gate" if mode == "hybrid" else mode,
                                gate=gate, use_lacam=mode == "hybrid",
                                hold=settings.get("hold", 1), seed=seed)
        result = controller.run(inst, trace=frames)
    # Verify every displayed transition, including adjacency and obstacle constraints.
    for left, right in zip(frames, frames[1:]):
        a, b = np.array(left["positions"]), np.array(right["positions"])
        if not is_valid_step(a, b) or not (inst.grid.nbr[a] == b[:, None]).any(axis=1).all():
            raise RuntimeError("The controller produced an invalid transition; replay was stopped.")
    lb = sum_dist(inst, inst.starts)
    # Failed, truncated trajectories do not constitute solutions with comparable quality.
    result.update(sum_lb=lb, subopt=result["flowtime"] / max(lb, 1) if result["success"] else None,
                  solution_flowtime=result["flowtime"] if result["success"] else None,
                  solution_makespan=result["makespan"] if result["success"] else None,
                  search_rate=result["n_cspibt"] / max(len(frames)-1, 1),
                  arrived=frames[-1]["arrived"], remaining=frames[-1]["remaining"],
                  collisions=0)
    return dict(seed=seed, count=count, mode=mode, width=32, height=32,
                free=inst.grid.free.astype(int).tolist(), goals=inst.goals.tolist(),
                frames=frames, result=result, threshold=float(gate.tau1))


def benchmark(seed, count, mode):
    """Fresh timed runs; trajectory cache hits are never used as benchmark timings."""
    if mode not in ("pibt", "lacam"):
        inst = make_instance(seed, count)
        POLICY.probs(inst.grid, ObsBuilder(inst), inst.starts)  # untimed inference warm-up
    runs = [simulate.__wrapped__(seed, count, mode)["result"] for _ in range(3)]
    times = [r["runtime"] for r in runs]
    return dict(seed=seed, count=count, mode=mode, repeats=3, result=runs[0],
                runtime_median=statistics.median(times), runtime_min=min(times), runtime_max=max(times),
                solved_runs=sum(r["success"] for r in runs),
                stable_outcomes=all({k: v for k, v in r.items() if k != "runtime"} ==
                                    {k: v for k, v in runs[0].items() if k != "runtime"} for r in runs))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in ("/api/simulate", "/api/benchmark"):
            return self.send_json({"error": "Not found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            args = json.loads(self.rfile.read(size))
            seed, count, mode = int(args["seed"]), int(args["count"]), args["mode"]
            if not 0 <= seed <= 2**32 - 1 or count not in (8, 16, 32, 64, 96, 128, 160, 192):
                raise ValueError("Choose a valid seed and agent count")
            if mode not in ("shield", "pibt", "cspibt", "gate", "hybrid", "lacam"):
                raise ValueError("Unknown controller")
            with LOCK:
                result = (benchmark if self.path == "/api/benchmark" else simulate)(seed, count, mode)
            self.send_json(result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_GET(self):
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        entry = files.get(urlparse(self.path).path)
        if not entry:
            return self.send_json({"error": "Not found"}, 404)
        data = (ASSETS / entry[0]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", entry[1])
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(1)
    POLICY = Policy.load(ROOT / "artifacts/policy.pt")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MAPF visual simulator: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Check model loading, replay safety, metric consistency and saved configuration."""
import json

import numpy as np
import torch

from . import demo
from .policy import Policy


def main():
    torch.set_num_threads(1)
    demo.POLICY = Policy.load(demo.ROOT / "artifacts/policy.pt")
    cfg = json.loads((demo.ROOT / "artifacts/r_order.json").read_text())
    assert cfg == {"R": 0.5, "sample": True}, "Restore the submitted sampled-order configuration."
    initial = None
    for mode in ("shield", "pibt", "cspibt", "gate", "hybrid", "lacam"):
        run = demo.simulate(42, 16, mode)
        if initial is None:
            initial = (run["free"], run["goals"], run["frames"][0]["positions"])
        assert initial == (run["free"], run["goals"], run["frames"][0]["positions"])
        end, result = run["frames"][-1], run["result"]
        assert end["search_calls"] == result["n_cspibt"]
        assert end["lacam_calls"] == result["n_lacam"]
        if result["success"]:
            assert np.array_equal(end["positions"], run["goals"])
            assert result["subopt"] >= 1
        else:
            assert result["subopt"] is None
        print(f"PASS {mode}: {len(run['frames']) - 1} steps; solved={result['success']}")
    fallback = demo.simulate(43, 128, "hybrid")
    assert fallback["result"]["success"] and fallback["result"]["n_lacam"] >= 1
    assert fallback["frames"][-1]["lacam_calls"] == fallback["result"]["n_lacam"]
    print("PASS fallback: valid full-plan replay and consistent call counters")
    print("All smoke checks passed. Replay generation also validates adjacency and collision constraints.")


if __name__ == "__main__":
    main()

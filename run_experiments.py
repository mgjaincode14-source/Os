import csv
import copy
from memsim.allocators import NativeAllocator, FusionPoolAllocator, PageBasedAdaptiveAllocator
from memsim.workload import make_training_workload, make_inference_workload
from memsim.simulate import run_simulation

CAPACITY = 20000  # MB, arbitrary but fixed across all runs for fairness

def build_allocator(name):
    if name == "native":
        return NativeAllocator(CAPACITY)
    if name == "fusion_pool":
        return FusionPoolAllocator(CAPACITY, coalesce_every=8)
    if name == "page_adaptive":
        return PageBasedAdaptiveAllocator(CAPACITY, page_size_mb=2, adaptive=True)
    if name == "page_fixed":
        return PageBasedAdaptiveAllocator(CAPACITY, page_size_mb=2, adaptive=False)
    raise ValueError(name)


WORKLOADS = {
    "training": lambda seed: make_training_workload(n_requests=6, seed=seed),
    "inference_steady_mixed": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="steady", length_mix="mixed", seed=seed),
    "inference_bursty_mixed": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="bursty", length_mix="mixed", seed=seed),
    "inference_steady_short": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="steady", length_mix="short", seed=seed),
    "inference_bursty_long": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="bursty", length_mix="long", seed=seed),
}

ALLOCATORS = ["native", "fusion_pool", "page_adaptive"]
SEEDS = [0, 1, 2]  # repeat runs to reduce measurement noise (methodology step 5)


def main():
    rows = []
    for wl_name, wl_fn in WORKLOADS.items():
        for alloc_name in ALLOCATORS:
            for seed in SEEDS:
                reqs = wl_fn(seed)
                alloc = build_allocator(alloc_name)
                res = run_simulation(alloc, reqs, alloc_name, wl_name)
                row = res.summary()
                row["seed"] = seed
                rows.append(row)
                print(f"{wl_name:24s} {alloc_name:14s} seed={seed}  "
                      f"frag={row['avg_fragmentation']:.3f}  "
                      f"lat/alloc={row['avg_latency_per_alloc']:.2f}  "
                      f"failed={row['failed_allocs']}")

    fieldnames = list(rows[0].keys())
    with open("/home/claude/results_main.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved /home/claude/results_main.csv")


if __name__ == "__main__":
    main()

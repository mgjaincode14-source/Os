import csv
from memsim.allocators import PageBasedAdaptiveAllocator
from memsim.workload import make_inference_workload
from memsim.simulate import run_simulation

CAPACITY = 20000

WORKLOADS = {
    "inference_steady_mixed": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="steady", length_mix="mixed", seed=seed),
    "inference_bursty_mixed": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="bursty", length_mix="mixed", seed=seed),
    "inference_steady_short": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="steady", length_mix="short", seed=seed),
    "inference_bursty_long": lambda seed: make_inference_workload(
        duration_ticks=1000, arrival_pattern="bursty", length_mix="long", seed=seed),
}

SEEDS = [0, 1, 2]


def main():
    rows = []
    for wl_name, wl_fn in WORKLOADS.items():
        for adaptive in [True, False]:
            for seed in SEEDS:
                reqs = wl_fn(seed)
                alloc = PageBasedAdaptiveAllocator(CAPACITY, page_size_mb=2, adaptive=adaptive)
                res = run_simulation(alloc, reqs, f"page_{'adaptive' if adaptive else 'fixed'}", wl_name)
                row = res.summary()
                row["seed"] = seed
                rows.append(row)
                print(f"{wl_name:24s} adaptive={adaptive!s:5s} seed={seed}  "
                      f"frag={row['avg_fragmentation']:.3f}  "
                      f"lat/alloc={row['avg_latency_per_alloc']:.2f}  "
                      f"failed={row['failed_allocs']}")

    fieldnames = list(rows[0].keys())
    with open("/home/claude/results_sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved /home/claude/results_sensitivity.csv")


if __name__ == "__main__":
    main()

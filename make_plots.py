import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/results_main.csv")
agg = df.groupby(["workload", "allocator"]).agg(
    avg_fragmentation=("avg_fragmentation", "mean"),
    avg_latency_per_alloc=("avg_latency_per_alloc", "mean"),
    throughput_per_tick=("throughput_per_tick", "mean"),
    failed_allocs=("failed_allocs", "sum"),
).reset_index()

print(agg.to_string(index=False))
agg.to_csv("/home/claude/agg_main.csv", index=False)

workloads = [w for w in agg["workload"].unique() if w != "training"]  # separate training (different scale)
allocators = ["native", "fusion_pool", "page_adaptive"]
colors = {"native": "#888888", "fusion_pool": "#4C72B0", "page_adaptive": "#DD8452"}

# ---- Figure 1: fragmentation across inference workloads ----
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(workloads))
width = 0.25
for i, alloc in enumerate(allocators):
    vals = [agg[(agg.workload == w) & (agg.allocator == alloc)]["avg_fragmentation"].values[0]
            for w in workloads]
    ax.bar([xi + i * width for xi in x], vals, width, label=alloc, color=colors[alloc])
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(workloads, rotation=20, ha="right")
ax.set_ylabel("Average fragmentation ratio")
ax.set_title("Fragmentation ratio by allocator across inference workloads")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/fig_fragmentation.png", dpi=150)

# ---- Figure 2: latency per alloc across inference workloads ----
fig, ax = plt.subplots(figsize=(9, 5))
for i, alloc in enumerate(allocators):
    vals = [agg[(agg.workload == w) & (agg.allocator == alloc)]["avg_latency_per_alloc"].values[0]
            for w in workloads]
    ax.bar([xi + i * width for xi in x], vals, width, label=alloc, color=colors[alloc])
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(workloads, rotation=20, ha="right")
ax.set_ylabel("Avg modeled latency per allocation (cost units)")
ax.set_title("Allocator overhead by workload (lower is better)")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/fig_latency.png", dpi=150)

# ---- Figure 3: training workload (separate, huge scale for page_adaptive) ----
train = agg[agg.workload == "training"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
ax[0].bar(train["allocator"], train["avg_fragmentation"], color=[colors[a] for a in train["allocator"]])
ax[0].set_title("Training workload: fragmentation")
ax[0].set_ylabel("Fragmentation ratio")
ax[1].bar(train["allocator"], train["avg_latency_per_alloc"], color=[colors[a] for a in train["allocator"]])
ax[1].set_title("Training workload: latency/alloc\n(log scale)")
ax[1].set_yscale("log")
fig.tight_layout()
fig.savefig("/home/claude/fig_training.png", dpi=150)

# ---- Figure 4: sensitivity (adaptive vs fixed) ----
sens = pd.read_csv("/home/claude/results_sensitivity.csv")
sagg = sens.groupby(["workload", "allocator"]).agg(
    avg_fragmentation=("avg_fragmentation", "mean"),
    failed_allocs=("failed_allocs", "sum"),
).reset_index()
print("\nSensitivity:\n", sagg.to_string(index=False))
sagg.to_csv("/home/claude/agg_sensitivity.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
wl2 = sagg["workload"].unique()
for i, alloc in enumerate(["page_adaptive", "page_fixed"]):
    vals = [sagg[(sagg.workload == w) & (sagg.allocator == alloc)]["failed_allocs"].values[0]
            if not sagg[(sagg.workload == w) & (sagg.allocator == alloc)].empty else 0
            for w in wl2]
    ax.bar([xi + i * 0.35 for xi in range(len(wl2))], vals, 0.35, label=alloc)
ax.set_xticks([xi + 0.175 for xi in range(len(wl2))])
ax.set_xticklabels(wl2, rotation=20, ha="right")
ax.set_ylabel("Total failed allocations (3 seeds summed)")
ax.set_title("Adaptive vs fixed page size: allocation failures under pressure")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/fig_sensitivity.png", dpi=150)

print("\nSaved 4 figures.")

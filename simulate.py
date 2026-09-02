"""
Runs a given allocator against a given workload and collects metrics.

LATENCY MODEL (explicit, documented assumption):
We do not have access to real NPU/GPU hardware timing, so allocator
latency is modeled with a simple, transparent cost model rather than
measured on real silicon:

    - reserving brand-new backing memory (aclrtReserveMemAddress-style)
      costs RESERVE_COST "time units" -- the base paper explicitly
      notes this step "actually takes quite a long time" (Sec II),
      so new-block reservation is deliberately the most expensive op.
    - reusing an already-reserved free block/page costs REUSE_COST
      (cheap -- just bookkeeping).
    - a split costs SPLIT_COST, a coalesce/stitch costs STITCH_COST.

These constants are the same across all three allocators, so the
comparison is fair: differences in total latency come purely from how
many of each operation an allocator strategy performs, not from
differing assumptions per allocator. Absolute latency numbers are
therefore illustrative; the *relative* comparison across allocators
and workloads is the meaningful research result.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from .workload import Request

RESERVE_COST = 5.0
REUSE_COST = 0.2
SPLIT_COST = 0.3
STITCH_COST = 0.4


@dataclass
class TickSnapshot:
    tick: int
    reserved: float
    active: float
    fragmentation: float
    live_requests: int


@dataclass
class RunResult:
    allocator_name: str
    workload_name: str
    snapshots: List[TickSnapshot] = field(default_factory=list)
    completed: int = 0
    failed_allocs: int = 0
    total_alloc_ops: int = 0
    total_free_ops: int = 0
    total_split_ops: int = 0
    total_stitch_ops: int = 0
    total_reserve_ops: int = 0
    total_latency: float = 0.0
    makespan: int = 0

    def summary(self) -> Dict:
        avg_frag = (sum(s.fragmentation for s in self.snapshots) / len(self.snapshots)
                    if self.snapshots else float("nan"))
        peak_frag = max((s.fragmentation for s in self.snapshots), default=float("nan"))
        avg_util = 1 - avg_frag if self.snapshots else float("nan")
        throughput = self.completed / self.makespan if self.makespan else 0.0
        avg_latency_per_alloc = (self.total_latency / self.total_alloc_ops
                                  if self.total_alloc_ops else 0.0)
        return {
            "allocator": self.allocator_name,
            "workload": self.workload_name,
            "avg_fragmentation": round(avg_frag, 4),
            "peak_fragmentation": round(peak_frag, 4),
            "avg_utilization": round(avg_util, 4),
            "completed": self.completed,
            "failed_allocs": self.failed_allocs,
            "throughput_per_tick": round(throughput, 4),
            "total_latency": round(self.total_latency, 1),
            "avg_latency_per_alloc": round(avg_latency_per_alloc, 3),
            "split_ops": self.total_split_ops,
            "stitch_ops": self.total_stitch_ops,
            "reserve_ops": self.total_reserve_ops,
        }


def _pool_snapshot(allocator, tick, live_count) -> TickSnapshot:
    # PageBasedAdaptiveAllocator tracks active/reserved slightly
    # differently (page bookkeeping) but pool.reserved/active still
    # reflects true totals since pages are Blocks in the same pool.
    reserved = allocator.pool.reserved
    active = allocator.pool.active
    frag = 1 - (active / reserved) if reserved > 0 else 0.0
    return TickSnapshot(tick=tick, reserved=reserved, active=active,
                         fragmentation=frag, live_requests=live_count)


def run_simulation(allocator, requests: List[Request],
                    allocator_name: str, workload_name: str,
                    snapshot_every: int = 1) -> RunResult:
    result = RunResult(allocator_name=allocator_name, workload_name=workload_name)

    # index requests by arrival tick
    by_arrival: Dict[int, List[Request]] = {}
    for r in requests:
        by_arrival.setdefault(r.arrival_tick, []).append(r)

    max_tick = max((r.arrival_tick + r.steps for r in requests), default=0) + 1

    # track remaining steps / whether initial alloc happened
    remaining_steps = {r.id: r.steps for r in requests}
    initialized = set()
    req_by_id = {r.id: r for r in requests}
    reserve_ops_before = 0

    live_ids = set()

    for tick in range(max_tick):
        # 1. new arrivals: initial allocation
        for r in by_arrival.get(tick, []):
            ok = allocator.allocate(r.id, r.initial_size, tick)
            result.total_alloc_ops += 1
            live_ids.add(r.id)
            if not ok:
                result.failed_allocs += 1
                live_ids.discard(r.id)
                remaining_steps.pop(r.id, None)
            initialized.add(r.id)

        # 2. growth step for all live requests (token generation)
        finished_now = []
        for rid in list(live_ids):
            req = req_by_id[rid]
            if req.growth_per_step > 0:
                ok = allocator.allocate(rid, req.growth_per_step, tick)
                result.total_alloc_ops += 1
                if not ok:
                    result.failed_allocs += 1
            remaining_steps[rid] -= 1
            if remaining_steps[rid] <= 0:
                finished_now.append(rid)

        # 3. free finished requests
        for rid in finished_now:
            allocator.free(rid, tick)
            result.total_free_ops += 1
            result.completed += 1
            live_ids.discard(rid)

        # 4. snapshot
        if tick % snapshot_every == 0:
            result.snapshots.append(_pool_snapshot(allocator, tick, len(live_ids)))

    result.makespan = max_tick
    result.total_split_ops = allocator.pool.split_ops
    result.total_stitch_ops = allocator.pool.stitch_ops
    result.total_reserve_ops = allocator.pool.address_reserve_ops

    # synthetic latency total from operation costs
    result.total_latency = (
        result.total_reserve_ops * RESERVE_COST +
        result.total_split_ops * SPLIT_COST +
        result.total_stitch_ops * STITCH_COST +
        max(0, result.total_alloc_ops - result.total_reserve_ops) * REUSE_COST
    )

    return result

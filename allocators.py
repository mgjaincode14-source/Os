"""
Three allocator strategies under test:

1. NativeAllocator        - naive best-fit, no coalescing, no rounding.
                            Stand-in for the "native" baseline in the
                            base paper.

2. FusionPoolAllocator    - re-implementation of the base paper's
                            training-oriented strategy: best-fit +
                            round-size alignment + periodic coalescing
                            ("fusion pool"). Designed for few, large,
                            predictable allocations (training). This is
                            the allocator we re-test under INFERENCE
                            conditions, which the base paper never did.

3. PageBasedAdaptiveAllocator - our proposed allocator for inference
                            serving: fixed-size "pages" (like OS virtual
                            memory pages / vLLM's PagedAttention) that
                            are reused across many small, short-lived
                            KV-cache requests, with a controller that
                            adapts the page size based on recent traffic
                            (adaptive vs a fixed constant is what our
                            sensitivity analysis isolates).
"""

import math
from collections import deque
from typing import Dict, Optional
from .core import MemoryPool, Block


def _round_size_paper_style(x: int, distance: int = 16) -> int:
    """Reimplementation of the base paper's rounding scheme (Sec II,
    'Round Size Optimization'). Rounds x up to a grid point between
    the previous and next power of two, spaced by `distance` MB,
    instead of jumping straight to the next power of two."""
    if x <= 0:
        return 0
    lower = 2 ** math.floor(math.log2(x))
    upper = 2 ** math.ceil(math.log2(x))
    if lower == upper:
        return x
    # smallest multiple of `distance` steps above lower that covers x
    n = math.ceil((x - lower) / distance)
    rounded = lower + n * distance
    return min(rounded, upper)


class BaseAllocator:
    name = "base"

    def __init__(self, capacity_mb: int):
        self.pool = MemoryPool(capacity_mb)
        # IMPORTANT: a request's KV-cache grows over many allocate()
        # calls (one per generated token / step). We must track ALL
        # blocks ever handed to a request, not just the latest one,
        # or earlier blocks leak (never freed) -- this was a bug in
        # an earlier version of this simulator.
        self.live: Dict[int, list] = {}   # request_id -> list[Block]
        self.alloc_ops = 0
        self.free_ops = 0
        self.failed_allocs = 0

    def allocate(self, request_id: int, size_mb: int, now: float) -> bool:
        raise NotImplementedError

    def free(self, request_id: int, now: float):
        blocks = self.live.pop(request_id, [])
        for blk in blocks:
            self.pool.free_block(blk)
        if blocks:
            self.free_ops += 1

    # -- shared best-fit search over free blocks ---------------------
    def _find_best_fit(self, size: int) -> Optional[Block]:
        candidates = [b for b in self.pool.blocks if b.free and b.size >= size]
        if not candidates:
            return None
        return min(candidates, key=lambda b: b.size)


class NativeAllocator(BaseAllocator):
    """Naive best-fit allocator, no rounding, no coalescing.
    Stand-in for the 'native' (unoptimized) baseline."""
    name = "native"

    def allocate(self, request_id: int, size_mb: int, now: float) -> bool:
        self.alloc_ops += 1
        blk = self._find_best_fit(size_mb)
        if blk is None:
            blk = self.pool.reserve_new_block(size_mb)
            if blk is None:
                self.failed_allocs += 1
                return False
        used = self.pool.split(blk, size_mb)
        used.owner = request_id
        self.live.setdefault(request_id, []).append(used)
        return True


class FusionPoolAllocator(BaseAllocator):
    """Training-oriented allocator re-implementing the base paper's
    fusion-pool + round-size strategy. Coalesces free blocks
    periodically and rounds allocation sizes to reduce misalignment.
    This is well-suited to FEW, LARGE, PREDICTABLE allocations
    (training) -- our experiment re-tests it under the opposite
    conditions (inference: MANY, SMALL, BURSTY allocations)."""
    name = "fusion_pool"

    def __init__(self, capacity_mb: int, coalesce_every: int = 8):
        super().__init__(capacity_mb)
        self._op_count = 0
        self.coalesce_every = coalesce_every

    def allocate(self, request_id: int, size_mb: int, now: float) -> bool:
        self.alloc_ops += 1
        rounded = _round_size_paper_style(size_mb)
        blk = self._find_best_fit(rounded)
        if blk is None:
            blk = self.pool.reserve_new_block(rounded)
            if blk is None:
                # try un-rounded size before giving up (paper's
                # allocator would still try to serve the real need)
                blk = self.pool.reserve_new_block(size_mb)
                if blk is None:
                    self.failed_allocs += 1
                    return False
                rounded = size_mb
        used = self.pool.split(blk, min(rounded, blk.size))
        used.owner = request_id
        self.live.setdefault(request_id, []).append(used)

        self._op_count += 1
        if self._op_count % self.coalesce_every == 0:
            self.pool.coalesce_adjacent()
        return True

    def free(self, request_id: int, now: float):
        super().free(request_id, now)
        self._op_count += 1
        if self._op_count % self.coalesce_every == 0:
            self.pool.coalesce_adjacent()


class PageBasedAdaptiveAllocator(BaseAllocator):
    """
    Proposed allocator for INFERENCE serving.

    Memory is carved into fixed-size "pages" (like OS virtual memory
    pages, or vLLM's PagedAttention KV-cache blocks). A request's
    KV-cache is composed of as many pages as it needs, grown
    incrementally as more tokens are generated, and pages are
    returned to a free-list (not merged/coalesced) the moment a
    request finishes -- avoiding the expensive stitching/merging
    that the training-oriented allocator relies on, which is wasted
    effort for many-small-short-lived requests.

    The `adaptive` flag controls whether the page size is tuned
    automatically from recent traffic (moving average of request
    growth-per-step) or held at a fixed constant -- this is exactly
    what the sensitivity analysis (methodology step 7) isolates.
    """
    name = "page_adaptive"

    def __init__(self, capacity_mb: int, page_size_mb: int = 2,
                 adaptive: bool = True, window: int = 50):
        super().__init__(capacity_mb)
        self.base_page_size = page_size_mb
        self.page_size = page_size_mb
        self.adaptive = adaptive
        self.window = window
        self._recent_growth = deque(maxlen=window)
        self.free_pages = deque()   # list of Block (all same current page_size ideally)
        self.pages_per_request: Dict[int, list] = {}

    # -- adaptive controller ------------------------------------------
    def _maybe_retune_page_size(self):
        if not self.adaptive or len(self._recent_growth) < self.window:
            return
        avg_growth = sum(self._recent_growth) / len(self._recent_growth)
        # snap to nearest power-of-two-ish granularity, bounded
        candidate = max(1, min(16, round(avg_growth)))
        if candidate != self.page_size:
            self.page_size = candidate
            # existing free pages of the old size are still usable;
            # we simply allocate new pages at the new size going
            # forward (avoids costly repacking of live pages)

    def _get_page(self) -> Block:
        if self.free_pages:
            return self.free_pages.popleft()
        blk = self.pool.reserve_new_block(self.page_size)
        if blk is None:
            # pool exhausted at current page size; try shrinking once
            shrink = max(1, self.page_size // 2)
            blk = self.pool.reserve_new_block(shrink)
        return blk

    def allocate(self, request_id: int, size_mb: int, now: float) -> bool:
        """Allocate (or grow) a request's KV-cache by size_mb this call."""
        self.alloc_ops += 1
        self._recent_growth.append(size_mb)
        self._maybe_retune_page_size()

        needed_pages = math.ceil(size_mb / max(1, self.page_size))
        pages = self.pages_per_request.setdefault(request_id, [])
        for _ in range(needed_pages):
            blk = self._get_page()
            if blk is None:
                self.failed_allocs += 1
                return False
            blk.free = False
            blk.owner = request_id
            pages.append(blk)
        return True

    def free(self, request_id: int, now: float):
        pages = self.pages_per_request.pop(request_id, [])
        for p in pages:
            p.free = True
            p.owner = None
            self.free_pages.append(p)
        self.free_ops += 1
        # keep the pool's `live` view roughly consistent for reporting
        self.live.pop(request_id, None)

    # override active/reserved reporting to use page bookkeeping
    @property
    def active_mb(self):
        return sum(p.size for pages in self.pages_per_request.values() for p in pages)

    @property
    def reserved_mb(self):
        return self.pool.reserved

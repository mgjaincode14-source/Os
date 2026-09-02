"""
Core primitives for simulating virtual-memory-style allocators for
LLM KV-cache workloads (training-style vs inference-style).

This models memory in abstract "MB" units. It captures the concepts
used in the base paper (reserved vs active memory, fragmentation
ratio, block splitting/fusing) without needing real NPU/GPU hardware.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import itertools

_id_counter = itertools.count(1)


@dataclass
class Block:
    """A contiguous chunk of reserved virtual memory."""
    start: int          # offset within the pool (MB)
    size: int           # size in MB
    free: bool = True
    owner: Optional[int] = None  # request id currently using it
    id: int = field(default_factory=lambda: next(_id_counter))


class MemoryPool:
    """
    A simplified virtual memory pool. Tracks a fixed total capacity
    and a list of contiguous blocks (some free, some allocated).

    reserved memory  = total size of blocks that exist (allocated to
                        the pool, whether currently in use or not)
    active memory    = total size of blocks currently "in use" by a
                        live request
    fragmentation    = 1 - (active / reserved), matching the base
                        paper's definition (Section III-A)
    """

    def __init__(self, capacity_mb: int):
        self.capacity = capacity_mb
        self.blocks: List[Block] = []
        self.split_ops = 0
        self.stitch_ops = 0   # merge/fuse operations
        self.address_reserve_ops = 0  # analogous to aclrtReserveMemAddress calls

    # ---- accounting -----------------------------------------------
    @property
    def reserved(self) -> int:
        return sum(b.size for b in self.blocks)

    @property
    def active(self) -> int:
        return sum(b.size for b in self.blocks if not b.free)

    @property
    def utilization_ratio(self) -> float:
        r = self.reserved
        return (self.active / r) if r > 0 else 1.0

    @property
    def fragmentation_ratio(self) -> float:
        return 1.0 - self.utilization_ratio

    def free_capacity(self) -> int:
        return self.capacity - self.reserved

    # ---- low level ops (mirroring aclrtReserveMemAddress etc.) -----
    def reserve_new_block(self, size: int) -> Optional[Block]:
        """Create a brand-new block backed by fresh capacity."""
        if self.free_capacity() < size:
            return None
        start = self.reserved
        blk = Block(start=start, size=size, free=True)
        self.blocks.append(blk)
        self.address_reserve_ops += 1
        return blk

    def split(self, block: Block, needed: int) -> Block:
        """Split a free block into a used piece (needed) and a
        remaining free piece. Returns the used piece."""
        assert block.free and block.size >= needed
        self.split_ops += 1
        if block.size == needed:
            block.free = False
            return block
        used = Block(start=block.start, size=needed, free=False)
        remainder = Block(start=block.start + needed, size=block.size - needed, free=True)
        idx = self.blocks.index(block)
        self.blocks[idx:idx + 1] = [used, remainder]
        return used

    def free_block(self, block: Block):
        block.free = True
        block.owner = None

    def coalesce_adjacent(self):
        """Merge adjacent free blocks (classic external-fragmentation
        fix, analogous to the paper's 'fusion pool')."""
        if not self.blocks:
            return
        self.blocks.sort(key=lambda b: b.start)
        merged: List[Block] = []
        for b in self.blocks:
            if merged and merged[-1].free and b.free and \
               merged[-1].start + merged[-1].size == b.start:
                merged[-1] = Block(start=merged[-1].start,
                                    size=merged[-1].size + b.size,
                                    free=True)
                self.stitch_ops += 1
            else:
                merged.append(b)
        self.blocks = merged

    def release_free_tail_blocks(self):
        """Drop fully-free blocks entirely (returns capacity to the
        'unreserved' pool), analogous to aclrtFreePhysical /
        aclrtReleaseMemAddress on an empty segment."""
        self.blocks = [b for b in self.blocks if not (b.free and b.owner is None and b.size > 0 and b in self.blocks and self._is_reclaimable(b))]

    def _is_reclaimable(self, b: Block) -> bool:
        return b.free

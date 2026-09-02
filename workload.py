"""
Synthetic workload generator.

TRAINING-style workload: few, large, highly predictable allocations
(mirrors the base paper's setting -- e.g. LLaMA2/OPT fine-tuning
epochs: reserve a big tensor, hold it for a long time, free at epoch
boundary).

INFERENCE-style workload: many concurrent user sessions, each growing
its KV-cache by a small amount per generated token, arriving either
steadily (Poisson-like) or in bursts, with short/medium/long total
output lengths, and either fixed-size batching or continuous
("rolling") admission of new requests.
"""

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Request:
    id: int
    arrival_tick: int
    steps: int            # number of token-generation steps (ticks it stays alive)
    growth_per_step: int  # MB added to its KV-cache each step
    initial_size: int     # MB needed at arrival (prompt KV-cache)


def make_training_workload(n_requests=6, seed=0) -> List[Request]:
    """Few, large, predictable allocations held for a long time --
    e.g. one allocation per training epoch."""
    rng = random.Random(seed)
    reqs = []
    tick = 0
    for i in range(n_requests):
        size = rng.choice([2048, 4096, 6144])  # large, "epoch tensor" sized (MB)
        reqs.append(Request(id=i, arrival_tick=tick, steps=40,
                             growth_per_step=0, initial_size=size))
        tick += 45  # next epoch starts right after (predictable cadence)
    return reqs


def make_inference_workload(duration_ticks=800, arrival_pattern="steady",
                             length_mix="mixed", seed=0) -> List[Request]:
    """
    Many small, short-lived requests representing chatbot-style
    inference sessions.

    arrival_pattern: "steady" (roughly uniform arrivals) or "bursty"
                      (long quiet periods punctuated by traffic spikes)
    length_mix:       "short", "medium", "long", or "mixed"
    """
    rng = random.Random(seed)
    reqs = []
    rid = 0

    length_choices = {
        "short":  (10, 30),
        "medium": (40, 90),
        "long":   (100, 200),
    }

    def sample_steps():
        if length_mix == "mixed":
            key = rng.choice(["short", "medium", "long"])
        else:
            key = length_mix
        lo, hi = length_choices[key]
        return rng.randint(lo, hi)

    tick = 0
    while tick < duration_ticks:
        if arrival_pattern == "steady":
            # ~1 arrival every 2-4 ticks
            gap = rng.randint(2, 4)
            batch_size = 1
        else:  # bursty
            # long quiet gaps, then a burst of several requests at once
            if rng.random() < 0.15:
                gap = rng.randint(1, 2)
                batch_size = rng.randint(5, 12)
            else:
                gap = rng.randint(5, 10)
                batch_size = 1

        tick += gap
        for _ in range(batch_size):
            steps = sample_steps()
            # KV-cache grows a small amount per generated token
            growth = rng.randint(1, 3)   # MB per step
            initial = rng.randint(2, 8)  # MB for the prompt itself
            reqs.append(Request(id=rid, arrival_tick=tick, steps=steps,
                                 growth_per_step=growth, initial_size=initial))
            rid += 1
        if tick >= duration_ticks:
            break

    return reqs

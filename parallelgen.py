# ParallelGen - Parallel matrix generation across worker threads
# Part of the Kapildhinwa2010-dev Toolkit.

"""Thread-parallel generation of independent recipe blocks.

Lineage: DNA_DYNIMO generators. This repo owns the PARALLEL mode:
generate several independent matrices from one recipe across worker
threads, plus a small timing report.

HONEST LIMITATION: Python threads + the GIL mean speedups come only
from NumPy releasing the GIL inside BLAS; for small matrices the
thread overhead can make parallel SLOWER than sequential. Measure with
parallel_report() before assuming a win.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["GenRecipe", "generate", "generate_parallel", "parallel_report"]


class GenRecipe:
    """A seeded low-rank recipe: rows x cols output from rank x (rows+cols) latents."""

    def __init__(self, rows: int, cols: int, rank: int = 8, seed: int = 0) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rows = rows
        self.cols = cols
        self.rank = min(rank, min(rows, cols))
        self.seed = seed

    def _latent(self, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + seed_offset)
        left = rng.standard_normal((self.rows, self.rank))
        right = rng.standard_normal((self.rank, self.cols))
        return (left @ right) / math.sqrt(self.rank)


def generate(recipe: GenRecipe) -> np.ndarray:
    """The canonical matrix of the recipe (seed + 0)."""
    return recipe._latent(0)

def generate_parallel(recipe: GenRecipe, blocks: int = 4, workers: int = 2) -> list:
    """Generate ``blocks`` independent matrices across worker threads.

    Block i uses seed_offset i, so results are independent of the worker
    count - only the wall-clock time changes.
    """
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(recipe._latent, range(blocks)))


def parallel_report(recipe: GenRecipe, blocks: int = 4, workers: int = 2) -> dict:
    """Timing comparison: sequential vs threaded generation of ``blocks``.

    Returns {"sequential_seconds", "parallel_seconds", "speedup", "blocks",
    "workers"}. speedup < 1 means threading HURT on this machine/size -
    a common outcome for small matrices (GIL + thread overhead).
    """
    if blocks < 1 or workers < 1:
        raise ValueError("blocks and workers must be >= 1")
    start = time.perf_counter()
    for i in range(blocks):
        recipe._latent(i)
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    generate_parallel(recipe, blocks=blocks, workers=workers)
    parallel = time.perf_counter() - start

    return {
        "sequential_seconds": sequential,
        "parallel_seconds": parallel,
        "speedup": (sequential / parallel) if parallel > 0 else float("inf"),
        "blocks": blocks,
        "workers": workers,
    }
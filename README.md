# ParallelGen

[![CI](https://github.com/kapildhinwa2010-dev/ParallelGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ParallelGen/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/ParallelGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ParallelGen/actions/workflows/codeql.yml)


Parallel matrix generation across worker threads, with an honest sequential-vs-threaded timing report.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

ParallelGen owns the PARALLEL mode of the DNA_DYNIMO generator
family: generate_parallel() produces independent recipe matrices
across worker threads (block i uses seed_offset i, so results are
independent of worker count); parallel_report() times sequential vs
threaded. The HONEST LIMITATION (also in the docstring): Python
threads + the GIL mean speedups come only from NumPy releasing the
GIL inside BLAS; for small matrices threading can be SLOWER.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from parallelgen import GenRecipe, generate_parallel, parallel_report

recipe = GenRecipe(rows=1024, cols=1024, rank=8, seed=0)
blocks = generate_parallel(recipe, blocks=4, workers=2)
print(parallel_report(recipe, blocks=4, workers=2)["speedup"])
```

## API

- GenRecipe(rows, cols, rank, seed)
- generate(recipe) - the canonical matrix
- generate_parallel(recipe, blocks, workers)
- parallel_report(recipe, blocks, workers) - {sequential_seconds,
  parallel_seconds, speedup, blocks, workers}

## Honest results and limitations

- speedup < 1 is a common outcome for small matrices (GIL + thread
  overhead): measure with parallel_report() before assuming a win.
- Threads, not processes: no multi-core gain outside GIL-releasing
  BLAS calls.

## License

MIT - see [LICENSE](LICENSE).

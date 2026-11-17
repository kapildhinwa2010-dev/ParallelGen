# ParallelGen

ParallelGen is a small Python library for generating seeded, low-rank NumPy matrices, either one at a time or as a set of independent blocks scheduled on worker threads. It also provides a basic sequential-versus-threaded timing report. The project metadata declares version `0.1.0` and Python `>=3.9`.

## Features

- `GenRecipe` describes a matrix shape, rank, and seed. The effective rank is capped at the smaller matrix dimension.
- `generate(recipe)` returns one matrix generated from the recipe's base seed.
- `generate_parallel(recipe, blocks, workers)` returns `blocks` matrices using consecutive seed offsets. Results are independent of the worker count.
- `parallel_report(recipe, blocks, workers)` compares one sequential generation loop with one threaded generation call and returns elapsed seconds, their ratio (`speedup`), and the block and worker counts.

The matrices are formed from random left and right factors and their product, scaled by the square root of the rank. This is a basic synthetic matrix generator; the repository does not implement a broader data-generation pipeline.

## Installation

The project declares a setuptools build backend and NumPy as its runtime dependency. From a checkout, install the project with:

```bash
python -m pip install .
```

For development and tests, install the listed requirements instead:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
from parallelgen import GenRecipe, generate, generate_parallel, parallel_report

recipe = GenRecipe(rows=128, cols=64, rank=8, seed=0)

matrix = generate(recipe)                 # shape: (128, 64)
blocks = generate_parallel(recipe, blocks=4, workers=2)
report = parallel_report(recipe, blocks=4, workers=2)

print(matrix.shape)
print([block.shape for block in blocks])
print(report)
```

`GenRecipe` requires positive row, column, and rank values. `generate_parallel` requires positive `blocks` and `workers`; `parallel_report` requires both to be positive as well. A rank larger than either matrix dimension is reduced to the smaller dimension.

## Tests and CI

Run the tests from the repository root with:

```bash
python -m pytest -q
```

On the reviewed repository commit, this command completed locally with **8 tests passed**. This is a local test result, not a claim that the latest GitHub Actions run passed. The CI workflow is configured to run pytest, compile the module and tests, and perform an import smoke test on Python 3.10, 3.11, and 3.12. The project metadata declares Python `>=3.9`; the workflow does not test 3.9.

## Limitations and status

This is a small `0.1.0` library, not a command-line application: there is no CLI entry point. Matrices are created in memory, and the implementation offers only thread-based parallel execution; it does not implement process workers, persistence, or a streaming interface. Threading is not guaranteed to be faster: Python thread overhead and the behavior of NumPy's underlying numerical libraries can affect the result. `parallel_report` performs a single timing for each method, so its ratio is a rough, machine- and workload-specific observation, not a benchmark guarantee. No benchmark results are claimed here.

## License

The repository's `LICENSE` file specifies the **MIT License**, with copyright `© 2026 Kapildhinwa2010-dev`. See [LICENSE](LICENSE) for the full terms.

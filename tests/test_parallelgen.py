import numpy as np
import pytest

from parallelgen import GenRecipe, generate, generate_parallel, parallel_report


def test_generate_shape():
    recipe = GenRecipe(rows=16, cols=32, rank=4, seed=0)
    assert generate(recipe).shape == (16, 32)


def test_generate_deterministic():
    recipe = GenRecipe(rows=8, cols=8, seed=3)
    assert np.array_equal(generate(recipe), generate(recipe))


def test_recipe_rejects_bad_args():
    with pytest.raises(ValueError):
        GenRecipe(rows=0, cols=8)


def test_parallel_returns_blocks():
    recipe = GenRecipe(rows=6, cols=6, seed=5)
    blocks = generate_parallel(recipe, blocks=3, workers=2)
    assert len(blocks) == 3
    assert all(b.shape == (6, 6) for b in blocks)
    assert not np.array_equal(blocks[0], blocks[1])


def test_parallel_rejects_bad_args():
    with pytest.raises(ValueError):
        generate_parallel(GenRecipe(rows=2, cols=2), blocks=0)
    with pytest.raises(ValueError):
        generate_parallel(GenRecipe(rows=2, cols=2), workers=0)


def test_parallel_independent_of_worker_count():
    recipe = GenRecipe(rows=6, cols=6, seed=5)
    two = generate_parallel(recipe, blocks=3, workers=2)
    four = generate_parallel(recipe, blocks=3, workers=4)
    assert all(np.array_equal(a, b) for a, b in zip(two, four))


# --- timing report (ParallelGen-specific) ---


def test_parallel_report_keys():
    report = parallel_report(GenRecipe(rows=32, cols=32, seed=1), blocks=2, workers=2)
    for key in ("sequential_seconds", "parallel_seconds", "speedup", "blocks", "workers"):
        assert key in report
    assert report["speedup"] > 0.0


def test_parallel_report_rejects_bad_args():
    with pytest.raises(ValueError):
        parallel_report(GenRecipe(rows=2, cols=2), blocks=0)
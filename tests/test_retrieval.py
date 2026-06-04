"""Unit tests for retrieval utilities (no vector DB required)."""

import pytest

from ingestion.utils import cosine_sim


class TestCosineSim:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_direction(self) -> None:
        assert cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_safe(self) -> None:
        assert cosine_sim([0, 0], [1, 1]) == pytest.approx(0.0, abs=1e-5)


class TestCosineSimProperties:
    def test_symmetry(self) -> None:
        a, b = [1.0, 2.0], [3.0, 4.0]
        assert cosine_sim(a, b) == pytest.approx(cosine_sim(b, a))

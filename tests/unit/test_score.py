from __future__ import annotations

import pytest

from llama_index.vector_stores.infino.base import _distance_to_similarity


def test_cosine_distance_maps_to_cosine_similarity():
    assert _distance_to_similarity("cosine", 0.0) == 1.0
    assert _distance_to_similarity("cosine", 1.0) == 0.0
    assert _distance_to_similarity("cosine", 2.0) == -1.0


def test_l2_is_monotonic_and_positive():
    assert _distance_to_similarity("l2", 0.0) == 1.0
    assert 0.0 < _distance_to_similarity("l2sq", 5.0) < 1.0


@pytest.mark.parametrize("metric", ["dot", "negdot"])
def test_dot_distance_negates_to_dot_product(metric):
    assert _distance_to_similarity(metric, -0.3) == 0.3


@pytest.mark.parametrize("metric", ["cosine", "l2", "l2sq", "dot", "negdot"])
def test_nearer_is_always_more_similar(metric):
    for near, far in [(0.1, 0.5), (1.2, 1.9)]:
        assert _distance_to_similarity(metric, near) > _distance_to_similarity(metric, far)

from __future__ import annotations

import math

import pytest

from llama_index.vector_stores.infino.base import _distance_to_similarity


def test_cosine_clamps_into_unit_interval():
    assert _distance_to_similarity("cosine", 0.0) == 1.0
    assert _distance_to_similarity("cosine", 1.0) == 0.0
    assert _distance_to_similarity("cosine", 1.5) == 0.0
    assert _distance_to_similarity("cosine", -0.2) == 1.0


def test_l2_is_monotonic_and_bounded_above_zero():
    near = _distance_to_similarity("l2", 0.0)
    far = _distance_to_similarity("l2", 5.0)
    assert near == 1.0
    assert 0.0 < far < near


@pytest.mark.parametrize("metric", ["dot", "negdot"])
def test_dot_metrics_pass_through_negated_distance(metric):
    assert _distance_to_similarity(metric, 0.3) == -0.3
    assert _distance_to_similarity(metric, -1.0) == 1.0


def test_orderings_match_intuition_for_supported_metrics():
    # Smaller distance must always map to larger similarity for ranking to work.
    for metric in ("cosine", "l2", "l2sq", "dot", "negdot"):
        nearer = _distance_to_similarity(metric, 0.1)
        farther = _distance_to_similarity(metric, 0.5)
        assert nearer >= farther, metric
        assert not math.isnan(nearer) and not math.isnan(farther)

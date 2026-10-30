"""Benjamini–Hochberg correction against hand-computed values."""

import pytest

from analytics.multiple_testing import benjamini_hochberg


def test_bh_hand_computed_four_tests():
    # sorted: 0.005(r1) 0.01(r2) 0.03(r3) 0.04(r4); m=4
    # raw p*m/r: 0.02, 0.02, 0.04, 0.04 ; running min from the top: 0.04, 0.04, 0.02, 0.02
    adj = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bh_monotonicity_is_enforced():
    # sorted 0.01(r1)->0.03, 0.02(r2)->0.03, 0.9(r3)->0.9 ; stays order-preserving
    assert benjamini_hochberg([0.9, 0.01, 0.02]) == pytest.approx([0.9, 0.03, 0.03])


def test_bh_caps_at_one_and_handles_single_and_empty():
    assert benjamini_hochberg([0.04]) == pytest.approx([0.04])
    assert benjamini_hochberg([0.9, 0.8]) == pytest.approx([0.9, 0.9])
    assert benjamini_hochberg([]) == []


def test_bh_ties_get_equal_adjusted_values():
    assert benjamini_hochberg([0.02, 0.02, 0.5]) == pytest.approx([0.03, 0.03, 0.5])


@pytest.mark.parametrize("bad", [[-0.1], [1.2], [None]])
def test_bh_rejects_invalid_p_values(bad):
    with pytest.raises(ValueError):
        benjamini_hochberg(bad)

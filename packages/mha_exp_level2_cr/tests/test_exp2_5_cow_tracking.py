"""Regressions for observable target continuity and consumption attribution."""

import pytest
from mha_exp_level2_cr.exp2_5.cow_tracking import associate_cow, consumed_tracked_cow


@pytest.mark.parametrize('target,previous,current,expected', [
    ((0, 0), [(0, 0)], [(0, 1)], (0, 1)),
    ((0, 0), [(0, 0), (4, 0)], [(4, 0)], None),
    ((1, 0), [(0, 0), (1, 0)], [(1, 0), (2, 0)], (2, 0)),
    ((0, 0), [(0, 0), (1, 0)], [(0, 0), (1, 0)], (0, 0)),
    ((0, 0), [(0, 0), (1, 1)], [(0, 1), (1, 0)], None),
    ((0, 0), [(0, 0)], [(2, 0)], None),
    (None, [(0, 0)], [(0, 0)], None),
    ((1, 0), [(1, 0)], [(1, 0), (2, 0)], None),
    ((1, 0), [(1, 0), (2, 0)], [(1, 0)], None),
    ((1, 0), [(1, 0)], [(1, 0), (4, 0)], (1, 0)),
])
def test_visible_cow_association(target, previous, current, expected):
    """Association handles continuity, vacated cells, ambiguity and disappearance."""
    assert associate_cow(target, previous, current) == expected
    assert associate_cow(target, reversed(previous), reversed(current)) == expected


def test_dense_cow_association_is_bounded():
    """Crowded ambiguous components cannot stall a reactive callback."""
    cows = [(x, y) for x in range(4) for y in range(4)]
    assert associate_cow((0, 0), cows, cows) is None


@pytest.mark.parametrize('action,facing,target,events,illegal,expected', [
    (5, (1, 0), (1, 0), ['eat_cow'], False, True),
    (5, (1, 0), (0, 1), ['eat_cow'], False, False),
    (5, (1, 0), None, ['eat_cow'], False, False),
    (5, (1, 0), (1, 0), [], False, False),
    (5, (1, 0), (1, 0), ['eat_cow'], True, False),
    (2, (1, 0), (1, 0), ['eat_cow'], False, False),
])
def test_consumption_requires_tracked_cell(action, facing, target, events, illegal, expected):
    """A different cow's achievement cannot complete the designated-cow goal."""
    assert consumed_tracked_cow(action, (0, 0), facing, target, events, illegal) is expected

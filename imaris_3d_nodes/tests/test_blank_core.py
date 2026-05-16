"""Tests for blank_core.normalize_by_reference."""
import numpy as np
import pandas as pd
import pytest

from imaris_3d_nodes import blank_core


def test_normalize_by_reference_divides_by_ref_mean():
    wide = pd.DataFrame({'neg': [1, 2, 3], 'pos': [10, 20, 30]})
    out = blank_core.normalize_by_reference(wide, ref_group='neg')
    # ref_mean = 2.0
    assert out['neg'].tolist() == pytest.approx([0.5, 1.0, 1.5])
    assert out['pos'].tolist() == pytest.approx([5.0, 10.0, 15.0])


def test_normalize_by_reference_missing_ref_is_noop():
    wide = pd.DataFrame({'neg': [1, 2, 3]})
    out = blank_core.normalize_by_reference(wide, ref_group='not_there')
    pd.testing.assert_frame_equal(out, wide)


def test_normalize_by_reference_zero_mean_is_noop():
    wide = pd.DataFrame({'neg': [0, 0, 0], 'pos': [1, 2, 3]})
    out = blank_core.normalize_by_reference(wide, ref_group='neg')
    pd.testing.assert_frame_equal(out, wide)

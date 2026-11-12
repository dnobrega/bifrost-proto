"""Regression tests for registered quantities with unknown mesh locations."""

from __future__ import annotations

import pytest
from test_engine import make_data


@pytest.mark.parametrize("center_keyword", ["centered", "center"])
def test_centering_registered_unknown_location_is_value_error(center_keyword):
    dd = make_data(n=12)
    dd.register_variable("unlocated_density", storage_name="r", unit_type="density")

    with pytest.raises(
        ValueError, match="Cannot center.*mesh location is not registered"
    ):
        dd.get_var(
            "unlocated_density",
            0,
            units="bifrost",
            **{center_keyword: True},
        )

    # An uncentered registered quantity is still readable normally.
    assert dd.get_var("unlocated_density", 0, units="bifrost").shape == (12, 12, 12)

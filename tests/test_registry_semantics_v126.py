"""Regression tests for the v0.1.26 quantity-metadata cleanup."""

import bifrost_proto.quantities  # noqa: F401 -- populate the built-in registry
from bifrost_proto.registry import GLOBAL_REGISTRY


def _aliases_for(name):
    return {
        alias
        for alias, canonical in GLOBAL_REGISTRY.aliases.items()
        if canonical == name
    }


def test_searchable_description_keywords_are_consistent():
    for name, spec in GLOBAL_REGISTRY.items():
        description = spec.description.lower()
        if name.startswith("mod") or name in {"mode"}:
            assert "magnitude" in description, (name, spec.description)
        if spec.unit_type == "acceleration":
            assert "acceleration" in description, (name, spec.description)
        if spec.unit_type == "inverse_length":
            assert "inverse length" in description, (name, spec.description)
        if name.startswith("tauq"):
            assert "timescale" in description, (name, spec.description)


def test_descriptions_do_not_duplicate_location_or_storage_metadata():
    forbidden = (
        "cell center",
        "x-face",
        "y-face",
        "z-face",
        "x-edge",
        "y-edge",
        "z-edge",
        "stored by",
        "stored in",
        "historical suffix",
        "legacy stagger",
        "after centering",
        "located on",
    )
    for name, spec in GLOBAL_REGISTRY.items():
        description = spec.description.lower()
        assert not any(term in description for term in forbidden), (
            name,
            spec.description,
        )


def test_clean_velocity_alias_policy():
    assert _aliases_for("u_h") == {"uh"}
    assert _aliases_for("b_h") == {"bh"}
    assert _aliases_for("modcurlu") == set()
    assert _aliases_for("uperpb") == set()
    for name in ("uparb_x", "uparb_y", "uparb_z", "uperpb_x", "uperpb_y", "uperpb_z"):
        assert _aliases_for(name) == set()


def test_pressure_gradient_metadata_describes_force_sign():
    assert GLOBAL_REGISTRY["gradpg_x"].description.endswith("-dpg/dx.")
    assert GLOBAL_REGISTRY["gradpg_y"].description.endswith("-dpg/dy.")
    assert GLOBAL_REGISTRY["gradpg_z"].description.endswith("-dpg/dz.")
    for name in ("gradpgor_x", "gradpgor_y", "gradpgor_z", "gradpgor_h", "modgradpgor"):
        assert "acceleration" in GLOBAL_REGISTRY[name].description.lower()

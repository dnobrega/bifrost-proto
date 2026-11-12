"""Regression tests for plotting labels and runtime documentation."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import bifrost_proto
from bifrost_proto.core import BifrostData
from bifrost_proto.labels import latex_varname
from bifrost_proto.registry import GLOBAL_REGISTRY


def test_get_varname_alias_and_latex():
    dd = BifrostData("dummy", fdir=".")
    assert dd.get_var_name("bx") == "b_x"
    assert dd.get_var_name("bx", latex=True) == r"$B_x$"
    assert dd.get_var_name("b_x", latex=True) == r"$B_x$"
    assert dd.get_var_name("tauqjoule", latex=True) == r"$\tau_{Q_J}$"
    assert dd.get_var_name("theta_jb", latex=True) == r"$\theta_{JB}$"
    assert dd.get_var_name("gradpg_z", latex=True) == r"$-(\nabla p_g)_z$"
    assert dd.get_var_name("gradpgor_z", latex=True) == r"$-(\nabla p_g)_{z}/\rho$"


def test_every_official_variable_has_plot_label():
    for name in GLOBAL_REGISTRY:
        label = latex_varname(name)
        assert isinstance(label, str)
        assert label.startswith("$") and label.endswith("$")
        assert label != rf"$\mathrm{{{name}}}$"


def test_public_bifrostdata_methods_have_docstrings():
    for name, member in inspect.getmembers(BifrostData, predicate=inspect.isfunction):
        if not name.startswith("_"):
            assert inspect.getdoc(member), name


def test_registered_getters_have_runtime_docstrings():
    for name, spec in GLOBAL_REGISTRY.items():
        if spec.getter is not None:
            assert inspect.getdoc(spec.getter), name


def test_python_functions_and_classes_are_documented():
    """Keep the documentation audit enforceable as the package grows."""
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "bifrost_proto").glob("*.py"))
    sources += list((root / "tools" / "build_instrument_responses").glob("*.py"))
    sources += list((root / "benchmarks").glob("*.py"))

    missing = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node) is None:
                    missing.append(
                        f"{path.relative_to(root)}:{node.lineno}:{node.name}"
                    )
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)

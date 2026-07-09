"""Smoke tests for the ipywidgets-based interactive paths (Jupyter usage).

These run headless: widgets are instantiated and callbacks wired exactly as in
a notebook, which validates compatibility with the installed ipywidgets
version (>= 8), without requiring a browser front-end.
"""

import ipywidgets
import pytest

from pyvplm.addon.variablepowerlaw import pi_dependency, pi_sensitivity
from pyvplm.core.definition import PositiveParameter, PositiveParameterSet


@pytest.fixture
def pi_set():
    pi1 = PositiveParameter("pi1", [0.1, 1], "", "pi1")
    pi2 = PositiveParameter("pi2", [0.1, 1], "", "pi2")
    pi3 = PositiveParameter("pi3", [0.1, 1], "", "pi3")
    return PositiveParameterSet(pi1, pi2, pi3)


def test_ipywidgets_version_is_modern():
    major = int(ipywidgets.__version__.split(".")[0])
    assert major >= 8


def test_pi_sensitivity_widget_path(pi_set, power_law_doe):
    # must build the Tab/Checkbox/Slider menu and register the callback
    # without raising under ipywidgets >= 8
    pi_sensitivity(pi_set, power_law_doe, True, test_mode=True)


def test_pi_dependency_widget_path(pi_set, power_law_doe):
    pi_dependency(pi_set, power_law_doe, True, test_mode=True)


def test_pi_dependency_widget_path_with_lists(pi_set, power_law_doe):
    pi_dependency(
        pi_set, power_law_doe, True, test_mode=True, x_list=["pi2", "pi3"], y_list=["pi1"]
    )


def test_widget_style_dict_assignment():
    # the library assigns dicts to .style; ipywidgets must coerce them
    checkbox = ipywidgets.Checkbox(value=True)
    checkbox.style = {"description_width": "0px"}
    assert checkbox.style.description_width == "0px"


def test_checkbox_tab_menu_pattern():
    # replicate the menu construction pattern used by the library
    panel = ipywidgets.HBox()
    panel.children = [ipywidgets.Checkbox(value=True), ipywidgets.Label(value="$\\pi_{1}$")]
    tab = ipywidgets.Tab(children=[panel])
    tab.set_title(0, "Primary PI")
    button = ipywidgets.Button(description="Apply")
    clicked = []
    button.on_click(lambda b: clicked.append(True))
    button.click()
    assert clicked == [True]
    menu = ipywidgets.VBox(children=[tab, button])
    assert len(menu.children) == 2


def test_pi_sensitivity_widget_apply_callback(pi_set, power_law_doe, monkeypatch):
    """Drive the 'Apply' button callback registered by pi_sensitivity."""
    captured = {}
    original_on_click = ipywidgets.Button.on_click

    def capture_on_click(self, callback, **kwargs):
        captured["callback"] = callback
        return original_on_click(self, callback, **kwargs)

    monkeypatch.setattr(ipywidgets.Button, "on_click", capture_on_click)
    pi_sensitivity(pi_set, power_law_doe, True, test_mode=True)
    assert "callback" in captured
    # simulate a click: the callback must re-plot without raising
    captured["callback"](None)


def test_pi_dependency_widget_apply_callback(pi_set, power_law_doe, monkeypatch):
    """Drive the 'Apply' button callback registered by pi_dependency."""
    captured = {}
    original_on_click = ipywidgets.Button.on_click

    def capture_on_click(self, callback, **kwargs):
        captured["callback"] = callback
        return original_on_click(self, callback, **kwargs)

    monkeypatch.setattr(ipywidgets.Button, "on_click", capture_on_click)
    pi_dependency(pi_set, power_law_doe, True, test_mode=True)
    assert "callback" in captured
    captured["callback"](None)

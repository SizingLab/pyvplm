"""Tests for the ipywidgets-based interactive paths (Jupyter usage).

These run headless: the live-control widget apps are built and their observe
callbacks driven exactly as a notebook front-end would, which validates both
ipywidgets >= 8 compatibility and the redraw logic, without a browser.
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


def _collect(widget, widget_type):
    """Recursively collect widgets of a given type from a container."""
    found = []
    for child in getattr(widget, "children", ()):
        if isinstance(child, widget_type):
            found.append(child)
        found.extend(_collect(child, widget_type))
    return found


def test_ipywidgets_version_is_modern():
    major = int(ipywidgets.__version__.split(".")[0])
    assert major >= 8


class TestPiSensitivityApp:
    def test_returns_app_with_live_controls(self, pi_set, power_law_doe):
        app = pi_sensitivity(pi_set, power_law_doe, True, test_mode=True)
        assert isinstance(app, ipywidgets.VBox)
        toggles = _collect(app, ipywidgets.ToggleButton)
        # one chip per pi for primary and secondary rows
        assert len(toggles) == 2 * power_law_doe.shape[1]
        # unicode pi labels
        assert toggles[0].description == "π₁"
        assert len(_collect(app, ipywidgets.Output)) == 1
        # no Apply button anymore: updates are live
        assert len(_collect(app, ipywidgets.Button)) == 0

    def test_toggling_a_chip_redraws(self, pi_set, power_law_doe):
        app = pi_sensitivity(pi_set, power_law_doe, True, test_mode=True)
        toggles = _collect(app, ipywidgets.ToggleButton)
        # select pi2 as an additional primary pi -> observe fires the redraw
        toggles[1].value = True
        # deselecting everything must not raise (update is skipped)
        for toggle in toggles:
            toggle.value = False

    def test_slider_change_redraws(self, pi_set, power_law_doe):
        app = pi_sensitivity(pi_set, power_law_doe, True, test_mode=True)
        slider = _collect(app, ipywidgets.IntSlider)[0]
        slider.value = 12


class TestPiDependencyApp:
    def test_returns_app_with_live_controls(self, pi_set, power_law_doe):
        app = pi_dependency(pi_set, power_law_doe, True, test_mode=True)
        assert isinstance(app, ipywidgets.VBox)
        toggles = _collect(app, ipywidgets.ToggleButton)
        assert len(toggles) == 2 * power_law_doe.shape[1]
        assert len(_collect(app, ipywidgets.Button)) == 0

    def test_toggling_and_sliders_redraw(self, pi_set, power_law_doe):
        app = pi_dependency(pi_set, power_law_doe, True, test_mode=True)
        toggles = _collect(app, ipywidgets.ToggleButton)
        toggles[0].value = False  # reduce x-axis selection
        order = [s for s in _collect(app, ipywidgets.IntSlider) if s.description == "Order:"][0]
        order.value = 1
        threshold = _collect(app, ipywidgets.FloatSlider)[0]
        threshold.value = 0.5

    def test_preselected_lists(self, pi_set, power_law_doe):
        app = pi_dependency(
            pi_set, power_law_doe, True, test_mode=True, x_list=["pi2", "pi3"], y_list=["pi1"]
        )
        toggles = _collect(app, ipywidgets.ToggleButton)
        x_row, y_row = toggles[:3], toggles[3:]
        assert [t.value for t in x_row] == [False, True, True]
        assert [t.value for t in y_row] == [True, False, False]

    def test_unknown_pi_in_list_rejected(self, pi_set, power_law_doe):
        with pytest.raises(KeyError):
            pi_dependency(pi_set, power_law_doe, True, test_mode=True, x_list=["pi9"])


class TestWidgetPrimitives:
    def test_widget_style_dict_assignment(self):
        # ipywidgets must coerce dict style assignments (used historically)
        checkbox = ipywidgets.Checkbox(value=True)
        checkbox.style = {"description_width": "0px"}
        assert checkbox.style.description_width == "0px"

    def test_toggle_observe_fires(self):
        toggle = ipywidgets.ToggleButton(value=False, description="π₁")
        fired = []
        toggle.observe(lambda change: fired.append(change["new"]), names="value")
        toggle.value = True
        assert fired == [True]

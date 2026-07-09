"""Shared fixtures for the pyVPLM test suite."""

import matplotlib

matplotlib.use("Agg")  # must happen before pyplot is imported by the package

import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402
import pytest  # noqa: E402

from pyvplm.core.definition import PositiveParameter, PositiveParameterSet  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory warnings."""
    yield
    plt.close("all")


@pytest.fixture
def cantilever_parameter_set() -> PositiveParameterSet:
    """The cantilever beam example used throughout the documentation."""
    u = PositiveParameter("u", [1e-9, 1e-6], "m", "Deflection")
    f = PositiveParameter("f", [150, 500], "N", "Load applied")
    l = PositiveParameter("l", [1, 3], "m", "Cantilever length")
    e = PositiveParameter("e", [60e9, 80e9], "Pa", "Young Modulus")
    d = PositiveParameter("d", [10, 60], "mm", "Diameter of cross-section")
    parameter_set = PositiveParameterSet(u, f, l, e, d)
    parameter_set.first("d", "l")
    return parameter_set


@pytest.fixture
def power_law_doe() -> numpy.ndarray:
    """Synthetic DOE following a known variable power-law: pi1 = f(pi2, pi3).

    log(pi1) = 2.33 + 1.35*log(pi2) + 0.17*log(pi2)**2 + 0.042*log(pi2)*log(pi3)
               - 0.25*log(pi3)
    """
    rng = numpy.random.default_rng(seed=42)
    pi2 = 10 ** rng.uniform(-1.0, 1.0, 40)
    pi3 = 10 ** rng.uniform(0.0, 2.0, 40)
    log_pi2 = numpy.log10(pi2)
    log_pi3 = numpy.log10(pi3)
    log_pi1 = 2.33 + 1.35 * log_pi2 + 0.17 * log_pi2**2 + 0.042 * log_pi2 * log_pi3 - 0.25 * log_pi3
    pi1 = 10**log_pi1
    return numpy.column_stack([pi1, pi2, pi3])

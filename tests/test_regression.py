"""Non-regression tests: the regression engine must recover a known power-law.

The synthetic DOE follows
    log(pi1) = 2.33 + 1.35*log(pi2) + 0.17*log(pi2)**2
               + 0.042*log(pi2)*log(pi3) - 0.25*log(pi3)
so a complete order-2 model in log space must recover these coefficients and
reach (numerically) zero error. These values act as the reference behaviour of
the regression engine: any algorithmic change breaking them is a regression.
"""

import re

import numpy
import pytest

from pyvplm.addon.variablepowerlaw import perform_regression, regression_models

TRUE_MODEL = {
    "": 2.33,  # constant term
    "log(pi2)": 1.35,
    "log(pi2)**2": 0.17,
    "log(pi2)*log(pi3)": 0.042,
    "log(pi3)": -0.25,
    "log(pi3)**2": 0.0,
}


def _fit_models(doe, **kwargs):
    return regression_models(doe, "pi1", 2, test_mode=True, plots=False, **kwargs)


class TestRegressionModels:
    def test_model_count_matches_term_number(self, power_law_doe):
        models = _fit_models(power_law_doe)
        # order-2 model with 2 explanatory pi: constant + 5 terms
        numeric_keys = [key for key in models.keys() if isinstance(key, int)]
        assert sorted(numeric_keys) == [1, 2, 3, 4, 5, 6]
        for indicator in ["max |e|", "ave. |e|", "ave. e", "sigma e"]:
            assert indicator in models

    def test_full_model_recovers_true_coefficients(self, power_law_doe):
        models = _fit_models(power_law_doe)
        expression, coeff, _, _ = models[6]
        assert expression.startswith("log(pi1) = ")
        # parse fitted terms: {'': a0, 'log(pi2)': a1, ...}
        body = expression.split("=", 1)[1].replace(" ", "")
        fitted = {}
        for match in re.finditer(
            r"([+-]?\d+\.\d+)(?:\*)?((?:log\(pi\d\)(?:\*\*2|\*log\(pi\d\))?)?)", body
        ):
            fitted[match.group(2)] = float(match.group(1))
        for term, expected in TRUE_MODEL.items():
            assert fitted.get(term, 0.0) == pytest.approx(expected, abs=1e-3), term

    def test_full_model_error_is_negligible(self, power_law_doe):
        models = _fit_models(power_law_doe)
        max_abs_error_train = models["max |e|"][0]
        # last model (all terms) reproduces the data
        assert max_abs_error_train[-1] < 1e-6
        # error decreases when adding relevant terms
        assert max_abs_error_train[-1] <= max_abs_error_train[0]

    def test_error_indicators_are_reproducible(self, power_law_doe):
        models_first = _fit_models(power_law_doe)
        models_second = _fit_models(power_law_doe)
        assert models_first["ave. |e|"][0] == pytest.approx(models_second["ave. |e|"][0])
        assert models_first[6][1] == pytest.approx(models_second[6][1])

    def test_linear_space_regression(self, power_law_doe):
        models = regression_models(
            power_law_doe, "pi1", 1, test_mode=True, plots=False, log_space=False
        )
        numeric_keys = [key for key in models.keys() if isinstance(key, int)]
        assert sorted(numeric_keys) == [1, 2, 3]

    def test_nan_values_rejected(self, power_law_doe):
        doe = power_law_doe.copy()
        doe[0, 0] = numpy.nan
        with pytest.raises(ValueError):
            _fit_models(doe)

    def test_negative_values_rejected_in_log_space(self, power_law_doe):
        doe = power_law_doe.copy()
        doe[0, 0] = -1.0
        with pytest.raises(ValueError):
            _fit_models(doe)

    def test_invalid_pi_name_rejected(self, power_law_doe):
        with pytest.raises(SyntaxError):
            regression_models(power_law_doe, "sigma1", 2, test_mode=True, plots=False)


class TestPerformRegression:
    def test_full_model_expression_and_latex(self, power_law_doe):
        models = _fit_models(power_law_doe)
        expression, expression_latex = perform_regression(
            power_law_doe, models, chosen_model=6, test_mode=True
        )
        assert "pi1" in expression
        assert expression_latex.startswith("$")

    def test_no_plots_returns_predictions(self, power_law_doe):
        models = _fit_models(power_law_doe)
        expression, expression_latex, y, y_reg = perform_regression(
            power_law_doe, models, chosen_model=6, test_mode=True, no_plots=True
        )
        # full model reproduces the data almost exactly
        assert y_reg == pytest.approx(y, rel=1e-6)

    def test_invalid_model_number_rejected(self, power_law_doe):
        models = _fit_models(power_law_doe)
        with pytest.raises(KeyError):
            perform_regression(power_law_doe, models, chosen_model=99, test_mode=True)
        with pytest.raises(ValueError):
            perform_regression(power_law_doe, models, chosen_model=0, test_mode=True)

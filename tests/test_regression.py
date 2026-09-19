"""Non-regression tests: the regression engine must recover a known power-law.

The synthetic DOE follows
    log(pi1) = 2.33 + 1.35*log(pi2) + 0.17*log(pi2)**2
               + 0.042*log(pi2)*log(pi3) - 0.25*log(pi3)
so a complete order-2 model in log space must recover these coefficients and
reach (numerically) zero error. These values act as the reference behaviour of
the regression engine: any algorithmic change breaking them is a regression.
"""

import json
import pathlib
import re

import numpy
import pytest

from pyvplm.addon.variablepowerlaw import (
    evaluate_model,
    model_from_json,
    model_terms,
    model_to_json,
    perform_regression,
    regression_models,
)

REFERENCE_FILE = pathlib.Path(__file__).parent / "data" / "regression_models_reference.json"

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


def _fitted_terms(expression):
    """Parse a model expression into {'': a0, 'log(pi2)': a1, ...}."""
    body = expression.split("=", 1)[1].replace(" ", "")
    fitted = {}
    for match in re.finditer(
        r"([+-]?\d+\.\d+)(?:\*)?((?:log\(pi\d\)(?:\*\*2|\*log\(pi\d\))?)?)", body
    ):
        fitted[match.group(2)] = float(match.group(1))
    return fitted


def _three_group_doe():
    """Groups 0 and 1 follow TRUE_MODEL, group 2 another power-law on its own pi2 range."""
    rng = numpy.random.default_rng(seed=0)
    groups = numpy.repeat([0, 1, 2], 20)
    log_pi3 = rng.uniform(0.0, 2.0, 60)
    log_pi2 = numpy.where(groups == 2, rng.uniform(0.5, 1.0, 60), rng.uniform(-1.0, 0.5, 60))
    log_pi1 = 2.33 + 1.35 * log_pi2 + 0.17 * log_pi2**2 + 0.042 * log_pi2 * log_pi3 - 0.25 * log_pi3
    other = groups == 2
    log_pi1[other] = 1.8 + 2.0 * log_pi2[other] - 0.5 * log_pi3[other]
    return numpy.column_stack([10**log_pi1, 10**log_pi2, 10**log_pi3]), groups


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
        fitted = _fitted_terms(expression)
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


class TestGroupsAndWeights:
    def test_default_outputs_match_reference(self, power_law_doe):
        # reference written by the version preceding groups and weights; bit-identical on the
        # machine that wrote it, tolerance only absorbs LAPACK builds of the CI matrix
        reference = json.loads(REFERENCE_FILE.read_text())
        models = _fit_models(power_law_doe)
        assert [key for key in models if isinstance(key, int)] == [1, 2, 3, 4, 5, 6]
        assert [key for key in models if not isinstance(key, int)] == [
            "max |e|",
            "ave. |e|",
            "ave. e",
            "sigma e",
        ]
        for key, expected in reference["models"].items():
            expression, coeff, error_train, error_test = models[int(key)]
            assert expression == expected["expression"]
            numpy.testing.assert_allclose(coeff, expected["coeff"], rtol=1e-12, atol=1e-12)
            for frame, values in [(error_train, expected["train"]), (error_test, expected["test"])]:
                assert list(frame.index) == ["max |e|", "ave. |e|", "ave. e", "sigma e"]
                numpy.testing.assert_allclose(frame[0], values, rtol=1e-12, atol=1e-12)
        for indicator in ["max |e|", "ave. |e|", "ave. e", "sigma e"]:
            numpy.testing.assert_allclose(
                models[indicator], reference[indicator], rtol=1e-12, atol=1e-12
            )

    def test_unit_weights_are_bit_identical(self, power_law_doe):
        models = _fit_models(power_law_doe)
        weighted = _fit_models(power_law_doe, weights=numpy.ones(len(power_law_doe)))
        assert weighted.keys() == models.keys()
        for key in [key for key in models if isinstance(key, int)]:
            assert weighted[key][0] == models[key][0]
            numpy.testing.assert_array_equal(weighted[key][1], models[key][1])
            numpy.testing.assert_array_equal(weighted[key][2], models[key][2])
            numpy.testing.assert_array_equal(weighted[key][3], models[key][3])

    def test_singleton_groups_reproduce_leave_one_point_out(self, power_law_doe):
        models = _fit_models(power_law_doe)
        grouped = _fit_models(power_law_doe, groups=numpy.arange(len(power_law_doe)))
        for indicator in ["max |e|", "ave. |e|", "ave. e", "sigma e"]:
            # full-model errors sit at round-off level (~1e-13 %), hence the absolute tolerance
            numpy.testing.assert_allclose(
                grouped[indicator][1], models[indicator][1], rtol=1e-9, atol=1e-10
            )

    def test_group_errors_frame(self):
        doe, groups = _three_group_doe()
        models = _fit_models(doe, groups=groups)
        frame = models["group errors"]
        assert list(frame.index) == [0, 1, 2]
        assert list(frame.columns) == ["max |e|", "ave. |e|", "ave. e", "sigma e", "n"]
        assert list(frame["n"]) == [20, 20, 20]
        assert len(models["group errors by model"]) == 6
        assert models["group errors by model"][-1] is frame
        # tested set is the union of the out-of-group predictions
        assert models["max |e|"][1][-1] == pytest.approx(frame["max |e|"].max())

    def test_out_of_group_error_exceeds_leave_one_point_out(self):
        doe, groups = _three_group_doe()
        out_of_group = _fit_models(doe, groups=groups)["group errors"]["ave. |e|"]
        per_point = _fit_models(doe, groups=numpy.arange(len(doe)))["group errors"]["ave. |e|"]
        leave_one_point_out = [per_point.values[groups == g].mean() for g in [0, 1, 2]]
        # the odd group cannot be predicted from the others, while its own points predict it
        assert out_of_group[2] > 3 * leave_one_point_out[2]
        assert out_of_group[2] > 3 * max(out_of_group[0], out_of_group[1])

    def test_weights_recover_the_sound_half(self, power_law_doe):
        doe = power_law_doe.copy()
        half = len(doe) // 2
        rng = numpy.random.default_rng(seed=7)
        doe[half:, 0] *= 10 ** rng.normal(0.0, 0.3, len(doe) - half)
        weights = numpy.r_[numpy.ones(half), numpy.full(len(doe) - half, 1e-6)]
        fitted = _fitted_terms(_fit_models(doe, weights=weights)[6][0])
        unweighted = _fitted_terms(_fit_models(doe)[6][0])
        for term, expected in TRUE_MODEL.items():
            assert fitted.get(term, 0.0) == pytest.approx(expected, rel=0.01, abs=1e-4), term
        # the corruption does move the unweighted fit, so the check above is not vacuous
        assert any(
            unweighted.get(term, 0.0) != pytest.approx(expected, rel=0.01, abs=1e-4)
            for term, expected in TRUE_MODEL.items()
        )

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"groups": [0, 1] * 20}, TypeError),
            ({"groups": numpy.zeros(40)}, TypeError),
            ({"groups": numpy.arange(39)}, ValueError),
            ({"groups": numpy.zeros(40, dtype=int)}, ValueError),
            ({"weights": [1.0] * 40}, TypeError),
            ({"weights": numpy.full(40, "1")}, TypeError),
            ({"weights": numpy.ones(41)}, ValueError),
            ({"weights": numpy.ones((40, 1))}, ValueError),
            ({"weights": numpy.r_[0.0, numpy.ones(39)]}, ValueError),
            ({"weights": numpy.r_[-1.0, numpy.ones(39)]}, ValueError),
            ({"weights": numpy.r_[numpy.nan, numpy.ones(39)]}, ValueError),
        ],
    )
    def test_invalid_groups_and_weights_rejected(self, power_law_doe, kwargs, error):
        with pytest.raises(error):
            _fit_models(power_law_doe, **kwargs)


class TestModelExport:
    def test_terms_carry_labels_coefficients_and_powers(self, power_law_doe):
        models = _fit_models(power_law_doe)
        terms, meta = model_terms(models, 6)
        assert meta["output"] == "pi1" and meta["inputs"] == ["pi2", "pi3"]
        assert meta["log_space"] and meta["log_base"] == 10
        assert list(terms.columns) == ["label", "coeff", "pi2", "pi3"]
        assert terms["label"][0] == "(constant)"
        numpy.testing.assert_array_equal(terms["coeff"], models[6][1])
        expected_powers = {
            "(constant)": (0, 0),
            "log(pi2)": (1, 0),
            "log(pi2)**2": (2, 0),
            "log(pi2)*log(pi3)": (1, 1),
            "log(pi3)": (0, 1),
            "log(pi3)**2": (0, 2),
        }
        for _, row in terms.iterrows():
            assert (row["pi2"], row["pi3"]) == expected_powers[row["label"]]
            key = "" if row["label"] == "(constant)" else row["label"]
            assert row["coeff"] == pytest.approx(TRUE_MODEL[key], abs=1e-3)

    def test_names_replace_pi(self, power_law_doe):
        terms, meta = model_terms(_fit_models(power_law_doe), 6, names=["y", "a", "b"])
        assert meta["output"] == "y" and meta["inputs"] == ["a", "b"]
        assert set(terms["label"]) == {
            "(constant)",
            "log(a)",
            "log(a)**2",
            "log(a)*log(b)",
            "log(b)",
            "log(b)**2",
        }

    @pytest.mark.parametrize("order, log_space", [(2, True), (3, True), (1, False)])
    def test_evaluation_reproduces_trained_set(self, power_law_doe, order, log_space):
        models = regression_models(
            power_law_doe,
            "pi1",
            order,
            test_mode=True,
            plots=False,
            log_space=log_space,
            groups=numpy.arange(len(power_law_doe)) % 4,
        )
        chosen = max(key for key in models if isinstance(key, int))
        terms, meta = model_terms(models, chosen)
        y = evaluate_model(terms, meta, power_law_doe[:, 1:])
        error = (y - power_law_doe[:, 0]) / power_law_doe[:, 0] * 100
        # with groups, the saved coefficients are the full fit that gives the trained set
        assert numpy.amax(numpy.absolute(error)) == pytest.approx(
            models["max |e|"][0][-1], rel=1e-9, abs=1e-9
        )

    def test_json_round_trip_is_exact(self, power_law_doe, tmp_path):
        terms, meta = model_terms(_fit_models(power_law_doe), 4, names=["y", "a", "b"])
        path = tmp_path / "model.json"
        text = model_to_json(terms, meta, str(path))
        for source in [str(path), text]:
            read_terms, read_meta = model_from_json(source)
            assert read_meta == meta
            assert read_terms.equals(terms)

    def test_invalid_arguments_rejected(self, power_law_doe):
        models = _fit_models(power_law_doe)
        with pytest.raises(KeyError):
            model_terms(models, 7)
        with pytest.raises(ValueError):
            model_terms(models, 6, names=["y", "a"])
        with pytest.raises(TypeError):
            model_terms(models, 6, names="yab")
        terms, meta = model_terms(models, 6)
        with pytest.raises(ValueError):
            evaluate_model(terms, meta, power_law_doe)
        with pytest.raises(ValueError):
            evaluate_model(terms, meta, -power_law_doe[:, 1:])


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

    def test_linear_space_with_read_only_doe(self, power_law_doe):
        # pandas 3 DataFrame.values returns read-only arrays; in linear space
        # perform_regression works on views of the DOE and must not mutate it
        models = regression_models(
            power_law_doe, "pi1", 1, test_mode=True, plots=False, log_space=False
        )
        doe = power_law_doe.copy()
        doe.setflags(write=False)
        expression, expression_latex = perform_regression(
            doe, models, chosen_model=3, test_mode=True
        )
        assert "pi1" in expression

    def test_invalid_model_number_rejected(self, power_law_doe):
        models = _fit_models(power_law_doe)
        with pytest.raises(KeyError):
            perform_regression(power_law_doe, models, chosen_model=99, test_mode=True)
        with pytest.raises(ValueError):
            perform_regression(power_law_doe, models, chosen_model=0, test_mode=True)

"""Unit tests for pyvplm.addon.variablepowerlaw (dimensional analysis & regression)."""

import numpy
import pandas
import pytest

from pyvplm.addon.variablepowerlaw import (
    adapt_parameter_set,
    automatic_buckingham,
    buckingham_theorem,
    compute_echelon_form,
    concatenate_expression,
    declare_constraints,
    declare_func_x_to_pi,
    force_buckingham,
    import_csv,
    latex_pi_expression,
    pi_dependency,
    pi_sensitivity,
    reduce_parameter_set,
    save_csv,
    write_dimensional_matrix,
)
from pyvplm.core.definition import (
    Constraint,
    ConstraintSet,
    PositiveParameter,
    PositiveParameterSet,
)


class TestWriteDimensionalMatrix:
    def test_cantilever_matrix(self, cantilever_parameter_set):
        matrix = write_dimensional_matrix(cantilever_parameter_set)
        assert list(matrix.index) == ["d", "l", "u", "f", "e"]
        assert list(matrix.columns) == ["[length]", "[mass]", "[time]"]
        assert matrix.values.tolist() == [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, -2],
            [-1, 1, -2],
        ]

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            write_dimensional_matrix("not a set")


class TestComputeEchelonForm:
    def test_known_matrix(self):
        in_matrix = numpy.array([[0, 1, 0], [1, 1, -2], [0, 1, 0], [1, -1, -2], [0, 1, 0]], int)
        out_matrix, pivot, points = compute_echelon_form(in_matrix)
        out_as_float = [[float(item) for item in row] for row in out_matrix]
        assert out_as_float == [
            [1.0, 0.0, -2.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        assert points == [1, 0]
        # pivot * in_matrix must reproduce the echelon form
        product = numpy.array(pivot, dtype=float) @ in_matrix.astype(float)
        numpy.testing.assert_allclose(product, out_as_float)

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            compute_echelon_form([[1, 0], [0, 1]])


class TestBuckinghamTheorem:
    def test_cantilever_pi_set(self, cantilever_parameter_set):
        pi_set, pi_list = buckingham_theorem(cantilever_parameter_set, False)
        assert pi_list == ["d**-1.0*u**1.0", "d**-1.0*l**1.0", "d**2.0*e**1.0*f**-1.0"]
        assert list(pi_set.dictionary.keys()) == ["pi1", "pi2", "pi3"]
        # pi bounds derived from parameter SI bounds
        pi1 = pi_set["pi1"]
        assert pi1._SI_bounds == pytest.approx([1e-9 / 0.06, 1e-6 / 0.01])

    def test_dimensionless_parameters_combined(self):
        # dimensionless parameters share the pseudo-dimension "dimensionless",
        # hence they are combined into a single pi (historical behaviour)
        a = PositiveParameter("a", [1, 10], "", "first")
        b = PositiveParameter("b", [1, 10], "", "second")
        parameter_set = PositiveParameterSet(a, b)
        pi_set, pi_list = buckingham_theorem(parameter_set, False)
        assert pi_list == ["a**-1.0*b**1.0"]

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            buckingham_theorem("not a set", False)


class TestAutomaticBuckingham:
    def test_alternative_sets_found(self, cantilever_parameter_set):
        combinatory_pi_set, alternative_set_dict = automatic_buckingham(
            cantilever_parameter_set, False
        )
        assert len(combinatory_pi_set) >= 1
        # each stored entry is a (pi_set, pi_list) tuple with 3 dimensionless parameters
        for key in combinatory_pi_set.keys():
            assert len(combinatory_pi_set[key][1]) == 3
        assert len(alternative_set_dict) == len(combinatory_pi_set)


class TestForceBuckingham:
    def test_valid_forced_set(self, cantilever_parameter_set):
        pi_set = force_buckingham(cantilever_parameter_set, "l/u", "e/f*u^2", "d/u")
        expressions = [pi_set[pi].description for pi in pi_set.dictionary.keys()]
        # factors are ordered following the parameter set order (d, l, u, f, e)
        assert expressions == ["l**1.0*u**-1.0", "u**2.0*f**-1.0*e**1.0", "d**1.0*u**-1.0"]

    def test_non_dimensionless_expression_rejected(self, cantilever_parameter_set):
        with pytest.raises(ValueError):
            force_buckingham(cantilever_parameter_set, "l/u", "e*u^2", "d/u")

    def test_wrong_pi_number_rejected(self, cantilever_parameter_set):
        with pytest.raises(ValueError):
            force_buckingham(cantilever_parameter_set, "l/u", "d/u")

    def test_unknown_parameter_rejected(self, cantilever_parameter_set):
        with pytest.raises(SyntaxError):
            force_buckingham(cantilever_parameter_set, "l/u", "e/f*u^2", "z/u")


class TestDeclareFuncXToPi:
    def test_transformation_values(self):
        # 5 dimensionless parameters -> pi expressions are the parameters themselves
        a = PositiveParameter("a", [0.01, 10], "m", "a")
        b = PositiveParameter("b", [0.01, 10], "m", "b")
        c = PositiveParameter("c", [0.01, 10], "m", "c")
        parameter_set = PositiveParameterSet(a, b, c)
        pi_set, _ = buckingham_theorem(parameter_set, False)
        func_x_to_pi = declare_func_x_to_pi(parameter_set, pi_set)
        x = numpy.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        result = func_x_to_pi(x)
        assert result.shape == (2, 2)
        # pi expressions are ratios to the repetitive parameter a
        assert (result > 0).all()

    def test_cantilever_roundtrip(self, cantilever_parameter_set):
        pi_set, _ = buckingham_theorem(cantilever_parameter_set, False)
        func_x_to_pi = declare_func_x_to_pi(cantilever_parameter_set, pi_set)
        # columns in parameter_set order: d, l, u, f, e (SI units)
        x = numpy.array([[0.02, 2.0, 1e-7, 300.0, 70e9]])
        result = func_x_to_pi(x)
        assert result[0, 0] == pytest.approx(1e-7 / 0.02)  # pi1 = u/d
        assert result[0, 1] == pytest.approx(2.0 / 0.02)  # pi2 = l/d
        assert result[0, 2] == pytest.approx(0.02**2 * 70e9 / 300.0)  # pi3 = d^2*e/f


class TestDeclareConstraints:
    def test_constraint_function_on_doe(self, cantilever_parameter_set):
        constraint_set = ConstraintSet(Constraint("l/d<=150"))
        func = declare_constraints(cantilever_parameter_set, constraint_set)
        # columns order: d, l, u, f, e
        X = numpy.array(
            [
                [0.02, 2.0, 1e-7, 300.0, 70e9],  # l/d = 100 -> valid
                [0.01, 3.0, 1e-7, 300.0, 70e9],  # l/d = 300 -> invalid
            ]
        )
        assert func(X).tolist() == [True, False]

    def test_undeclared_parameter_rejected(self, cantilever_parameter_set):
        constraint_set = ConstraintSet(Constraint("z/d<=150"))
        with pytest.raises(ValueError):
            declare_constraints(cantilever_parameter_set, constraint_set)


class TestReduceParameterSet:
    def test_output_extraction(self, cantilever_parameter_set):
        pi_set, _ = buckingham_theorem(cantilever_parameter_set, False)
        reduced_parameter_set, reduced_pi_set = reduce_parameter_set(
            cantilever_parameter_set, pi_set, "u"
        )
        assert "u" not in reduced_parameter_set.dictionary
        assert "pi1" not in reduced_pi_set.dictionary  # pi1 contains u
        assert len(reduced_pi_set.dictionary) == 2

    def test_missing_output_rejected(self, cantilever_parameter_set):
        pi_set, _ = buckingham_theorem(cantilever_parameter_set, False)
        with pytest.raises(KeyError):
            reduce_parameter_set(cantilever_parameter_set, pi_set, "z")


class TestAdaptParameterSet:
    def test_replace_parameter(self, cantilever_parameter_set):
        parameter_set = cantilever_parameter_set
        pi_set, _ = buckingham_theorem(parameter_set, False)
        rng = numpy.random.default_rng(seed=0)
        columns = list(parameter_set.dictionary.keys())
        data = {}
        for key in columns:
            bounds = parameter_set[key]._SI_bounds
            data[key] = rng.uniform(bounds[0], bounds[1], 10)
        doe_x = pandas.DataFrame(data)[columns]
        new_parameter_set, new_pi_set, new_doe_x = adapt_parameter_set(
            parameter_set, pi_set, doe_x, "l", "l_tot", "l+2*d", "total length"
        )
        assert "l" not in new_parameter_set.dictionary
        assert "l_tot" in new_parameter_set.dictionary
        assert "l_tot" in new_doe_x.columns
        expected = doe_x["l"] + 2 * doe_x["d"]
        assert new_doe_x["l_tot"].values == pytest.approx(expected.values)
        # pi2 expression now involves the new parameter
        assert "l_tot" in new_pi_set["pi2"].description

    def test_inhomogeneous_expression_rejected(self, cantilever_parameter_set):
        parameter_set = cantilever_parameter_set
        pi_set, _ = buckingham_theorem(parameter_set, False)
        columns = list(parameter_set.dictionary.keys())
        data = {key: [parameter_set[key]._SI_bounds[0]] * 3 for key in columns}
        doe_x = pandas.DataFrame(data)[columns]
        with pytest.raises((SyntaxError, ValueError)):
            adapt_parameter_set(parameter_set, pi_set, doe_x, "l", "l_tot", "l+f", "wrong units")


class TestCsvRoundtrip:
    def test_save_and_import(self, cantilever_parameter_set, tmp_path):
        parameter_set = cantilever_parameter_set
        rng = numpy.random.default_rng(seed=1)
        columns = list(parameter_set.dictionary.keys())
        doe_x = numpy.column_stack(
            [
                rng.uniform(parameter_set[key]._SI_bounds[0], parameter_set[key]._SI_bounds[1], 5)
                for key in columns
            ]
        )
        file_name = str(tmp_path / "doe")
        save_csv(doe_x, file_name, parameter_set, is_SI=True)
        imported = import_csv(file_name + ".csv", parameter_set)
        assert list(imported.columns) == columns
        assert imported.values == pytest.approx(doe_x)


class TestLatexHelpers:
    def test_concatenate_power_law_expression(self):
        expression = (
            "log(pi1) = 2.33011+1.35000*log(pi2)-0.25004*log(pi3)"
            "+0.04200*log(pi2)*log(pi3)+0.17000*log(pi2)**2"
        )
        pi_list = ["\\pi_{1}", "\\pi_{2}", "\\pi_{3}"]
        result = concatenate_expression(expression, pi_list)
        assert result.startswith("$\\pi_{1}=10^{2.33011}")
        assert "\\pi_{2}^{" in result and "\\pi_{3}^{" in result

    def test_latex_pi_expression(self, cantilever_parameter_set):
        pi_set, _ = buckingham_theorem(cantilever_parameter_set, False)
        latex_pi_list, problem = latex_pi_expression(pi_set, cantilever_parameter_set)
        assert not problem
        assert len(latex_pi_list) == 3
        assert latex_pi_list[0].startswith("$\\pi_{1}")


class TestSensitivityAndDependency:
    def test_pi_sensitivity_runs(self, power_law_doe):
        pi1 = PositiveParameter("pi1", [0.1, 1], "", "pi1")
        pi2 = PositiveParameter("pi2", [0.1, 1], "", "pi2")
        pi3 = PositiveParameter("pi3", [0.1, 1], "", "pi3")
        pi_set = PositiveParameterSet(pi1, pi2, pi3)
        pi_sensitivity(pi_set, power_law_doe, False, test_mode=True)

    def test_pi_dependency_runs(self, power_law_doe):
        pi1 = PositiveParameter("pi1", [0.1, 1], "", "pi1")
        pi2 = PositiveParameter("pi2", [0.1, 1], "", "pi2")
        pi3 = PositiveParameter("pi3", [0.1, 1], "", "pi3")
        pi_set = PositiveParameterSet(pi1, pi2, pi3)
        pi_dependency(pi_set, power_law_doe, False, test_mode=True)

"""Unit tests for pyvplm.core.definition classes."""

import math

import numpy
import pytest

from pyvplm.core.definition import (
    Constraint,
    ConstraintSet,
    Parameter,
    ParameterSet,
    PositiveParameter,
    PositiveParameterSet,
)


class TestParameter:
    def test_variable_creation(self):
        m = Parameter("M", [50, 100], "kg", "mass")
        assert m.name == "m"  # variables are lower-case by convention
        assert m.defined_bounds == [50.0, 100.0]
        assert m.value == []
        assert m.defined_units == "kilogram"
        assert m._SI_bounds == [50.0, 100.0]
        assert m._dimensionality == "[mass]"

    def test_constant_creation(self):
        k = Parameter("k", [2], "", "oversizing coefficient")
        assert k.value == [2]
        assert k.defined_bounds == []
        assert float(k) == 2.0

    def test_unit_conversion_to_SI(self):
        d = Parameter("d", [10, 60], "mm", "diameter")
        assert d._SI_units == "meter"
        assert d._SI_bounds == pytest.approx([0.01, 0.06])

    def test_float_of_variable_is_nan(self):
        m = Parameter("m", [50, 100], "kg", "mass")
        assert math.isnan(float(m))

    def test_invalid_bounds_order(self):
        with pytest.raises(AssertionError):
            Parameter("m", [100, 50], "kg", "mass")

    def test_invalid_units(self):
        with pytest.raises(ValueError):
            Parameter("m", [50, 100], "not_a_unit", "mass")

    def test_invalid_types(self):
        with pytest.raises(TypeError):
            Parameter(1, [50, 100], "kg", "mass")

    def test_equality(self):
        m1 = Parameter("m", [50, 100], "kg", "mass")
        m2 = Parameter("m", [50, 100], "kg", "mass")
        m3 = Parameter("m", [50, 101], "kg", "mass")
        assert m1 == m2
        assert m1 != m3

    def test_change_bounds_updates_SI_bounds(self):
        d = Parameter("d", [10, 60], "mm", "diameter")
        d.defined_bounds = [20, 80]
        assert d._SI_bounds == pytest.approx([0.02, 0.08])

    def test_write_protected_attribute(self):
        m = Parameter("m", [50, 100], "kg", "mass")
        with pytest.raises(AttributeError):
            m.nonexistent = 1


class TestPositiveParameter:
    def test_negative_bounds_rejected(self):
        with pytest.raises(AssertionError):
            PositiveParameter("m", [-50, 100], "kg", "mass")

    def test_zero_lower_bound_rejected(self):
        with pytest.raises(AssertionError):
            PositiveParameter("m", [0, 100], "kg", "mass")

    def test_valid_creation(self):
        m = PositiveParameter("m", [50, 100], "kg", "mass")
        assert m.defined_bounds == [50.0, 100.0]


class TestParameterSet:
    def test_creation_and_access(self):
        m = PositiveParameter("m", [50, 100], "g", "mass")
        k1 = Parameter("K1", [2], "g", "oversizing coefficient")
        parameter_set = ParameterSet(m, k1)
        assert list(parameter_set.dictionary.keys()) == ["m", "K1"]
        assert float(parameter_set["K1"]) == 2.0

    def test_setitem_and_delitem(self):
        m = Parameter("m", [50, 100], "g", "mass")
        parameter_set = ParameterSet(m)
        k2 = Parameter("K2", [1.5], "", "oversizing coefficient")
        parameter_set["k2"] = k2  # dimensionless constant -> name is lower-case
        assert "k2" in parameter_set.dictionary
        del parameter_set["k2"]
        assert "k2" not in parameter_set.dictionary

    def test_setitem_key_mismatch(self):
        m = Parameter("m", [50, 100], "g", "mass")
        parameter_set = ParameterSet(m)
        with pytest.raises(KeyError):
            parameter_set["wrong_name"] = Parameter("K2", [1.5], "", "coefficient")

    def test_setitem_wrong_type(self):
        m = Parameter("m", [50, 100], "g", "mass")
        parameter_set = ParameterSet(m)
        with pytest.raises(TypeError):
            parameter_set["x"] = "not a parameter"

    def test_delitem_missing_key(self):
        m = Parameter("m", [50, 100], "g", "mass")
        parameter_set = ParameterSet(m)
        with pytest.raises(KeyError):
            del parameter_set["missing"]

    def test_delitem_last_parameter_warns(self):
        m = Parameter("m", [50, 100], "g", "mass")
        parameter_set = ParameterSet(m)
        with pytest.warns(UserWarning):
            del parameter_set["m"]
        assert len(parameter_set.dictionary) == 0

    def test_first_reorders_dictionary(self):
        m = Parameter("m", [50, 100], "g", "mass")
        k1 = Parameter("K1", [2], "g", "coefficient")
        k2 = Parameter("K2", [1.5], "g", "coefficient")
        parameter_set = ParameterSet(m, k1, k2)
        parameter_set.first("K2", "K1")
        assert list(parameter_set.dictionary.keys()) == ["K2", "K1", "m"]

    def test_first_missing_parameter(self):
        m = Parameter("m", [50, 100], "g", "mass")
        parameter_set = ParameterSet(m)
        with pytest.raises(KeyError):
            parameter_set.first("missing")

    def test_non_parameter_rejected(self):
        with pytest.raises(TypeError):
            ParameterSet("not a parameter")


class TestPositiveParameterSet:
    def test_rejects_plain_parameter(self):
        m = Parameter("m", [50, 100], "g", "mass")
        with pytest.raises(TypeError):
            PositiveParameterSet(m)

    def test_setitem_rejects_plain_parameter(self):
        m = PositiveParameter("m", [50, 100], "g", "mass")
        parameter_set = PositiveParameterSet(m)
        with pytest.raises(TypeError):
            parameter_set["k"] = Parameter("k", [2], "", "coefficient")


class TestConstraint:
    def test_lower_equal_constraint(self):
        constraint = Constraint("x<=10", "x should be small")
        assert set(constraint.parameters) == {"x"}
        assert constraint.compute({"x": 3}) == pytest.approx(7.0)

    def test_strict_inequality_converted(self):
        constraint = Constraint("x/y<100")
        assert set(constraint.parameters) == {"x", "y"}
        assert constraint.compute({"x": 50, "y": 1}) == pytest.approx(50.0)

    def test_greater_equal_constraint(self):
        constraint = Constraint("x-2*y>=0")
        result = constraint.compute({"x": 10, "y": 2})
        assert result == pytest.approx(6.0)

    def test_missing_inequality_raises(self):
        with pytest.raises(SyntaxError):
            Constraint("x+y")

    def test_vectorized_computation(self):
        constraint = Constraint("x<=10")
        values = constraint.compute({"x": numpy.array([3.0, 15.0])})
        assert values == pytest.approx([7.0, -5.0])


class TestConstraintSet:
    def test_creation_and_parameters(self):
        c1 = Constraint("x<=10")
        c2 = Constraint("x+y>=1")
        constraint_set = ConstraintSet(c1, c2)
        assert len(constraint_set.constraints_list) == 2
        assert set(constraint_set.parameters) == {"x", "y"}

    def test_non_constraint_ignored_with_warning(self):
        c1 = Constraint("x<=10")
        with pytest.warns(UserWarning):
            constraint_set = ConstraintSet(c1, "not a constraint")
        assert len(constraint_set.constraints_list) == 1

    def test_declare_doe_constraint(self):
        x = PositiveParameter("x", [1, 100], "m", "length")
        y = PositiveParameter("y", [1, 100], "m", "width")
        parameter_set = PositiveParameterSet(x, y)
        constraint_set = ConstraintSet(Constraint("x<=10"), Constraint("x+y>=5"))
        func = constraint_set.declare_doe_constraint(parameter_set)
        X = numpy.array([[3.0, 4.0], [15.0, 4.0], [1.0, 1.0]])
        result = func(X)
        assert result.tolist() == [True, False, False]

    def test_declare_doe_constraint_undeclared_parameter(self):
        x = PositiveParameter("x", [1, 100], "m", "length")
        parameter_set = PositiveParameterSet(x)
        constraint_set = ConstraintSet(Constraint("x+z>=5"))
        assert constraint_set.declare_doe_constraint(parameter_set) == []

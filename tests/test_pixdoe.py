"""Unit tests for pyvplm.addon.pixdoe (DOE generation)."""

import numpy
import pytest

from pyvplm.addon.pixdoe import (
    apply_constraints,
    create_const_doe,
    create_doe,
    elect_nearest,
    find_nearest,
    surroundings,
)
from pyvplm.addon.variablepowerlaw import (
    buckingham_theorem,
    declare_func_x_to_pi,
    reduce_parameter_set,
)


@pytest.fixture
def bounds():
    return numpy.array([[10, 100], [100, 1000]], float)


class TestCreateDoe:
    def test_log_space_values(self, bounds):
        parameters_level = numpy.array([2, 3], int)
        doe_values, spacing = create_doe(bounds, parameters_level, True)
        expected = [
            [10, 100],
            [100, 100],
            [10, 316.22776601683796],
            [100, 316.22776601683796],
            [10, 1000],
            [100, 1000],
        ]
        numpy.testing.assert_allclose(doe_values, expected)
        assert spacing.tolist() == pytest.approx([1.0, 0.5])

    def test_linear_space_values(self, bounds):
        parameters_level = numpy.array([2, 2], int)
        doe_values, _ = create_doe(bounds, parameters_level, False)
        numpy.testing.assert_allclose(doe_values, [[10, 100], [100, 100], [10, 1000], [100, 1000]])

    def test_negative_bounds_log_space_rejected(self):
        bad_bounds = numpy.array([[-10, 100]], float)
        with pytest.raises(ValueError):
            create_doe(bad_bounds, numpy.array([2], int), True)

    def test_level_below_two_rejected(self, bounds):
        with pytest.raises(ValueError):
            create_doe(bounds, numpy.array([1, 3], int), True)

    def test_dimension_mismatch_rejected(self, bounds):
        with pytest.raises(ValueError):
            create_doe(bounds, numpy.array([2], int), True)

    def test_wrong_type_rejected(self, bounds):
        with pytest.raises(TypeError):
            create_doe(bounds.tolist(), numpy.array([2, 3], int), True)


class TestSurroundings:
    def test_all_nominal_points_covered(self, bounds):
        doe, _ = create_doe(bounds, numpy.array([4, 6], int), True)
        nominal_doe, proper_spacing = create_doe(bounds, numpy.array([2, 3], int), True)
        reduced, to_be_removed = surroundings(doe, nominal_doe, proper_spacing, True)
        assert reduced.shape == nominal_doe.shape
        assert not to_be_removed.any()

    def test_out_of_reach_points_removed(self, bounds):
        doe = numpy.array([[10.0, 100.0]])
        nominal_doe, proper_spacing = create_doe(bounds, numpy.array([2, 3], int), True)
        reduced, to_be_removed = surroundings(doe, nominal_doe, proper_spacing / 10, True)
        assert len(reduced) == 1
        assert to_be_removed.sum() == len(nominal_doe) - 1


class TestFindNearest:
    def test_indices_within_spacing(self, bounds):
        doe, _ = create_doe(bounds, numpy.array([4, 6], int), True)
        nominal_doe, proper_spacing = create_doe(bounds, numpy.array([2, 3], int), True)
        index = find_nearest(doe, nominal_doe, 2, proper_spacing, True)
        assert index.shape == (len(nominal_doe), 2)
        assert (index >= -1).all()
        # each nominal point has at least one feasible neighbour here
        assert (index[:, 0] != -1).all()

    def test_invalid_choice_nb(self, bounds):
        doe, _ = create_doe(bounds, numpy.array([4, 6], int), True)
        nominal_doe, proper_spacing = create_doe(bounds, numpy.array([2, 3], int), True)
        with pytest.raises(ValueError):
            find_nearest(doe, nominal_doe, 0, proper_spacing, True)


class TestElectNearest:
    def test_one_point_elected_per_nominal_point(self, bounds):
        doe, _ = create_doe(bounds, numpy.array([4, 6], int), True)
        nominal_doe, proper_spacing = create_doe(bounds, numpy.array([2, 3], int), True)
        index = find_nearest(doe, nominal_doe, 2, proper_spacing, True)
        doe_elected, reduced_nominal_doe = elect_nearest(doe, nominal_doe, index)
        assert len(doe_elected) == len(reduced_nominal_doe)
        assert len(doe_elected) <= len(nominal_doe)
        # elected points are unique
        assert len(numpy.unique(doe_elected, axis=0)) == len(doe_elected)


class TestApplyConstraints:
    def test_no_constraint_returns_all_true(self):
        X = numpy.zeros((5, 2))
        assert apply_constraints(X, []).tolist() == [True] * 5

    def test_valid_constraint_applied(self):
        X = numpy.array([[1.0], [3.0]])

        def constraint(x):
            return (x[:, 0] < 2).ravel()

        assert apply_constraints(X, constraint).tolist() == [True, False]

    def test_failing_constraint_ignored(self):
        X = numpy.zeros((3, 2))

        def bad_constraint(x):
            raise RuntimeError("boom")

        assert apply_constraints(X, bad_constraint).tolist() == [True] * 3


class TestCreateConstDoe:
    def test_end_to_end_cantilever(self, cantilever_parameter_set):
        parameter_set = cantilever_parameter_set
        pi_set, _ = buckingham_theorem(parameter_set, False)
        reduced_parameter_set, reduced_pi_set = reduce_parameter_set(parameter_set, pi_set, "u")
        func_x_to_pi = declare_func_x_to_pi(reduced_parameter_set, reduced_pi_set)
        doeXc, doePIc, *_ = create_const_doe(
            reduced_parameter_set, reduced_pi_set, func_x_to_pi, 20, test_mode=True
        )
        assert len(doeXc) >= 20
        assert doeXc.shape[1] == len(reduced_parameter_set.dictionary)
        assert doePIc.shape[1] == len(reduced_pi_set.dictionary)
        # the returned Pi DOE is consistent with the transformation function
        numpy.testing.assert_allclose(doePIc, func_x_to_pi(doeXc.tolist()))
        # physical values within SI bounds
        for idx, key in enumerate(reduced_parameter_set.dictionary.keys()):
            bounds = reduced_parameter_set[key]._SI_bounds
            assert doeXc[:, idx].min() >= bounds[0] * 0.999
            assert doeXc[:, idx].max() <= bounds[1] * 1.001

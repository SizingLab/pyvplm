# -*- coding: utf-8 -*-
"""
Addon module generating constrained full-factorial DOE on 2-spaces (pi/x) problems
"""

# -------[Import necessary packages]--------------------------------------------
import os
import sys
import pyDOE2
import numpy
import math
import logging
from typing import Tuple
from inspect import isfunction
import functools
import pandas
import matplotlib.pyplot as plot
import warnings
from numpy import ndarray
from pyvplm.core.definition import PositiveParameter, PositiveParameterSet, greek_list

# -------[Global variables and settings]----------------------------------------
path = os.path.abspath(__file__)
temp_path = path.replace("\\addon\\" + os.path.basename(path), "") + "\\_temp\\"
module_logger = logging.getLogger(__name__)


# -------[Logg Exception]-------------------------------------------------------
def logg_exception(ex: Exception):
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    module_logger.info(message)


# -------[Define function creating full-fact using bounds and levels]-----------
def create_doe(
    bounds: ndarray, parameters_level: ndarray, log_space: bool = True
) -> Tuple[ndarray, ndarray]:
    # noinspection PyUnresolvedReferences,PyShadowingNames
    """Functions that generates a fullfact DOE mesh using bounds and levels number.

    Parameters
    ----------
    bounds: [n*2] numpy.ndarray of floats, defines the n parameters [lower, upper] bounds

    parameters_level: [1*n] numpy.ndarray of int, defines the parameters levels

    log_space: Defines if fullfact has to be in log space or when false, linear (default is True)

    Returns
    -------
    doe_values: [m*n] numpy.ndarray of float
             A fullfact DOE, with n the number of parameters and m the number of experiments (linked to level
             repartition)

    spacing: [1*n] numpy.ndarray
          Represents the DOE's points spacing on each paramater axis in the space

    Example
    -------
    define bounds and parameters' levels:
     >>> In [1]: bounds = numpy.array([[10, 100], [100, 1000]], float)
     >>> In [2]: parameters_level = numpy.array([2, 3], int)

    generate doe in log space:
     >>> In [3]: doe_values, spacing = create_doe(bounds, parameters_level, True)

    returns:
     >>> In [4]: doe_values.tolist()
     >>> Out[4]: [[10, 100], [100, 100], [10, 316.228], [100, 316.228], [10, 1000], [100, 1000]]
     >>> In [5]: spacing.tolist()
     >>> Out[5]: [1.0, 0.5]

    """
    if (
        isinstance(bounds, numpy.ndarray)
        and isinstance(parameters_level, numpy.ndarray)
        and isinstance(log_space, bool)
    ):
        if log_space and numpy.amin(bounds) < 0:
            raise ValueError(
                "to translate on log space all bounds should be >0, else choose log_space = False."
            )
        if numpy.issubdtype(bounds.dtype, numpy.float64) and numpy.issubdtype(
            parameters_level.dtype, numpy.integer
        ):
            # Check that parameters levels syntax is correct
            if (numpy.size(parameters_level)) != (numpy.shape(bounds)[0]):
                raise ValueError("parameters_level and bounds dimensions mismatch.")
            # Check that parameters levels>=2
            if (sum(parameters_level >= 2) + sum(parameters_level == 0)) != numpy.size(
                parameters_level
            ):
                raise ValueError("parameters_level should be >=2.")
            # If log space transpose bounds in log space
            if log_space:
                bounds = numpy.log10(bounds)
            # Generate DOE on levels
            parameters_level = parameters_level + 1 * (parameters_level == 0)
            doe_levels = pyDOE2.fullfact(parameters_level).astype(int)
            for idx in range(numpy.shape(doe_levels)[1]):
                if sum(doe_levels[:, idx]) == 0:
                    doe_levels[:, idx] = 1
            # Translate levels into values x=xmin+level/max(level)*(xmax-xmin)
            doe_values = bounds[:, 0] + doe_levels / doe_levels.max(axis=0) * (
                bounds[:, 1] - bounds[:, 0]
            )
            # Calculate spacing in full-fact space (linear or log)
            spacing = 1 / doe_levels.max(axis=0) * (bounds[:, 1] - bounds[:, 0])
            # Transform calculated value from log to linear if necessary
            doe_values = 10**doe_values if log_space else doe_values
            return doe_values, spacing
        elif not (numpy.issubdtype(bounds.dtype, numpy.float64)):
            raise TypeError("elements type in in bounds should be float.")
        else:
            raise TypeError("elements type in in parameters_level should be integer.")
    elif not (isinstance(bounds, numpy.ndarray)):
        raise TypeError("bounds shoold be numpy array.")
    elif not (isinstance(parameters_level, numpy.ndarray)):
        raise TypeError("parameters_level shoold be numpy array.")
    else:
        raise TypeError("log_space shoold be boolean.")


# -------[Define function keeping nominal point if surrounded by feasible]------
def surroundings(
    doe, nominal_doe: ndarray, proper_spacing: ndarray, LogLin: bool = True
) -> Tuple[ndarray, ndarray]:
    # noinspection PyUnresolvedReferences,PyShadowingNames
    """Function to reduce a given nominal DOE on a max distance criteria with points from feasible DOE ('reachable'
    points).

    Parameters
    ----------
    doe: [m*n] numpy.ndarray of int or float
      DOE representing m feasible experiments expressed with n parameters with non-optimal spacing

    nominal_doe: [k*n] numpy.ndarray of int or float
              Fullfact DOE with k wished experiment (k<<m) expressed with the same n parameters

    proper_spacing: [n*1] numpy.ndarray of float
                 Represents max distance criteria on each DOE axis (i.e. parameter scale)

    LogLin: defines if fullfact has to be in log space or when false, linear (default is True)

    Returns
    -------
    reduced_nominal_doe: [l*n] numpy.ndarray
                      A reduced set of nominal_doe (l<=k) validating proper_spacing criteria with feasible
                      points from DOE

    to_be_removed: numpy.ndarray of bool
              Returns the corresponding indices that do not validate proper_spacing criteria

    Example
    -------
    define bounds and parameters' levels:
     >>> In [1]: bounds = numpy.array([[10, 100], [100, 1000]], float)
     >>> In [2]: parameters_level_nominal = numpy.array([2, 3], int)
     >>> In [3]: parameters_level_feasible = numpy.array([4, 6], int)

    generate doe(s) in log space:
     >>> In [4]: doe, _ = create_doe(bounds, parameters_level_feasible, True)
     >>> In [5]: nominal_doe, proper_spacing = create_doe(bounds, parameters_level_nominal, True)

    search surrounding points:
     >>> In [6]: reduced_nominal_doe, to_be_removed = surroundings(doe, nominal_doe, proper_spacing, True)
     >>> In [7]: reduced_nominal_doe.tolist()
     >>> Out[7]: [[10.0, 100.0], [100.0, 100.0], [10.0, 316.22776601683796], [100.0, 316.22776601683796],
     >>>    ...: [10.0, 1000.0], [100.0, 1000.0]]
     >>> In [8]: to_be_removed.tolist()
     >>> Out[8]: [False, False, False, False, False, False]

    """
    if (
        isinstance(doe, numpy.ndarray)
        and isinstance(nominal_doe, numpy.ndarray)
        and isinstance(proper_spacing, numpy.ndarray)
        and isinstance(LogLin, bool)
    ):
        # Proceed on type verifications
        if not (
            numpy.issubdtype(doe.dtype, numpy.float64) or numpy.issubdtype(doe.dtype, numpy.integer)
        ):
            raise TypeError("elements type in doe should be float or integer.")
        if not (
            numpy.issubdtype(nominal_doe.dtype, numpy.float64)
            or numpy.issubdtype(nominal_doe.dtype, numpy.integer)
        ):
            raise TypeError("elements type in nominal_doe should be float or integer.")
        if not (numpy.issubdtype(proper_spacing.dtype, numpy.float64)):
            raise TypeError("elements type in nominal_doe should be float or integer.")
        if not (
            numpy.shape(doe)[1] == numpy.shape(nominal_doe)[1]
            and numpy.shape(doe)[1] == len(proper_spacing)
        ):
            raise IndexError("column numbers mismatch between doe, nominal_doe and dmax.")
        # Transform DOE into log space if needed
        X = numpy.log10(doe) if LogLin else doe
        Y = numpy.log10(nominal_doe) if LogLin else nominal_doe
        # For each point in nominal_doe find if a point in doe at dmax distance in each dimension and canceled removal
        to_be_removed = numpy.ones(len(nominal_doe), bool)
        for y_idx in range(numpy.shape(Y)[0]):
            valid_distance = bool(
                numpy.sum(
                    (
                        numpy.sum((abs(X - Y[y_idx, :]) <= proper_spacing).astype(int), axis=1)
                        == len(proper_spacing)
                    ).astype(int)
                )
                >= 1
            )
            if valid_distance:
                to_be_removed[y_idx] = False
        return nominal_doe[to_be_removed == False], to_be_removed
    elif not (isinstance(doe, numpy.ndarray)):
        raise TypeError("doe should be numpy array.")
    elif not (isinstance(nominal_doe, numpy.ndarray)):
        raise TypeError("nominal_doe shoold be numpy array.")
    elif not (isinstance(proper_spacing, numpy.ndarray)):
        raise TypeError("proper_spacing shoold be numpy array.")
    else:
        raise TypeError("log_space shoold be boolean.")


# -------[Define function finding the choice_nb nearest points to nominal]------
def find_nearest(doe, nominal_doe, choice_nb, proper_spacing, log_space=True):
    # noinspection PyUnresolvedReferences
    """Function that returns for each point in nominal DOE point, the indices and max relative error for choice_nb
    nearest points in feasible DOE. As a distance has to be computed to select nearest in further functions, it is
    the max value of the relative errors (compared to the bounds) that is returned (this avoids infinite relative error
    for [0, 0] origin point).

    Parameters
    ----------
    doe: [m*n] numpy.ndarray of int or float
      DOE representing m feasible experiments expressed with n parameters with non-optimal spacing

    nominal_doe: [k*n] numpy.ndarray of int or float
              Fullfact DOE with k wished experiment (k<<m) expressed with the same n parameters

    choice_nb: int
            Number of the nearest points returned from DOE for each nominal DOE point, criteria is max relative
            distance error max(x-x_n/(max(x_n)-min(x_n)))

    proper_spacing: [n*1] numpy.ndarray of float
                     Represents max distance criteria on each DOE axis (i.e. parameter scale)

    log_space: bool
            Defines if fullfact has to be in log space or when false, linear (default is True)

    Returns
    -------
    nearest_index_in_doe: [k*choice_nb] numpy.ndarray of int
                        Gathers the corresponding 'choice_nb' nearest DOE points indices

    Example
    -------
    to define DOEs, see :func:`~sizinglab.addon.pixdoe.surroundings`

    then extract the 2 nearest feasible points for each nominal point:
     >>> In [6]: index, max_rel_distance = find_nearest(doe, nominal_doe, 2, proper_spacing, True)
     >>> In [7]: index.tolist()
     >>> Out[7]: [[0, 4], [3, 7], [8, 12], [11, 15], [20, 16], [23, 19]]
     >>> In [8]: max_rel_distance.tolist()
     >>> Out[8]: [[0.0, 0.20000000000000018], [0.0, 0.20000000000000018],
     [0.10000000000000009, 0.10000000000000009], [0.10000000000000009, 0.10000000000000009]
     [0.0, 0.20000000000000018], [0.0, 0.20000000000000018]]

    """
    if (
        isinstance(doe, numpy.ndarray)
        and isinstance(nominal_doe, numpy.ndarray)
        and isinstance(proper_spacing, numpy.ndarray)
        and isinstance(choice_nb, int)
        and isinstance(log_space, bool)
    ):
        # Proceed on type verifications
        if not (
            numpy.issubdtype(doe.dtype, numpy.float64) or numpy.issubdtype(doe.dtype, numpy.integer)
        ):
            raise TypeError("elements type in doe should be float or integer.")
        if not (
            numpy.issubdtype(nominal_doe.dtype, numpy.float64)
            or numpy.issubdtype(nominal_doe.dtype, numpy.integer)
        ):
            raise TypeError("elements type in nominal_doe should be float or integer.")
        if not (numpy.shape(doe)[1] == numpy.shape(nominal_doe)[1]):
            raise IndexError("column numbers mismatch between doe and nominal_doe.")
        if choice_nb < 1:
            raise ValueError("choice_nb numbers should be >= 1")
        # Initialise distance and index matrices
        i = 0
        nearest_index_in_doe = -1 * numpy.ones([numpy.shape(nominal_doe)[0], choice_nb], dtype=int)
        # If necessary convert data
        X = numpy.log10(doe) if log_space else doe
        Y = numpy.log10(nominal_doe) if log_space else nominal_doe
        # Find for each nominal PI value in DOE n<=choice_nb nearest points
        for x_value in Y:
            # Filter data to limit to the ones in the proper_spacing space envelope
            index = numpy.array(range(numpy.shape(X)[0]))
            reduced_index = index[
                numpy.sum((abs(X - x_value) <= proper_spacing).astype(int), axis=1)
                == len(proper_spacing)
            ]
            if len(reduced_index) <= choice_nb:
                nearest_index_in_doe[i, : len(reduced_index)] = reduced_index
            else:
                # If more than choice_nb point available, select the one with smaller relative distance
                reduced_X = X[reduced_index, :]
                rel_distance_matrix = (reduced_X - x_value) / (
                    numpy.amax(Y, axis=0) - numpy.amin(Y, axis=0)
                )
                rel_distance_vector = numpy.sum(rel_distance_matrix**2, axis=1) ** 0.5
                nearest_index_in_doe[i] = reduced_index[
                    numpy.argpartition(rel_distance_vector, choice_nb)[:choice_nb]
                ]
            i += 1
        return nearest_index_in_doe
    elif not (isinstance(doe, numpy.ndarray)):
        raise TypeError("doe shoold be numpy array.")
    elif not (isinstance(nominal_doe, numpy.ndarray)):
        raise TypeError("nominal_doe shoold be numpy array.")
    elif not (isinstance(choice_nb, numpy.ndarray)):
        raise TypeError("choice_nb shoold be an integer.")
    elif not (isinstance(proper_spacing, numpy.ndarray)):
        raise TypeError("proper_spacing shoold be numpy array.")
    else:
        raise TypeError("log_space shoold be boolean.")


# -------[Define function electing point by increasing occurrence]---------------
def elect_nearest(doe: ndarray, nominal_doe: ndarray, index: ndarray):
    # noinspection PyUnresolvedReferences,PyShadowingNames
    """Function that tries to assign for each point in nominal DOE, one point in feasible DOE elected from its
    'choice_nb' found indices. The assignments are done point-to-point electing each time the one maximizing minimum
    relative distance with current elected set. If from available indices they all are already in the set, point is
    deleted and thus: j<=k (not likely to happen).

    Parameters
    ----------
    doe: [m*n] numpy.ndarray of int or float
         DOE representing m feasible experiments expressed with n parameters with non-optimal spacing

    nominal_doe: [k*n] numpy.ndarray of int or float
                 Fullfact DOE with k wished experiment (k<<m) expressed with the same n parameters

    index: [k*nb_choice] numpy.ndarray of int
            Gathers the corresponding 'choice_nb' nearest DOE points indices (computed with :~pixdoe.find_nearest)

    Returns
    -------
    doe_elected: [j*n] numpy.ndarray of int or float
                 Returned DOE with feasible points assigned to reduced nominal DOE (deleted points with no
                 assignment, i.e. all indices already assigned)

    reduced_nominal_doe: [j*n] numpy.ndarray of int or float
                           Reduced nominal DOE (j<=k), all point are covered with feasible point

    Example
    -------
    to define DOEs and find the nearest points, see :func:`~sizinglab.addon.pixdoe.surroundings`

    then elect one point for each nominal point:
        >>> In [7]: doe_elected, reduced_nominal_doe, max_error = elect_nearest(doe, nominal_doe, index)
        >>> In [8]: doe_elected.tolist()
        >>> Out[8]: [[100.0, 100.0], [10.0, 251.18864315095797], [100.0, 251.18864315095797], [10.0, 1000.0],
        [100.0, 1000.0]]
        >>> In [9]: reduced_nominal_doe.tolist()
        >>> Out[9]: [[100.0, 100.0], [10.0, 316.22776601683796], [100.0, 316.22776601683796], [10.0, 1000.0],
        [100.0, 1000.0]]
        >>> In [10]: max_error.tolist()
        >>> Out[10]: [0.0, 0.10000000000000009, 0.10000000000000009, 0.0, 0.0]

    """
    if (
        isinstance(doe, numpy.ndarray)
        and isinstance(nominal_doe, numpy.ndarray)
        and isinstance(index, numpy.ndarray)
    ):
        # Proceed on type verifications
        if not (
            numpy.issubdtype(doe.dtype, numpy.float64) or numpy.issubdtype(doe.dtype, numpy.integer)
        ):
            raise TypeError("elements type in doe should be float or integer.")
        if not (
            numpy.issubdtype(nominal_doe.dtype, numpy.float64)
            or numpy.issubdtype(nominal_doe.dtype, numpy.integer)
        ):
            raise TypeError("elements type in nominal_doe should be float or integer.")
        if not (numpy.issubdtype(index.dtype, numpy.integer)):
            raise TypeError("elements type in index should be integer.")
        if numpy.shape(doe)[0] < numpy.amax(index):
            raise ValueError("maximum stored index is greater than doe size.")
        if numpy.shape(nominal_doe)[0] != numpy.shape(index)[0]:
            raise ValueError("nominal_doe and index should have same number of rows.")
        # Order matching by increasing number of available points
        available_index = numpy.sum(index != -1, axis=1)
        index = index[numpy.argsort(available_index), :]
        nominal_doe = nominal_doe[numpy.argsort(available_index), :]
        # Match 1-by-1 points maximizing minimum relative distance with elected_set
        index_elected = numpy.array([]).astype(int)
        doe_elected = numpy.array([]).astype(float)
        reduced_nominal_doe = numpy.array([]).astype(float)
        for nr in range(numpy.shape(index)[0]):
            # For 1st point elect first available index
            if len(doe_elected) == 0:
                index_elected = numpy.append(index_elected, index[nr, 0])
                doe_elected = doe[index[nr, 0], :]
                reduced_nominal_doe = nominal_doe[nr, :]
            # For other points elect from available index the one with maximum distance
            else:
                # Extract available index
                available_index = index[nr, :]
                available_index = available_index[available_index != -1]
                # Remove already elected index
                to_be_kept = numpy.ones(len(available_index)).astype(bool)
                for i in range(len(available_index)):
                    if available_index[i] in index_elected:
                        to_be_kept[i] = False
                available_index = available_index[to_be_kept]
                # If 1 point remaining take it, if more than one, elect it on max_min criteria
                if len(available_index) >= 1:
                    if len(available_index) > 1:
                        rel_distance = numpy.zeros(len(available_index)).astype(float)
                        doe_range = numpy.amax(doe, axis=0) - numpy.amin(doe, axis=0)
                        doe_range = doe_range + 1 * (doe_range == 0)
                        for i in range(len(available_index)):
                            try:
                                rel_distance[i] = numpy.amin(
                                    (
                                        numpy.sum(
                                            ((doe_elected - doe[available_index[i], :]) / doe_range)
                                            ** 2,
                                            axis=1,
                                        )
                                    )
                                    ** 0.5
                                )
                            except Exception as ex:
                                logg_exception(ex)
                                rel_distance[i] = (
                                    numpy.sum(
                                        ((doe_elected - doe[available_index[i], :]) / doe_range)
                                        ** 2
                                    )
                                    ** 0.5
                                )
                        available_index = available_index[numpy.argsort(-1 * rel_distance)]
                    index_elected = numpy.append(index_elected, available_index[0])
                    doe_elected = numpy.vstack([doe_elected, doe[available_index[0], :]])
                    reduced_nominal_doe = numpy.vstack([reduced_nominal_doe, nominal_doe[nr, :]])
        return doe_elected, reduced_nominal_doe
    elif not (isinstance(doe, numpy.ndarray)):
        raise TypeError("doe shoold be numpy array.")
    elif not (isinstance(nominal_doe, numpy.ndarray)):
        raise TypeError("nominal_doe shoold be numpy array.")
    else:
        raise TypeError("index shoold be numpy array.")


# -------[Define sub-function avoid script repetition, defines constrained DOEs]
# noinspection PyShadowingNames
def declare_does(
    x_Bounds: ndarray,
    x_levels: ndarray,
    parameters_constraints,
    pi_constraints,
    func_x_to_pi,
    log_space: bool = True,
):
    """Function to generate X and Pi DOE with constraints (called as sub-function script).

    Parameters
    ----------
    x_Bounds: [n*2] numpy.ndarray of floats, defines the n parameters [lower, upper] bounds

    x_levels: [1*n] numpy.ndarray of int, defines the parameters levels

    parameters_constraints, pi_constraints: functions that define parameter and Pi constraints

    func_x_to_pi: function that translates X physical values into Pi dimensionless values
                    (space transformation matrix)

    log_space: Defines if full-factorial has to be in log space or when false, linear (default is True)

    Returns
    -------
    doeX: [m*n] numpy.ndarray of float
                A full-factorial DOE, with n the number of parameters and m the number of experiments (linked to level)

    doePI: [k*n] numpy.ndarray of float
             Represents the Pi DOE's points computed from doeX and applying both X and Pi constraints (k<=m)

    """
    doe_x, _ = create_doe(x_Bounds, x_levels, log_space)
    doe_x = doe_x[apply_constraints(doe_x, parameters_constraints) == True]
    if len(doe_x) == 0:
        doePI = []
    else:
        doePI = func_x_to_pi(doe_x.tolist())
        doePI = doePI[apply_constraints(doePI, pi_constraints) == True, :]
    return doe_x, doePI


# -------[Main function: create physical points matching nominal Pi DOE]--------
# noinspection PyShadowingNames
def create_const_doe(
    parameter_set: PositiveParameterSet,
    pi_set: PositiveParameterSet,
    func_x_to_pi,
    wished_size: int,
    **kwargs
):
    # noinspection PyUnresolvedReferences,PyTypeChecker,PyShadowingNames
    """Function to generate a constrained feasible set DOE with repartition on PI not far from nominal fullfact DOE.

    Parameters
    ----------
    parameter_set: Defines the n physical parameters for the studied problem

    pi_set: Defines the k (k<n) dimensionless parameters of the problem (WARNING: no cross-validation with
            parameter_set, uses func_x_to_pi for translation)

    func_x_to_pi: Function that translates X physical values into Pi dimensionless values (space transformation matrix)

    wished_size: Is the wished size of the final elected X-DOE that represents a constrained fullfact Pi-DOE

    **kwargs: additional argumens
                 * **level_repartition** (*numpy.array* of *int*): defines the parameters levels relative repartition,
                 default is equaly shared (same number of levels)
                 * **parameters_constraints** (*function*): returns numpy.array of bool to validate each point in X-DOE,
                 default is []
                 * **pi_constraints** (*function*): returns numpy.array of bool to validate each point in Pi-DOE,
                 default is []
                 * **choice_nb** (*int*): number of returned nearest point from DOE for each nominal DOE point,
                 default is 3
                 * **spacing_division_criteria** (*int*): (>=2) defines the subdivision admitted error in Pi nominal
                 space for feasible point, default is 5
                 * **log_space** (*bool*): defines if fullfact has to be in log space or when false, linear
                 (default is log - True)
                 * **track** (*bool*): defines if the different process steps information have to be displayed
                 (default is False)
                 * **test_mode** (*bool*): set to False to show plots (default is False)

    Returns
    -------
    doeXc: [j*n] numpy.array of float
           Represents the elected feasible constrained sets of physical parameters matching spacing criteria with
           j >= whished_size

    doePIc: [j*n] numpy.array of float
            Represents the elected feasible constrained sets of dimensionless parameters matching spacing criteria
            with j >= whished_size

    Example
    -------
    define properly the parameter, pi set and transformation function:
        >>> In [1]: from pyvplm.core.definition import PositiveParameter, PositiveParameterSet
        >>> In [2]: from pyvplm.addon.variablepowerlaw import buckingham_theorem, declare_func_x_to_pi,
        reduce_parameter_set
        >>> In [3]: u = PositiveParameter('u', [1e-9, 1e-6], 'm', 'Deflection')
        >>> In [4]: f = PositiveParameter('f', [150, 500], 'N', 'Load applied')
        >>> In [5]: l = PositiveParameter('l', [1, 3], 'm', 'Cantilever length')
        >>> In [6]: e = PositiveParameter('e', [60e9, 80e9], 'Pa', 'Young Modulus')
        >>> In [7]: d = PositiveParameter('d', [10, 60], 'mm', 'Diameter of cross-section')
        >>> In [8]: parameter_set = PositiveParameterSet(u, f, l, e, d)
        >>> In [9]: parameter_set.first('u','l')
        >>> In [10]: pi_set, _ = buckingham_theorem(parameter_set, False)
        >>> In [11]: reduced_parameter_set, reduced_pi_set = reduce_parameter_set(parameter_set, pi_set, 'l')
        >>> In [12]: func_x_to_pi = declare_func_x_to_pi(reduced_parameter_set, reduced_pi_set)

    then create a complete DOE:
        >>> In [13]: doeXc, doePIc = create_const_doe(reduced_parameter_set, reduced_pi_set, func_x_to_pi, 30,
        track=True)

        .. image:: ../source/_static/Pictures/pixdoe_create_const_doe1.png
        .. image:: ../source/_static/Pictures/pixdoe_create_const_doe2.png
    """
    # Proceed on type verifications
    if (
        isinstance(parameter_set, PositiveParameterSet)
        and isinstance(pi_set, PositiveParameterSet)
        and isfunction(func_x_to_pi)
        and isinstance(wished_size, int)
    ):
        # Set additional arguments values
        level_repartition = numpy.ones(len(list(parameter_set.dictionary.keys()))).astype(int)
        parameters_constraints = []
        pi_constraints = []
        choice_nb = 3
        spacing_division_criteria = 5
        log_space = True
        track = False
        test_mode = False
        for key, value in kwargs.items():
            if not (
                key
                in [
                    "level_repartition",
                    "parameters_constraints",
                    "pi_constraints",
                    "choice_nb",
                    "spacing_division_criteria",
                    "log_space",
                    "track",
                    "test_mode",
                ]
            ):
                raise KeyError("unknown argument " + key)
            elif key == "level_repartition":
                if isinstance(value, numpy.ndarray):
                    if len(value) == len(list(parameter_set.dictionary.keys())):
                        level_repartition = value
                        for level in level_repartition:
                            if isinstance(level, int):
                                if level < 1:
                                    raise ValueError(
                                        "each level in level_repartition should be >1."
                                    )
                            else:
                                raise ValueError(
                                    "each level in level_repartition should be an integer."
                                )
                    else:
                        raise ValueError("level_repartition mismatch parameter_set keys number.")
                else:
                    raise TypeError("level_repartition should be a numpy array.")
            elif key == "parameters_constraints":
                if isfunction(value):
                    parameters_constraints = value
                else:
                    raise TypeError("parameters_constraints should be a function.")
            elif key == "pi_constraints":
                if isfunction(value):
                    pi_constraints = value
                else:
                    raise TypeError("pi_constraints should be a function.")
            elif key == "choice_nb":
                if isinstance(value, int):
                    if value <= 0:
                        ValueError("choice_nb should be >=1.")
                    else:
                        choice_nb = value
                else:
                    raise TypeError("choice_nb should be an integer.")
            elif key == "spacing_division_criteria":
                if isinstance(value, int):
                    if value <= 1:
                        ValueError("spacing_division_criteria should be >=2.")
                    else:
                        spacing_division_criteria = value
                else:
                    raise TypeError("spacing_division_criteria should be an integer.")
            elif key == "log_space":
                if isinstance(value, bool):
                    log_space = value
                else:
                    raise ValueError("log_space should be a boolean.")
            elif key == "track":
                if isinstance(value, bool):
                    track = value
                else:
                    raise ValueError("track should be a boolean.")
            elif key == "test_mode":
                if isinstance(value, bool):
                    test_mode = value
                else:
                    raise ValueError("test_mode should be a boolean.")
        # Extract bounds on parameters set and parameters number
        x_Bounds = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in parameter_set.dictionary.keys():
                # noinspection PyProtectedMember
                x_Bounds.append(parameter_set[index]._SI_bounds)
        x_Bounds = numpy.array(x_Bounds)
        parameters_number = numpy.shape(x_Bounds)[0]
        # Extract bounds on pi set and pi number
        pi_Bounds = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in pi_set.dictionary.keys():
                # noinspection PyProtectedMember
                pi_Bounds.append(pi_set[index]._SI_bounds)
        pi_Bounds = numpy.array(pi_Bounds)
        pi_number = numpy.shape(pi_Bounds)[0]
        # Check func_x_to_pi function
        try:
            pi = func_x_to_pi(numpy.transpose(x_Bounds))
            if numpy.shape(pi)[1] != len(pi_Bounds):
                raise TypeError
        except Exception:
            raise IndexError(
                "func_x_to_pi can't be used to translate physical parameters into dimensionless ones."
            )
        # Proceed on size verifications and value type
        if numpy.shape(level_repartition)[0] != parameters_number:
            raise IndexError("level_repartition index differs from parameters in parameter_set.")
        if choice_nb <= 0:
            raise ValueError("choice_nb should be >= 1.")
        if not (numpy.issubdtype(level_repartition.dtype, numpy.integer)):
            raise TypeError("level_repartition type in index should be integer.")

        # Define factorisation for population calculation

        # noinspection PyShadowingNames
        def fact_level(x):
            for idx in range(len(x)):
                if x[idx] == 0:
                    x[idx] = 1
            y = functools.reduce(lambda x, y: x * y, x)
            return y

        # Define DOE level name
        # noinspection PyShadowingNames
        def level_name(x):
            name = ""
            for idx in range(len(x)):
                if x[idx] == 0:
                    x[idx] = 1
                name += str(x[idx]) + "x"
            name = name[0 : len(name) - 1] + "=" + str(fact_level(x))
            return name

        # Save level repartition as x_levels and adapt it for constant parameter
        x_levels = level_repartition
        i = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in parameter_set.dictionary.keys():
                # noinspection PyProtectedMember
                if parameter_set[index]._SI_bounds[0] == parameter_set[index]._SI_bounds[1]:
                    x_levels[i] = 0
                i += 1
        min_level = min(x_levels + sys.maxsize * (x_levels == 0))
        for idx in range(len(x_levels)):
            x_levels[idx] = int(1 / min_level * x_levels[idx])
        # Adapt values
        if wished_size < 2**pi_number:
            wished_size = 2**pi_number
            warnings.warn(
                "Experiments size changed to {} to obtain 2-levels full-fractional on Pi parameters".format(
                    wished_size
                )
            )
        # Create level repartition on pi and adapt it for constant parameter
        pi_levels = numpy.ones(pi_number, dtype=int)
        i = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in pi_set.dictionary.keys():
                # noinspection PyProtectedMember
                if pi_set[index]._SI_bounds[0] == pi_set[index]._SI_bounds[1]:
                    pi_levels[i] = 0
                i += 1
        # Set an initial set point on X 3 times greater than the wished constrained set (size will be auto. adjusted)
        init_coverage_factor = 3
        x_steps = 2
        while fact_level(x_steps * x_levels) < (init_coverage_factor * wished_size):
            x_steps += 1
        x_steps += -1
        # Force entry to loop and initiate pi_level
        obtained_size_on_pi = 0
        pi_steps = 0
        save = {}
        # Starts automatic definition of initial X non-constrained set and nominal Pi set to have sufficient constrained
        # nominal Pi set
        while obtained_size_on_pi < wished_size:
            # Init variables before entering X-DOE automatic loop
            step = 1
            obtained_size_on_x = 0
            # [PHASE1] Loop increasing x parameters' level until obtaining a constrained set
            # size >= wished_size * init_coverage_factor [CAN BE SLOW]
            if track:
                print(
                    "PHASE1: Constructing constrained X-DOE based on size >= {} criteria".format(
                        wished_size * init_coverage_factor
                    )
                )
            while obtained_size_on_x < (wished_size * init_coverage_factor):
                x_steps += 1
                doeX, doePI = declare_does(
                    x_Bounds,
                    x_steps * x_levels,
                    parameters_constraints,
                    pi_constraints,
                    func_x_to_pi,
                    log_space,
                )
                save["doePI"] = doePI
                obtained_size_on_x = numpy.shape(doeX)[0]
                if track:
                    print(
                        "Step{}: non constrained {} X-DOE factorial experiment leads to constrained [{}*{}] "
                        "X-DOE matrix".format(
                            step,
                            level_name(x_steps * x_levels),
                            obtained_size_on_x,
                            parameters_number,
                        )
                    )
                    if not (obtained_size_on_x < (wished_size * init_coverage_factor)):
                        print("Skipping to PHASE2...\n")
                step += 1
            # Calculate the equivalent init_coverage_factor for obtained parameters_level such as constrained doe
            # size >= wished_size * init_coverage_factor
            init_coverage_factor = math.trunc(fact_level(x_steps * x_levels) / wished_size)
            # Init variables before entering nominal PI-DOE automatic loop
            step = 1
            previous_size = 0
            obtained_size_on_pi = 0
            pi_steps = 2
            while fact_level(pi_steps * pi_levels) < wished_size:
                pi_steps += 1
            pi_steps += -1
            # Loop increasing nominal pi parameters' level till obtaining a constrained set
            # size >= wished_size [CAN BE SLOW]
            if track:
                print(
                    "PHASE2: Constructing unconstrained Pi-DOE validating max spacing criteria = 1/{} "
                    "with feasible points".format(spacing_division_criteria)
                )
            while obtained_size_on_pi < wished_size:
                pi_steps += 1
                # If nominal pi set becomes greater than feasible pi set increase parameter level
                # (i.e. generated constrained X-DOE)
                if obtained_size_on_x < fact_level(pi_steps * pi_levels):
                    if track:
                        print(
                            "Step{}: [ERROR] Pi set size would tend to be greater than constrained X set: restart "
                            "in PHASE1 to increase X parameters levels\n".format(step)
                        )
                    break
                else:
                    doePIn, spacing = create_doe(pi_Bounds, pi_steps * pi_levels, log_space)
                    save["doePIn"] = doePIn
                    doePIn = doePIn[apply_constraints(doePIn, pi_constraints) == True]
                    save["doePIn_c"] = doePIn
                    # noinspection PyUnboundLocalVariable
                    doePIn, _ = surroundings(
                        doePI, doePIn, spacing / spacing_division_criteria, log_space
                    )
                    save["doePIn_a"] = doePIn
                    obtained_size_on_pi = numpy.shape(doePIn)[0]
                    if previous_size > obtained_size_on_pi:
                        if track:
                            print(
                                "Step{}: [ERROR] Pi set size decreasing while increasing levels {}->{}: restart "
                                "in PHASE1 to increase X parameters levels\n".format(
                                    step, previous_size, obtained_size_on_pi
                                )
                            )
                        break
                    else:
                        if track:
                            print(
                                "Step{}: non constrained [{}] Pi-DOE factorial experiment leads to constrained "
                                "[{}*{}] Pi-DOE matrix".format(
                                    step,
                                    level_name(pi_steps * pi_levels),
                                    obtained_size_on_pi,
                                    pi_number,
                                )
                            )
                previous_size = obtained_size_on_pi
                step += 1
        # From initial nominal Pi set and constrained X set extract the nearest points
        # noinspection PyUnboundLocalVariable
        index = find_nearest(
            doePI, doePIn, choice_nb, spacing / spacing_division_criteria, log_space
        )
        index_vector = (
            numpy.reshape(index, numpy.shape(index)[0] * choice_nb) if choice_nb != 1 else index
        )
        save["doePI_n"] = doePI[index_vector, :]
        # noinspection PyUnboundLocalVariable
        doeXn, doePIn = elect_nearest(doeX, doePIn, index)
        # Delete points that do not match spacing criteria
        doePI, to_be_removed = surroundings(
            doePIn, func_x_to_pi(doeXn), spacing / spacing_division_criteria, log_space
        )
        doeXn = doeXn[to_be_removed == False]
        reduction_factor = 1 - len(doeXn) / fact_level(x_steps * x_levels)
        if not test_mode:
            print("\n")
            print(
                "Set reduction factor (from feasible to optimal) is {}%\n".format(
                    round(reduction_factor * 10000) / 100
                )
            )
        # Calculate pi-DOE from elected X-DOE set
        doeXc = doeXn
        doePIc = func_x_to_pi(doeXc.tolist())
        save["doePI_e"] = doePIc
        # Plot Pi vs Pi full-factorial graphs
        if not test_mode:
            X = numpy.log10(save["doePI"]) if log_space else save["doePI"]
            X1 = numpy.log10(save["doePI_n"]) if log_space else save["doePI_n"]
            X2 = numpy.log10(save["doePI_e"]) if log_space else save["doePI_e"]
            Y = numpy.log10(save["doePIn_c"]) if log_space else save["doePIn_c"]
            Y1 = numpy.log10(save["doePIn_a"]) if log_space else save["doePIn_a"]
            x_labels = list(pi_set.dictionary.keys())
            graph_nb = 0
            for i in range(numpy.shape(Y)[1] - 1):
                for k in range(i + 1, numpy.shape(Y)[1]):
                    graph_nb += 1
            n = math.ceil(graph_nb**0.5)
            fig, axes = plot.subplots(n, n, figsize=(6 * n, 6 * n))
            graph_idx = 0
            for i in range(numpy.shape(Y)[1] - 1):
                for k in range(i + 1, numpy.shape(Y)[1]):
                    if graph_nb == 1:
                        axes_handle = axes
                    else:
                        nr = math.floor(graph_idx / n)
                        nc = graph_idx - nr * n
                        axes_handle = axes[nr, nc]
                    axes_handle.plot(X[:, i], X[:, k], "g.", label="All (Feas.)")
                    axes_handle.plot(
                        X1[:, i], X1[:, k], "c.", label="{}-nearest (Feas.)".format(choice_nb)
                    )
                    axes_handle.plot(X2[:, i], X2[:, k], "b.", label="Elected (Feas.)")
                    axes_handle.plot(Y[:, i], Y[:, k], "k.", label="All (Obj.)")
                    axes_handle.plot(Y1[:, i], Y1[:, k], "r.", label="Active (Obj.)")
                    expression = (
                        ("$log(" + x_labels[i].replace("pi", "\\pi_{") + "})$")
                        if log_space
                        else (x_labels[i].replace("pi", "$\\pi_{") + "}$")
                    )
                    axes_handle.set_xlabel(expression)
                    expression = (
                        ("$log(" + x_labels[k].replace("pi", "\\pi_{") + "})$")
                        if log_space
                        else (x_labels[k].replace("pi", "$\\pi_{") + "}$")
                    )
                    axes_handle.set_ylabel(expression)
                    axes_handle.legend()
                    ymax = max(numpy.amax(X[:, k]), numpy.amax(Y[:, k]))
                    ymin = min(numpy.amin(X[:, k]), numpy.amin(Y[:, k]))
                    xmax = max(numpy.amax(X[:, i]), numpy.amax(Y[:, i]))
                    xmin = min(numpy.amin(X[:, i]), numpy.amin(Y[:, i]))
                    try:
                        x_lines = (pi_steps - 1) * pi_levels[k] * spacing_division_criteria + 1
                        axes_handle.set_xticks(numpy.linspace(xmin, xmax, x_lines))
                    except Exception as ex:
                        logg_exception(ex)
                    axes_handle.xaxis.set_ticklabels([])
                    try:
                        y_lines = (pi_steps - 1) * pi_levels[i] * spacing_division_criteria + 1
                        axes_handle.set_yticks(numpy.linspace(ymin, ymax, y_lines))
                    except Exception as ex:
                        logg_exception(ex)
                    axes_handle.yaxis.set_ticklabels([])
                    axes_handle.grid()
                    axes_handle.set_ylim((ymin, ymax))
                    axes_handle.set_xlim((xmin, xmax))
                    graph_idx += 1
            while graph_idx < n**2:
                nr = math.floor(graph_idx / n)
                nc = graph_idx - nr * n
                axes[nr, nc].axis("off")
                graph_idx += 1
            try:
                plot.savefig(temp_path + "create_const_doe_fig1.pdf", dpi=1200, format="pdf")
            except Exception as ex:
                logg_exception(ex)
            plot.show()
            # Plot x elected vs x constrained full-fact graphs (only for variables)
            X = numpy.log10(doeXc) if log_space else doeXc
            Y = numpy.log10(doeX) if log_space else doeX
            Y_range = numpy.amax(Y, axis=0) - numpy.amin(Y, axis=0)
            Y_range = Y_range + 1 * (Y_range == 0)
            X = (X - numpy.amin(Y, axis=0)) / Y_range
            for i in range(numpy.shape(X)[1]):
                if numpy.amax(X[:, i]) == 0 and numpy.amin(X[:, i]) == 0:
                    X[:, i] = 0.5 * (X[:, i] == 0)
                    continue
            x_labels = []
            for parameter_name in parameter_set.dictionary.keys():
                if len(parameter_name.replace("_", "")) == (len(parameter_name) - 1):
                    terms = parameter_name.split("_")
                    if terms[0] in greek_list:
                        if terms[0] == terms[0].upper():
                            terms[0] = terms[0][0] + terms[0][1:].lower()
                        parameter_name = "\\" + terms[0]
                    else:
                        if terms[0].lower() in greek_list:
                            parameter_name = "\\" + terms[0].lower()
                        else:
                            parameter_name = terms[0]
                    if terms[1] in greek_list:
                        if terms[1] == terms[1].upper():
                            terms[1] = terms[1][0] + terms[1][1:].lower()
                        parameter_name += "_{\\" + terms[1] + "}"
                    else:
                        if terms[1].lower() in greek_list:
                            parameter_name += "_{\\" + terms[1].lower() + "}"
                        else:
                            parameter_name += "_{" + terms[1] + "}"
                else:
                    parameter_name = parameter_name.replace("_", "")
                    if parameter_name in greek_list:
                        if parameter_name == parameter_name.upper:
                            parameter_name = parameter_name[0] + parameter_name[:-1].lower()
                        parameter_name = "\\" + parameter_name
                x_labels.append("$" + parameter_name + "^{*}$")
            if log_space:
                title_name = "$X^{*}=\\frac{log(X)-min(log(X))}{max(log(X))-min(log(X))}$"
            else:
                title_name = "$X^{*}=\\frac{X-min(X)}{max(X)-min(X)}$"
            X_data = pandas.DataFrame(X, columns=x_labels)
            X_data["Name"] = "Feasible point"
            plot.figure(figsize=(2 * (len(x_labels) - 1), 5))
            pandas.plotting.parallel_coordinates(X_data, "Name")
            plot.xticks(fontsize=12)
            plot.title(label=title_name)
            try:
                plot.savefig(temp_path + "create_const_doe_fig2.pdf", dpi=1200, format="pdf")
            except Exception as ex:
                logg_exception(ex)
            plot.show()
        return doeXc, doePIc, save["doePI"], save["doePI_n"], save["doePIn_c"], save["doePIn_a"]
    elif not (isinstance(parameter_set, PositiveParameterSet)):
        raise TypeError("level_repartition type should be PositiveParameterSet.")
    elif not (isinstance(pi_set, PositiveParameterSet)):
        raise TypeError("pi_set type should be PositiveParameterSet.")
    elif not (isfunction(func_x_to_pi)):
        raise TypeError("func_x_to_pi should be a function.")
    else:
        raise TypeError("wished_size should be an integer.")


# -------[Wrap constraint function to avoid definition error: unconstrained]----
def apply_constraints(X: ndarray, Constraints):
    """Function to test declared constraint and return true vector if an error occurs.

    Parameters
    ----------
    X: [m*n] numpy.ndarray of float or int
       Defines the m DOE points values over the n physical parameters

    Constraints: function that should return a [1*m] numpy.ndarray of bool, that validates the m points constraint

    Returns
    -------
    Constraints(X): [1*m] numpy.ndarray of bool
                    If dimension mismatch or constraint can't be applied returns True values (no constraint applied)

    """
    # Test if some constraints are declared
    if isfunction(Constraints):
        try:  # Remove point that does not respect X constraints
            Y = Constraints(X)
            if len(Y) == len(X):
                return Constraints(X)
            else:
                print("Error applying constraints: constraints not applied!")
        except Exception as ex:
            logg_exception(ex)
            print("Error applying constraints: constraints not applied!")
    return numpy.ones(len(X), dtype=bool)

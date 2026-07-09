# -*- coding: utf-8 -*-
"""
Core module defining elementary class and methods for SizingLab
"""

# -------[Import necessary packages]--------------------------------------------
import sys
import pint
import warnings
import logging
import sympy
import numpy
import copy
from collections import OrderedDict
from inspect import isfunction
from IPython.display import display, Math

module_logger = logging.getLogger(__name__)


# -------[Greek list definition]------------------------------------------------
# noinspection SpellCheckingInspection
greek_list = [
    "gamma",
    "delta",
    "eta",
    "theta",
    "lambda",
    "xi",
    "pi",
    "sigma",
    "upsilon",
    "phi",
    "psi",
    "omega",
]
for idx in range(len(copy.deepcopy(greek_list))):
    greek_list.append(greek_list[idx].upper())
greek_list.extend(
    [
        "alpha",
        "beta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "vartheta",
        "iota",
        "kappa",
        "mu",
        "nu",
        "rho",
        "varrho",
        "tau",
        "varphi",
        "chi",
    ]
)


# -------[Logg Exception]-------------------------------------------------------
def logg_exception(ex: Exception):
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    module_logger.info(message)


# -------[Parameter Class Definition]-------------------------------------------
class Parameter:
    # noinspection PyUnresolvedReferences,GrazieInspection
    """Class defining one physical parameter.

    Attributes
    ----------
    name: str
          the parameter name as a convention will be converted to :
            - upper char(s) for constant
            - lower char(s) for variable

    defined_bounds: [1*2] list of int or float
                    converted to float and checked that defined_bounds[0]<defined_bounds[1], set to [] for constant

    value: int or float
           converted to float, set to [] for variable

    defined_units: str
                   the parameters defined units (expression checked using PINT package)

    description: str
                 some description on the parameter

    _SI_bounds (private): [1*2] list of float
                          the parameter bounds expressed into SI units automatically derived from defined bounds
                          using PINT package

    _SI_units (private): str
                         SI units are automatically derived from defined units using PINT package

    _dimensionality (private): str
                               the parameter dimensions derived from units using PINT package (ex: meter->[length])

    Examples
    --------
    save a parameter:
        >>> In [1]: m = Parameter('m', [50, 100], 'kg', 'mass')

    save a constant:
        >>> In [2]: k = Parameter('k', [2], '', 'oversizing coefficient')

    get k parameter value in equation:
        >>> In [3]: a = float(k)*2.0
        >>> Out[3]: 4.0

    print parameters'attributes:
        >>> In [4]: print(m)
        m.name=m
        m.defined_bounds=[50.0, 100.0]
        ...
        m._dimensionality=[mass]

    change parameter's attribute values:
        >>> In [5]: m.description = 'body mass'

    """

    ureg = pint.UnitRegistry()

    def __init__(self, name: str, defined_bounds: list, defined_units: str, description: str):
        """Method to create initial parameter object using syntax expressed in example."""
        # Check input types
        if not (
            isinstance(name, str)
            and isinstance(defined_bounds, list)
            and isinstance(defined_units, str)
            and isinstance(description, str)
        ):
            raise TypeError("attributes type mismatch class definition")
        # Check units syntax
        proper_syntax, formatted_units, dimensionality = self.check_units(defined_units)
        # Check bounds syntax and values
        proper_syntax = proper_syntax and self.check_bounds(defined_bounds)
        # Save parameter attributes' values
        if proper_syntax:
            if len(defined_bounds) == 2 or formatted_units == "dimensionless":
                self.name = name.lower()
            else:
                self.name = name.upper()
            if len(defined_bounds) == 2:
                lower_bound = float(defined_bounds[0])
                upper_bound = float(defined_bounds[1])
                object.__setattr__(self, "defined_bounds", [lower_bound, upper_bound])
                self.value = []
            else:
                object.__setattr__(self, "defined_bounds", [])
                self.value = defined_bounds
            object.__setattr__(self, "defined_units", formatted_units)
            self.description = description
            ureg = self.ureg
            ureg.default_system = "mks"
            Q_ = ureg.Quantity
            if len(defined_bounds) == 2:
                # noinspection PyUnboundLocalVariable
                SI_lower_bound = Q_(lower_bound, formatted_units).to_base_units()
                # noinspection PyUnboundLocalVariable
                SI_upper_bound = Q_(upper_bound, formatted_units).to_base_units()
                object.__setattr__(
                    self, "_SI_bounds", [SI_lower_bound.magnitude, SI_upper_bound.magnitude]
                )
            else:
                SI_lower_bound = Q_(defined_bounds[0], formatted_units).to_base_units()
                object.__setattr__(
                    self, "_SI_bounds", [SI_lower_bound.magnitude, SI_lower_bound.magnitude]
                )
            object.__setattr__(self, "_SI_units", str(SI_lower_bound.units))
            object.__setattr__(self, "_dimensionality", dimensionality)

    def __getattribute__(self, attribute_name):
        """Method to access parameter attribute value (for private attributes, warning is displayed).
        Access is granted using command: **parameter_name.attribute_name**.

        """
        if attribute_name in ["_SI_units", "_SI_bounds", "_dimensionality"]:
            warnings.warn("accessing private attribute value")
        return super(Parameter, self).__getattribute__(attribute_name)

    def __setattr__(self, attribute_name, value, check_units=True):
        """Method to write parameter attribute value, **parameter_name.attribute_name=value** (private attribute writing
        access denied).

        """
        if attribute_name == "defined_units":
            # Check units syntax
            proper_syntax, formatted_units, dimensionality = self.check_units(value)
            if proper_syntax:
                ureg = self.ureg
                ureg.default_system = "mks"
                Q_ = ureg.Quantity
                object.__setattr__(self, "_dimensionality", dimensionality)
                object.__setattr__(self, "defined_units", formatted_units)
                if len(self.value) == 0:
                    SI_lower_bound = Q_(self.defined_bounds[0], formatted_units).to_base_units()
                    SI_upper_bound = Q_(self.defined_bounds[1], formatted_units).to_base_units()
                else:
                    SI_lower_bound = Q_(self.value, formatted_units).to_base_units()
                    SI_upper_bound = SI_lower_bound
                object.__setattr__(
                    self, "_SI_bounds", [SI_lower_bound.magnitude, SI_upper_bound.magnitude]
                )
        elif attribute_name == "defined_bounds":
            defined_bounds = value
            # Check bounds syntax and values
            proper_syntax = self.check_bounds(defined_bounds)
            # Get units
            formatted_units = self.defined_units
            # Save parameter bounds values
            if proper_syntax:
                if len(defined_bounds) == 2:
                    object.__setattr__(self, "name", self.name.lower())
                    lower_bound = float(defined_bounds[0])
                    upper_bound = float(defined_bounds[1])
                    object.__setattr__(self, "defined_bounds", [lower_bound, upper_bound])
                    object.__setattr__(self, "value", [])
                else:
                    object.__setattr__(self, "name", self.name.upper())
                    object.__setattr__(self, "defined_bounds", [])
                    object.__setattr__(self, "value", defined_bounds)
                ureg = self.ureg
                ureg.default_system = "mks"
                Q_ = ureg.Quantity
                if len(defined_bounds) == 2:
                    # noinspection PyUnboundLocalVariable
                    SI_lower_bound = Q_(lower_bound, formatted_units).to_base_units()
                    # noinspection PyUnboundLocalVariable
                    SI_upper_bound = Q_(upper_bound, formatted_units).to_base_units()
                    object.__setattr__(
                        self, "_SI_bounds", [SI_lower_bound.magnitude, SI_upper_bound.magnitude]
                    )
                else:
                    SI_lower_bound = Q_(defined_bounds[0], formatted_units).to_base_units()
                    object.__setattr__(
                        self, "_SI_bounds", [SI_lower_bound.magnitude, SI_lower_bound.magnitude]
                    )
        elif attribute_name in [
            "name",
            "value",
            "description",
            "_SI_units",
            "_SI_bounds",
            "_dimensionality",
        ]:
            object.__setattr__(self, attribute_name, value)
        else:
            raise AttributeError("nonexistent attribute or private (write access denied)")

    def __float__(self):
        """Method to return parameter value with syntax **float(parameter_name)**.
        If value is empty (i.e. parameter is a variable), returns NaN.

        """
        if len(self.value) == 0:
            return float("nan")
        else:
            return float(self.value[0])

    def __str__(self):
        """Method used to print parameter, called with **print(parameter_name)** function."""
        statement = ""
        for key in self.__dict__.keys():
            if key[0:1] == "_":
                statement += self.name + "." + key + "(private) = " + str(self.__dict__[key]) + "\n"
            else:
                statement += self.name + "." + key + " = " + str(self.__dict__[key]) + "\n"
        return statement

    def __repr__(self):
        """Method to represent parameter definition when entering only parameter_name command."""
        return (
            self.__class__.__name__
            + " {"
            + "name:{},defined_bounds:{},value:{},defined_units:{},description:{}".format(
                self.name, self.defined_bounds, self.value, self.defined_units, self.description
            )
            + "}"
        )

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return (
                self.name == other.name
                and self.defined_bounds == other.defined_bounds
                and self.value == other.value
                and self.defined_units == other.defined_units
                and self.description == other.description
            )
        return False

    def check_bounds(self, defined_bounds):
        """Method (*internal*) to check bounds syntax and value(s)."""
        proper_syntax = True
        if len(defined_bounds) == 2:
            try:
                lower_bound = float(defined_bounds[0])
                upper_bound = float(defined_bounds[1])
                if lower_bound >= upper_bound:
                    raise AssertionError
            except AssertionError:
                raise AssertionError(
                    "bad definition of bounds, should be defined_bounds[0]<defined_bounds[1]"
                )
            except Exception:
                raise TypeError("defined_bounds should be a [1x2] list of floats or integers")
        elif len(defined_bounds) == 1:
            try:
                float(defined_bounds[0])
            except Exception:
                raise TypeError("value should be a [1x1] list of float or integer")
        else:
            raise IndexError(
                "value/defined_bounds should be a [1x1] or [1x2] list of float(s)/integer(s)"
            )
        return proper_syntax

    def check_units(self, defined_units):
        """Method (*internal*) to check units value."""
        proper_syntax = True
        try:
            ureg = self.ureg
            ureg.default_system = "mks"
            Q_ = ureg.Quantity
            formatted_units = Q_("0" + defined_units)
            dimensionality = str(formatted_units.dimensionality)
            formatted_units = str(formatted_units.units)
        except Exception:
            raise ValueError(
                "bad units, type dir(pint.UnitRegistry().sys.system) with system in "
                "['US', 'cgs', 'imperial', 'mks'] for detailed list of units"
            )
        return proper_syntax, formatted_units, dimensionality


# -------[Positive Parameter Class Definition]----------------------------------
class PositiveParameter(Parameter):
    """Subclass of the class Parameter.

    Note
    ----
    This class has identical methods and parameters as the Parameter class
    except for internal check_bounds method, therefore Parameter should be
    defined with strictly positive bounds: 0<defined_bounds[0]<defined_bounds[1]

    For more details see :func:`~sizinglab.core.definition.Parameter`

    """

    def check_bounds(self, defined_bounds):
        """Method (*internal*) to check bounds syntax and value(s)."""
        proper_syntax = True
        if len(defined_bounds) == 2:
            try:
                lower_bound = float(defined_bounds[0])
                upper_bound = float(defined_bounds[1])
                if (lower_bound >= upper_bound) or (lower_bound <= 0):
                    raise AssertionError
            except AssertionError:
                raise AssertionError(
                    "bad definition of bounds, should be 0<defined_bounds[0]<defined_bounds[1]"
                )
            except Exception:
                raise TypeError("defined_bounds should be a [1x2] list of floats or integers")
        elif len(defined_bounds) == 1:
            try:
                if float(defined_bounds[0]) < 0:
                    raise AssertionError("bad definition of bounds, should be 0<defined_bounds")
            except Exception:
                raise TypeError("value should be a [1x1] list of float or integer")
        else:
            raise IndexError(
                "defined_bounds should be a [1x2] list of floats/integers such that 0<defined_bounds[0]<"
                "defined_bounds[1]"
            )
        return proper_syntax


# -------[ParameterSet Class Definition]----------------------------------------
class ParameterSet:
    # noinspection PyUnresolvedReferences
    """Class defining a set of different Parameter(s).

    Attributes
    ----------
    dictionary: OrderedDict of Parameter
                The Parameter are registered in oredered dictionary at key [Parameter.name]

    Example
    -------
    save parameters m and k:
        >>> In [1]: m = PositiveParameter('m', [50, 100], 'g', 'mass')
        >>> In [2]: K1 = Parameter('K1', [2], 'g', 'oversizing coefficient')
        >>> In [3]: parameter_set = ParameterSet(m, K1)

    add a parameter afterwards:
        >>> In [4]: K2 = Parameter('K2', [1.5], '', 'oversizing coefficient')
        >>> In [5]: parameter_set['K2'] = K2

    get K1 parameter value:
        >>> In [6]: a = float(parameter_set['K1'])

    change parameters order:
        >>> In [7]: parameter_set.first('K2', 'K1')
        >>> In [8]: print(parameter_set)
        K2: K2=1.5, oversizing coefficient
        K1: K1=2gram, oversizing coefficient
        m: m in [50.0,100.0]gram, mass

    delete K2 parameter:
        >>> In [9]: del parameter_set['K2']

    Note
    ----
    While using print function, display differs between variable and constraint.

    """

    def __init__(self, *parameters_list):
        """Method to create initial parameter set object using syntax expressed in example."""
        # Convert single parameter into tuple
        if isinstance(parameters_list, Parameter):
            parameters_list = parameters_list
        # Check input type
        if not (isinstance(parameters_list, tuple)):
            raise TypeError("parameter list should be a single Parameter or tuple")
        # Check parameters list
        proper_syntax = self.check_parameters(parameters_list)
        # Save parameters in dictionary
        if proper_syntax:
            object.__setattr__(self, "dictionary", OrderedDict())
            for i in range(len(parameters_list)):
                self.dictionary[parameters_list[i].name] = parameters_list[i]

    def __getitem__(self, index):
        """Method to return a parameter from a parameter set using its name as key:
        **parameter=parameter_set[parameter.name]**.

        """
        if index in self.dictionary.keys():
            return self.dictionary[index]

    def __setitem__(self, key, value):
        """Method to replace parameter in a parameter set or expend dictionary if new key."""
        if isinstance(key, str) and (isinstance(value, Parameter)):
            if key == value.name:
                self.dictionary[key] = value
            else:
                raise KeyError("the key mismatch parameter name")
        elif not (isinstance(key, str)):
            raise KeyError("key should be a string")
        elif not (isinstance(Parameter, str)):
            raise TypeError("assigned type should be a Parameter")

    def __delitem__(self, key):
        """Method to delete a parameter in a parameter set: **del parameter_set[parameter.name]**."""
        if key in self.dictionary.keys():
            if len(self.dictionary.keys()) == 1:
                self.dictionary = OrderedDict()
                raise Warning("empty dictionary")
            else:
                del self.dictionary[key]
        else:
            raise KeyError("the key is not in dictionary")

    def __str__(self):
        """Method used to print parameters in the set with function: **print(parameter_set)**."""
        if len(self.dictionary.keys()) == 0:
            statement = "Current set is empty"
        else:
            statement = ""
            for key in self.dictionary.keys():
                statement += key + ": " + str(self.dictionary[key].name)
                if len(self.dictionary[key].value) == 0:
                    statement += " in [" + str(self.dictionary[key].defined_bounds[0])
                    statement += "," + str(self.dictionary[key].defined_bounds[1]) + "]"
                else:
                    statement += "=" + str(self.dictionary[key].value[0])
                if not (self.dictionary[key].defined_units == "dimensionless"):
                    statement += self.dictionary[key].defined_units
                statement += ", " + self.dictionary[key].description
                statement += "\n"
        return statement

    def __eq__(self, other):
        if isinstance(other, ParameterSet):
            return self.dictionary == other.dictionary
        return False

    def latex_render(self, textArea=False):
        """Method used to print parameters in latex form: **latex_render(parameter_set)**
        When parameter name is of the form name_indice this will lead to $name_{indice}$ latex form, number is
        automatically rendered as indice.
        Greek letters will also be escaped automatically lambda_wind will lead to $\lambda_{wind}$.

        """
        logging.captureWarnings(True)
        print("Defined set is:")
        for key in self.dictionary.keys():
            key_str = str(key)
            key_str = key_str.lower()
            previous_char_is_int = False
            # Transform idx adding '_' pi1 -> pi_1
            # noinspection PyShadowingNames
            for idx in range(len(key_str)):
                char = key_str[idx]
                try:
                    int(char)
                    if not previous_char_is_int and idx != 0:
                        key_str = key_str[0:idx] + "_" + key_str[idx : len(key_str)]
                        previous_char_is_int = True
                except Exception as ex:
                    logg_exception(ex)
            # Adapt pi_1 to recognize greek expression
            if key_str.find("_") != -1 and key_str.find("_") == key_str.rfind("_"):
                key_list = key_str.split("_")
                if key_list[0] in greek_list:
                    key_list[0] = "\\" + key_list[0]
                if key_list[1] in greek_list:
                    key_list[1] = "\\" + key_list[1]
                key_str = key_list[0] + "_{" + key_list[1] + "}"
            # Adapt parameter name (if no indice used) to recognize greek expression
            if key_str in greek_list:
                key_str = "\\" + key_str
            # noinspection PyProtectedMember
            dimension = (
                ""
                if self.dictionary[key].defined_units == "dimensionless"
                else self.dictionary[key]._SI_units
            )
            if len(self.dictionary[key].defined_bounds) == 0:
                # noinspection PyProtectedMember
                expression = key_str + " = {:.2E}{}, ".format(
                    self.dictionary[key]._SI_bounds[0], dimension
                )
            else:
                # noinspection PyProtectedMember
                expression = key_str + " \in [{:.2E},{:.2E}]{}, ".format(
                    self.dictionary[key]._SI_bounds[0],
                    self.dictionary[key]._SI_bounds[1],
                    dimension,
                )
            if "*" in self.dictionary[key].description:
                try:
                    expression += (
                        key_str + "=" + sympy.latex(sympy.sympify(self.dictionary[key].description))
                    )
                except Exception as ex:
                    logg_exception(ex)
                    description = self.dictionary[key].description
                    description = description.replace(" ", "\\,")
                    expression += description
            else:
                description = self.dictionary[key].description
                description = description.replace(" ", "\\,")
                expression += description
            # added to render in TextArea
            if textArea:
                return Math(expression)
            else:
                display(Math(expression))
        logging.captureWarnings(False)
        print("")

    def __getstate__(self):
        """Method to save parameter set using pickle."""
        return self.dictionary

    # noinspection PyShadowingBuiltins
    def __setstate__(self, dict):
        """Method to read parameter set using picklerLoad()."""
        self.dictionary = dict

    def check_parameters(self, parameters_list: tuple):
        """Method (*internal*) to check parameters."""
        proper_syntax = True
        for i in range(len(parameters_list)):
            if not (isinstance(parameters_list[i], Parameter)):
                raise TypeError("all the parameters should be of Parameter type")
        return proper_syntax

    def first(self, *parameters_list):
        """Run through parameters_list tuple to move dictionary key to its position in the list."""
        # Check input type
        if not (isinstance(parameters_list, tuple)):
            raise TypeError("parameters_list should be a tuple of strings")
        if len(parameters_list) == 1 and isinstance(parameters_list[0], tuple):
            parameters_list = parameters_list[0]
        for parameter_name in parameters_list:
            if not (isinstance(parameter_name, str)):
                raise TypeError("parameters_list should be a tuple of strings")
        # Check for accessible parameters (stored)
        proper_syntax = True
        for parameter_name in parameters_list:
            if not (parameter_name in self.dictionary.keys()):
                raise KeyError("parameter {} not in parameter set".format(parameter_name))
        if proper_syntax:
            # Delete the parameters to be saved first and save it back to the end of the list
            used_keys = []
            for parameter_name in parameters_list:
                temp = self.dictionary[parameter_name]
                del self.dictionary[parameter_name]
                self.dictionary[parameter_name] = temp
                used_keys.append(parameter_name)
            # Write all the remaining parameters at the end of the dictionary
            for key in list(self.dictionary.keys()):
                if not (key in used_keys):
                    temp = self.dictionary[key]
                    del self.dictionary[key]
                    self.dictionary[key] = temp


# -------[PositiveParameterSet Class Definition]--------------------------------
class PositiveParameterSet(ParameterSet):
    """Subclass of the class Parameter.

    Note
    ----
    This class has identical methods and parameters as the Parameter class
    except for internal check_parameters method, therefore parameters_list should
    be a tuple of PositiveParameter or a single PositiveParameter

    For more details see :func:`~sizinglab.core.definition.ParameterSet`

    """

    def __setitem__(self, key, value):
        """Method to replace parameter in a parameter set or expend dictionary if new key."""
        if isinstance(key, str) and isinstance(value, PositiveParameter):
            if key == value.name:
                self.dictionary[key] = value
            else:
                raise KeyError("the key mismatch parameter name")
        elif not (isinstance(key, str)):
            raise KeyError("key should be a string")
        elif not (isinstance(PositiveParameter, str)):
            raise TypeError("assigned type should be a PositiveParameter")

    def check_parameters(self, parameters_list: tuple):
        """Method (*internal*) to check parameters."""
        proper_syntax = True
        for i in range(len(parameters_list)):
            if not (isinstance(parameters_list[i], PositiveParameter)):
                raise TypeError("all the parameters should be of PositiveParameter type")
        return proper_syntax


# -------[Constraint Class Definition]------------------------------------------
class Constraint(object):
    """Class defining a Constraint.

    Attributes
    ----------
    description: str
                 additional text to describe equation

    parameters: list(str)
                list of all the parameters names

    function: func
              is the computational expression of the constraint

    function_expr: str
                   is the literal expression of the constraint

    Example
    -------

    Note
    ----

    """

    def __init__(self, expression, desc=""):
        """Method to create initial Constraint."""
        # Forbidden character and parameters names
        forbidden_char = [
            "!",
            "$",
            "£",
            "%",
            "^",
            "#",
            "&",
            "?",
            ";",
            "ù",
            "é",
            "@",
            "¤",
            "µ",
            "è",
            "°",
            "\\",
        ]
        forbidden_param = ["I", "gamma", "beta", "re", "ln", "sqrt", "arg"]
        # Assign default values
        self.description = desc
        self.parameters = []
        self.function = None
        self.function_expr = None
        self._isAlgebraic = True
        # Adapt expression to the form expression>=0
        if not ("<=" in expression) and not (">=" in expression):
            eps = sys.float_info.min
            if "<" in expression:
                expression = expression.replace("<", "<=") + "-{}".format(eps)
            elif ">" in expression:
                expression = expression.replace(">", ">=") + "+{}".format(eps)
            else:
                raise SyntaxError("constraint expression should include inequality character.")
        if "<=" in expression:
            left = expression.split("<=")[0]
            right = expression.split("<=")[1]
            if right == "0":
                expression = "-(" + left + ")>=0"
            else:
                expression = right + "-(" + left + ")>=0"
        else:
            left = expression.split(">=")[0]
            right = expression.split(">=")[1]
            if right != "0":
                expression = left + "-(" + right + ")"
        expression = expression.strip()
        # Search for parameters
        sp_expr = sympy.sympify(expression.split(">=")[0])
        parameters = []
        if not (isinstance(sp_expr, tuple)):
            sp_expr = [sp_expr]
        for i in range(len(sp_expr)):
            symbols = sp_expr[i].free_symbols
            for symb in symbols:
                symb = str(symb)
                try:
                    if not isfunction(symb):
                        parameters.append(symb)
                    else:
                        self._isAlgebraic = False
                except Exception as ex:
                    logg_exception(ex)
                    parameters.append(symb)
        # Delete multiple data save
        if len(parameters) > 1:
            parameters = list(set(parameters))
        # Test expression
        test_expression = expression
        index = numpy.argsort(-1 * numpy.array([len(parameter) for parameter in parameters]))
        parameters = numpy.array(parameters)
        parameters = parameters[index].tolist()
        for value in parameters:
            test_expression = test_expression.replace(value, "1")
        try:
            # Check parameters names does not contain forbidden character
            for parameter in parameters:
                for char in forbidden_char:
                    if char in parameter:
                        raise ValueError("parameter names should not contain special character.")
            # Check parameters names are not forbidden names
            for parameter in parameters:
                if parameter in forbidden_param:
                    raise ValueError(
                        "parameter names should not be recognized by sympy as constant or function: {}.".format(
                            forbidden_param
                        )
                    )
            # Save parameters and expression
            self.parameters = parameters
            self.function_expr = expression
            # Define function
            expression = expression.split(">=")[0]
            s = ""
            for parameter in parameters:
                if s != "":
                    s = s + " , " + parameter
                else:
                    s = s + parameter
            self.function = eval("lambda " + s + ": (" + str(expression) + ")")
            print("step3 passed")
        except TypeError:
            raise SyntaxError("expression syntax is incorrect.")
        except Exception:
            raise SyntaxError(
                "expression error type not handled, check that none of the parameter are in forbidden set:{}.".format(
                    forbidden_param
                )
            )

    def compute(self, parameters_dict):

        parameters_values = []
        for parameter in self.parameters:
            parameters_values.append(parameters_dict[parameter])
        result = self.function(*parameters_values)
        return result

    def __str__(self):

        s = "\nConstraint : {}\n".format(self.description)
        s += "Contains {} parameter(s)\n".format(len(self.parameters))
        s += "--------------------------------------------\n"
        s += "Parameters are: ("
        for parameter in self.parameters:
            s += parameter + ","
        s = s[0 : len(s) - 1] + ")\n"
        s += "--------------------------------------------\n"
        if self._isAlgebraic:
            s += "Algebraic expression: \n \t" + self.function_expr + "\n"
        else:
            s += "Embedded function: \n \t" + self.function_expr + "\n"
        return s


# -------[ConstraintSet Class Definition]---------------------------------------
class ConstraintSet(object):
    """Class defining a ConstraintSet.

    Attributes
    ----------
    parameters: list(str)
                list of all the parameters names from all constraints

    constraints_list: list(Constraint)
                      is the list of Constraint

    Example
    -------

    Note
    ----

    """

    def __init__(self, *constraints):
        """Method to create initial ConstraintSet."""
        self.parameters = []
        self.constraints_list = []
        # save constraints and parameters
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                if not (constraint in self.constraints_list):
                    self.parameters.extend(constraint.parameters)
                    self.constraints_list.append(constraint)
                else:
                    warnings.warn(
                        "trying to save the same constraint multiple times: duplicate ignored."
                    )
            else:
                warnings.warn("some of the entry are not Constraint type and will be ignored.")
        # Delete multiple data save
        if len(self.parameters) > 1:
            self.parameters = list(set(self.parameters))

    def declare_doe_constraint(self, parameter_set):
        """Specific method to generate constraint function for pyvplm pixdoe use."""
        if isinstance(parameter_set, ParameterSet):
            # check that all used parameters from constraints are declared
            for parameter in self.parameters:
                # noinspection PyUnresolvedReferences
                if not (parameter in list(ParameterSet.dictionary.keys())):
                    return []

            # write function

            def f(X):
                # check X type
                if not isinstance(X, numpy.ndarray):
                    return []
                # check X dimension
                if numpy.shape(X)[1] != len(list(parameter_set.dictionary.keys())):
                    return []
                # define function using constraints
                Y = []
                # Get pi equations
                for constraint in self.constraints_list:
                    expression = constraint.function_expr
                    # noinspection PyShadowingNames
                    idx = 0
                    # noinspection PyShadowingNames
                    for parameter in parameter_set.dictionary.keys():
                        idx += 1
                        expression = expression.replace(parameter, "X[:,{}]".format(idx))
                    Y_local = numpy.array(exec(expression)).dtype(bool)
                    if len(Y) == 0:
                        Y = Y_local
                    else:
                        Y = Y * Y_local
                return Y

            return f

    def __str__(self):

        s = "\nConstraint Set : \n"
        s += "Contains {} constraint(s)\n".format(len(self.constraints_list))
        s += "--------------------------------------------\n"
        return s

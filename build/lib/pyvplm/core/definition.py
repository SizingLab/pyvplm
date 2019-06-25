# -*- coding: utf-8 -*-
"""
Core module defining elementary class and methods for SizingLab
"""

#-------[Import necessary packages]--------------------------------------------
from __future__ import absolute_import
import pint
import warnings
import logging
import sympy
from collections import OrderedDict
from IPython.display import display, Math

#-------[Parameter Class Definition]-------------------------------------------
class Parameter:
    """Class defining one physical parameter.
    
    Attributes
    ----------
    name: str
          the parameter name as a convention will be converted to upper char(s) for constant and lower char(s) for variable
        
    defined_bounds: [1*2] list of int or float 
                    converted to floats and checked that defined_bounds[0]<defined_bounds[1], set to [] for constant
    
    value: int or float 
           converted to float, set to [] for variable
    
    defined_units: str
                   the parameters defined units (expression checked using PINT package)
    
    description: str
                 some description on the parameter
    
    _SI_bounds (private): [1*2] list of float
                          the parameter bounds expressed into SI units automatically derived from  defined bounds using PINT package
    
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
    def __init__(self, name:str, defined_bounds:list, defined_units:str, description:str):
        """Method to create initial parameter object using syntax expressed in example.
        
        """
        # Check input types
        if not(isinstance(name, str) and isinstance(defined_bounds, list) and isinstance(defined_units, str)\
            and isinstance(description, str)):
            raise TypeError('attributes type mismatch class definition')
        # Check units syntax
        proper_syntax, formated_units, dimensionality = self.check_units(defined_units)
        # Check bounds syntax and values
        proper_syntax = (proper_syntax and self.check_bounds(defined_bounds))
        # Save parameter attributes' values
        if proper_syntax:
            if len(defined_bounds) == 2:
                self.name = name.lower()
            else:
                self.name = name.upper()
            if len(defined_bounds)==2:
                lower_bound = float(defined_bounds[0])
                upper_bound = float(defined_bounds[1])
                object.__setattr__(self, 'defined_bounds', [lower_bound, upper_bound])
                self.value = []
            else:
                object.__setattr__(self, 'defined_bounds', [])
                self.value = defined_bounds
            object.__setattr__(self, 'defined_units', formated_units)
            self.description = description
            ureg = pint.UnitRegistry()
            ureg.default_system = 'mks'
            Q_ = ureg.Quantity
            if len(defined_bounds) == 2:
                SI_lower_bound = Q_(lower_bound, formated_units).to_base_units()
                SI_upper_bound = Q_(upper_bound, formated_units).to_base_units()
                object.__setattr__(self, '_SI_bounds', [SI_lower_bound.magnitude, SI_upper_bound.magnitude])
            else:
                SI_lower_bound = Q_(defined_bounds[0], formated_units).to_base_units()
                object.__setattr__(self, '_SI_bounds',[SI_lower_bound.magnitude, SI_lower_bound.magnitude])
            object.__setattr__(self, '_SI_units', str(SI_lower_bound.units))
            object.__setattr__(self, '_dimensionality', dimensionality)
    
    def __getattribute__(self, attribute_name):
        """Method to access parameter attribute value (for private attributes, warning is displayed).
            Access is granted using command: **parameter_name.attribute_name**.
        
        """
        if attribute_name in ['_SI_units', '_SI_bounds', '_dimensionality']:
            warnings.warn('accessing private attribute value')
        return super(Parameter, self).__getattribute__(attribute_name)
                
    def __setattr__(self, attribute_name, value):
        """Method to write parameter attribute value, **parameter_name.attribute_name=value** (private attribute writing access denied).
        
        """         
        if attribute_name is "defined_units":
            # Check units syntax
            proper_syntax, formated_units, dimensionality = self.check_units(value)
            if proper_syntax:
                ureg = pint.UnitRegistry()
                ureg.default_system = 'mks'
                Q_ = ureg.Quantity
                object.__setattr__(self, '_dimensionality', dimensionality)
                object.__setattr__(self, 'defined_units', formated_units)
                if len(self.value) == 0:
                    SI_lower_bound = Q_(self.defined_bounds[0], formated_units).to_base_units()
                    SI_upper_bound = Q_(self.defined_bounds[1], formated_units).to_base_units()
                else:
                    SI_lower_bound = Q_(self.value, formated_units).to_base_units()
                    SI_upper_bound = SI_lower_bound
                object.__setattr__(self, '_SI_bounds', [SI_lower_bound.magnitude, SI_upper_bound.magnitude])
        elif attribute_name is "defined_bounds":
            defined_bounds = value
            # Check bounds syntax and values
            proper_syntax = self.check_bounds(defined_bounds)
            # Get units
            formated_units = self.defined_units
            # Save parameter bounds values
            if proper_syntax:
                if len(defined_bounds)==2:
                    object.__setattr__(self, 'name', self.name.lower())
                    lower_bound = float(defined_bounds[0])
                    upper_bound = float(defined_bounds[1])
                    object.__setattr__(self, 'defined_bounds', [lower_bound, upper_bound])
                    object.__setattr__(self, 'value', [])
                else:
                    object.__setattr__(self, 'name', self.name.upper())
                    object.__setattr__(self, 'defined_bounds', [])
                    object.__setattr__(self, 'value', defined_bounds)
                ureg = pint.UnitRegistry()
                ureg.default_system = 'mks'
                formated_units
                Q_ = ureg.Quantity
                if len(defined_bounds) == 2:
                    SI_lower_bound = Q_(lower_bound, formated_units).to_base_units()
                    SI_upper_bound = Q_(upper_bound, formated_units).to_base_units()
                    object.__setattr__(self, '_SI_bounds', [SI_lower_bound.magnitude, SI_upper_bound.magnitude])
                else:
                    SI_lower_bound = Q_(defined_bounds[0], formated_units).to_base_units()
                    object.__setattr__(self, '_SI_bounds', [SI_lower_bound.magnitude, SI_lower_bound.magnitude]) 
        elif attribute_name in ['name', 'value', 'description', '_SI_units', '_SI_bounds', '_dimensionality']:
            object.__setattr__(self, attribute_name, value)
        else:
            raise AttributeError('inexistent attribute or private (write access denied)')
    
    def __float__(self):
        """Method to return parameter value with syntax **float(parameter_name)**.
            If value is empty (i.e. parameter is a variable), returns NaN.
        
        """  
        if len(self.value) == 0: 
            return float('nan')
        else:
            return float(self.value[0])

    def __str__(self):
        """Method used to print parameter, called with **print(parameter_name)** function.
        
        """
        statement = ''
        for key in self.__dict__.keys():
            if key[0:1] == '_':
                statement += self.name + '.' + key + '(private) = ' + str(self.__dict__[key]) + '\n'
            else:
                statement += self.name + '.' + key + ' = ' + str(self.__dict__[key]) + '\n'
        return statement
    
    def __repr__(self):
        """Method to represent parameter definition when entering only parameter_name command.
        
        """
        return self.__class__.__name__ + ' {' + 'name:{},defined_bounds:{},value:{},defined_units:{},description:{}'\
                .format(self.name, self.defined_bounds, self.value, self.defined_units, self.description) + '}'
            
    def check_bounds(self, defined_bounds):
        """Method (*internal*) to check bounds syntax and value(s).
        
        """
        proper_syntax = True
        if len(defined_bounds)==2:
            try:
                lower_bound = float(defined_bounds[0])
                upper_bound = float(defined_bounds[1])
                if lower_bound >= upper_bound:
                    proper_syntax = False
                    raise AssertionError
            except AssertionError:
                raise AssertionError('bad definition of bounds, should be defined_bounds[0]<defined_bounds[1]')
            except:
                proper_syntax = False
                raise TypeError('defined_bounds should be a [1x2] list of floats or integers')
        elif len(defined_bounds) == 1:
            try:
                float(defined_bounds[0])
            except:
                proper_syntax = False
                raise TypeError('value should be a [1x1] list of float or integer')
        else:
            proper_syntax = False
            raise IndexError('value/defined_bounds should be a [1x1] or [1x2] list of float(s)/integer(s)')
        return proper_syntax
    
    def check_units(self, defined_units):
        """Method (*internal*) to check units value.
        
        """
        proper_syntax = True
        try:
            ureg = pint.UnitRegistry()
            ureg.default_system = 'mks'
            Q_ = ureg.Quantity
            formated_units = Q_('0' + defined_units)
            dimensionality = str(formated_units.dimensionality)
            formated_units = str(formated_units.units)
        except:
            proper_syntax = False
            raise ValueError('bad units, type dir(pint.UnitRegistry().sys.system) with system in '\
                             '[\'US\', \'cgs\', \'imperial\', \'mks\'] for detailed list of units')
        return proper_syntax, formated_units, dimensionality
     
#-------[Positive Parameter Class Definition]----------------------------------
class PositiveParameter(Parameter):
    """Sub-class of the class Parameter.
    
    Note
    ----
    This class has identical methods and parameters as the Parameter class
    except for internal check_bounds method, therefore Parameter should be 
    defined with strictly positive bounds: 0<defined_bounds[0]<defined_bounds[1]
    
    For more details see :func:`~sizinglab.core.definition.Parameter`
    
    """
    def check_bounds(self, defined_bounds):
        """Method (*internal*) to check bounds syntax and value(s). 
        
        """
        proper_syntax = True
        if len(defined_bounds) == 2:
            try:
                lower_bound = float(defined_bounds[0])
                upper_bound = float(defined_bounds[1])
                if (lower_bound >= upper_bound) or (lower_bound <= 0):
                    proper_syntax = False
                    raise AssertionError
            except AssertionError:
                raise AssertionError('bad definition of bounds, should be 0<defined_bounds[0]<defined_bounds[1]')
            except:
                proper_syntax = False
                raise TypeError('defined_bounds should be a [1x2] list of floats or integers')
        elif len(defined_bounds) == 1:
            try:
                if float(defined_bounds[0])< 0:
                    raise AssertionError('bad definition of bounds, should be 0<defined_bounds')
            except:
                proper_syntax = False
                raise TypeError('value should be a [1x1] list of float or integer')
        else:
            proper_syntax = False
            raise IndexError('defined_bounds should be a [1x2] list of floats/integers such that 0<defined_bounds[0]<defined_bounds[1]')
        return proper_syntax

#-------[ParameterSet Class Definition]----------------------------------------                    
class ParameterSet:
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
        """Method to create initial parameter set object using syntax expressed in example.
        
        """
        # Convert single parameter into tuple
        if isinstance(parameters_list, Parameter):
            parameters_list = (parameters_list)
        # Check input type
        if not(isinstance(parameters_list, tuple)):
            raise TypeError('parameter list should be a single Parameter or tuple')
        # Check parameters list
        proper_syntax = self.check_parameters(parameters_list)
        # Save parameters in dictionary
        if proper_syntax:
            object.__setattr__(self, 'dictionary', OrderedDict())
            for i in range(len(parameters_list)):
                self.dictionary[parameters_list[i].name] = parameters_list[i]
    
    def __getitem__(self, index):
        """Method to return a parameter from a parameter set using its name as key: **parameter=parameter_set[parameter.name]**.
        
        """
        if index in self.dictionary.keys():
            return self.dictionary[index]
        
    def __setitem__(self, key, value):
        """Method to replace parameter in a parameter set or expend dictionary if new key.
        
        """
        if isinstance(key, str) and (isinstance(value, Parameter)):
            if key == value.name:
                self.dictionary[key] = value
            else:
                raise KeyError('the key mismatch parameter name')
        elif not(isinstance(key, str)):
            raise KeyError('key should be a string')
        elif not(isinstance(Parameter, str)):
            raise TypeError('assigned type should be a Parameter')
    
    def __delitem__(self, key):
        """Method to delete a parameter in a parameter set: **del parameter_set[parameter.name]**.
        
        """        
        if key in self.dictionary.keys():
            if len(self.dictionary.keys()) == 1:
                self.dictionary = OrderedDict()
                raise Warning('empty dictionary')
            else:
                del self.dictionary[key]
        else:
            raise KeyError('the key is not in dictionary')
    
    def __str__(self):
        """Method used to print parameters in the set with funciton: **print(parameter_set)**.
        
        """        
        if len(self.dictionary.keys())==0:
            statement = 'Current set is empty'
        else:
            statement = ''
            for key in self.dictionary.keys():
                statement += key + ': ' + str(self.dictionary[key].name)
                if len(self.dictionary[key].value)==0:
                    statement += ' in [' + str(self.dictionary[key].defined_bounds[0])
                    statement += ',' + str(self.dictionary[key].defined_bounds[1]) + ']'
                else:
                    statement += '=' + str(self.dictionary[key].value[0])
                if not(self.dictionary[key].defined_units == 'dimensionless'):
                    statement += self.dictionary[key].defined_units
                statement += ', ' + self.dictionary[key].description
                statement += '\n'
        return statement
    
    def latex_render(self):
        """Method used to print parameters in latex form: **latex_render(parameter_set)**
            When parameter name is of the form name_indice this will lead to $name_{indice}$ latex form, number is automatically rendered as indice.
            Greek letters will also be escaped automatically lambda_wind will lead to $\lambda_{wind}$.
        
        """
        logging.captureWarnings(True)
        greek_list = ['alpha','beta','gamma','delta','epsilon','varepsilon','zeta','eta','theta','vartheta','gamma','kappa','lambda',\
                      'mu','nu','xi','pi','varpi','rho','varrho','sigma','varsigma','tau','upsilon','phi','varphi','chi','psi','omega']
        print('Defined set is:')
        for key in self.dictionary.keys():
            key_str = str(key)
            key_str = key_str.lower()
            previous_char_is_int = False
            # Transform idx adding '_' pi1 -> pi_1
            for idx in range(len(key_str)):
                char = key_str[idx]
                try:
                    int(char)
                    if not(previous_char_is_int) and idx !=0:
                        key_str = key_str[0:idx] + '_' + key_str[idx:len(key_str)]
                        previous_char_is_int = True
                except:
                    pass
            # Adapt pi_1 to recognize greek expression
            if key_str.find('_') != -1 and key_str.find('_') == key_str.rfind('_'):
                key_list = key_str.split('_')
                if key_list[0] in greek_list:
                    key_list[0] = '\\' + key_list[0]
                if key_list[1] in greek_list:
                    key_list[1] = '\\' + key_list[1]
                key_str = key_list[0] + '_{' + key_list[1] + '}'
            # Adapt parameter name (if no indice used) to recognize greek expression
            if key_str in greek_list:
                    key_str = '\\' + key_str
            dimension = '' if self.dictionary[key].defined_units == 'dimensionless' else self.dictionary[key]._SI_units
            if len(self.dictionary[key].defined_bounds) == 0:
                expression = key_str + ' = {:.2E}{}, '.format(self.dictionary[key]._SI_bounds[0], dimension)
            else:
                expression = key_str + ' \in [{:.2E},{:.2E}]{}, '.format(self.dictionary[key]._SI_bounds[0], self.dictionary[key]._SI_bounds[1], dimension)
            if '*' in self.dictionary[key].description:
                try:
                    expression += key_str + '=' + sympy.latex(sympy.sympify(self.dictionary[key].description))
                except:
                    description = self.dictionary[key].description
                    description = description.replace(' ','\\,')
                    expression += description
            else:
                description = self.dictionary[key].description
                description = description.replace(' ','\\,')
                expression += description
            display(Math(expression))
        logging.captureWarnings(False)
        print('')
    
    def __getstate__(self):
        """Method to save parameter set using pickle.
        
        """        
        return self.dictionary
    
    def __setstate__(self,dict):
        """Method to read parameter set using picklerLload().
        
        """        
        self.dictionary = dict
            
    def check_parameters(self, parameters_list:tuple):
        """Method (*internal*) to check parameters.
        
        """
        proper_syntax = True
        for i in range(len(parameters_list)):
            if not(isinstance(parameters_list[i], Parameter)):
                proper_syntax = False
                raise TypeError('all the parameters should be of Parameter type')
                break
        return proper_syntax
    
    def first(self, *parameters_list):
        """Run trough parameters_list tuple order to move dictionary key to its position in the list.
        
        """
        # Check input type
        if not(isinstance(parameters_list, tuple)):
            raise TypeError('parameters_list should be a tuple of strings')
        if len(parameters_list) == 1 and isinstance(parameters_list[0], tuple):
            parameters_list = parameters_list[0]
        for parameter_name in parameters_list:
            if not(isinstance(parameter_name, str)):
                raise TypeError('parameters_list should be a tuple of strings')
        # Check for accessible parameters (stored)
        proper_syntax = True
        for parameter_name in parameters_list:
            if not(parameter_name in self.dictionary.keys()):
                proper_syntax = False
                raise KeyError('parameter {} not in parameter set'.format(parameter_name))
                break
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
                if not(key in used_keys):
                    temp = self.dictionary[key]
                    del self.dictionary[key]
                    self.dictionary[key] = temp
                    
#-------[PositiveParameterSet Class Definition]--------------------------------                    
class PositiveParameterSet(ParameterSet):
    """Sub-class of the class Parameter.
    
    Note
    ----
    This class has identical methods and parameters as the Parameter class
    except for internal check_parameters method, therefore parameters_list should 
    be a tuple of PositiveParameter or a single PositiveParameter
    
    For more details see :func:`~sizinglab.core.definition.ParameterSet`
    
    """
    def __setitem__(self, key, value):
        """Method to replace parameter in a parameter set or expend dictionary if new key.
        
        """
        if isinstance(key, str) and isinstance(value, PositiveParameter):
            if key == value.name:
                self.dictionary[key] = value
            else:
                raise KeyError('the key mismatch parameter name')
        elif not(isinstance(key, str)):
            raise KeyError('key should be a string')
        elif not(isinstance(PositiveParameter, str)):
            raise TypeError('assigned type should be a PositiveParameter')
    
    def check_parameters(self,parameters_list:tuple):
        """Method (*internal*) to check parameters.
        
        """
        proper_syntax = True
        for i in range(len(parameters_list)):
            if not(isinstance(parameters_list[i], PositiveParameter)):
                proper_syntax = False
                raise TypeError('all the parameters should be of PositiveParameter type')
                break
        return proper_syntax

##-------[CausalEquation Class Definition]------------------------------------------
#class CausalEquation(object):
#    """Class defining a CausalEquation.
#        
#        Attributes
#        ----------
#        expression: str
#                    Contains the equation expression
#
#        output: str
#                    The variable name corresponding to the output computed by the expression
#
#        inputs: list(str)
#                    The list of the input variables name used in the expression
#
#        relative_error: (float, float)
#                    The value of the relative error (u,s) where u is the mean error and s the standard deviation, can be (0.0, 0.0)
#
#        partials: dict[str] = str
#                    The partials derivatives of the output with respect to inputs
#        
#        Example
#        -------
#        save parameters m and k:
#            >>> In [1]: m = PositiveParameter('m', [50, 100], 'g', 'mass')
#            >>> In [2]: K1 = Parameter('K1', [2], 'g', 'oversizing coefficient')
#            >>> In [3]: parameter_set = ParameterSet(m, K1)
#               
#        Note
#        ----
#    
#    """
#    def __init__(self, expression, relative_error=None, partials=None):
#        """Method to create initial CausalEquation.
#        
#        """
#        self.expression = expression
#
#        output, inputs = self.analyze_expression()
#        self.output = output
#        self.inputs = inputs
#
#        if relative_error == None:
#            print("Setting relative error to 0.0 for expression '"+ self.expression + "'.")
#            self.relative_error = (0.0, 0.0)
#        else:
#            self.relative_error = relative_error
#
#        if partials == None:
#            partials = self.derive_partials_symbolically()
#            self.partials = partials
#        else: 
#            self.partials = partials
#
#    def analyze_expression(self):
#        try:
#            left = self.expression.split('=')[0]
#            right = self.expression.split('=')[1]
#
#            output = str(sp.sympify(left).free_symbols.pop())
#            inputs = []
#            sp_expr = sp.sympify(right)
#            symbols = sp_expr.free_symbols
#            for symb in symbols:
#                inputs.append(str(symb))
#        except NameError: 
#            print("The expression '" + self.expression + "' is wrongly written.")
#        return output, inputs
#
#    def derive_partials_symbolically(self):
#        try:
#            partials = {}
#            right = self.expression.split('=')[1]
#            sp_expr = sp.sympify(right)
#
#            inputs = self.inputs
#            for inp in inputs:
#                symb = sp.Symbol(inp)
#                sp_diff = sp.diff(sp_expr, symb)
#                partials[(self.output, str(symb))] = str(sp_diff)
#
#        except NameError: 
#            print("Impossible to compute partials for expression '" + self.expression + "'.")
#
#        return partials
#
#    def __str__(self):
#
#        return self.expression


#-------[Example when executed as main]----------------------------------------
if __name__ == '__main__':
    K = PositiveParameter('K', [10], 'g', 'oversizing coefficient')
    print(K)
    m = PositiveParameter('m', [50,100], 'cm', 'mass')
    print(m)
    m.defined_units='kg'
    print(K)
    print(m._SI_units)
    print(m.defined_units)
    print('')
    #a = m._SI_units
    a = float(K)
    b = float(m)
    parameter_set = PositiveParameterSet(m, K)
    b = parameter_set['m']
    print(parameter_set)
    parameter_set.first('K')
    parameter_set.first('K', 'm')
    print(parameter_set)
    del a, b

    first_expr = CausalEquation('y = exp(x_1 * x_2**3.0)/u')
    print(first_expr)
    print(first_expr.output)
    print(first_expr.inputs)
    print(first_expr.partials)

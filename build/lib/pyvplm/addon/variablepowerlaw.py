# -*- coding: utf-8 -*-
"""
Addon module fitting variable power law response surface on dimensionless parameter computed with FEM
"""

#-------[Import necessary packages]--------------------------------------------
import pint
import ast
import numpy
import logging
import pandas
import copy
import scipy
import warnings
from sklearn.preprocessing import PolynomialFeatures
from fractions import Fraction
from itertools import permutations, combinations
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication, function_exponentiation
import matplotlib
import matplotlib.pyplot as plot
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
from IPython.display import Latex, display, clear_output
from sympy.interactive import printing
from ipywidgets import widgets, VBox
from pyvplm.core.definition import PositiveParameter, PositiveParameterSet
printing.init_printing(use_latex='png')
#from threading import Thread, Lock
#lock = Lock()

#-------[Write dimensional matrix from parameter set]--------------------------                   
def write_dimensional_matrix(parameter_set):
    """Function to extract dimensional matrix from a PositiveParameterSet.
    
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        Returns
        -------
        dimensional_matrix: pandas.DataFrame 
                            Column labels refers to the parameters names and rows to the dimensions        
        
        Example
        -------
        define a positive set first:
            >>> In [1]: u = PositiveParameter('u', [1e-9,1e-6], 'm', 'Deflection')
            >>> In [2]: f = PositiveParameter('f', [150,500], 'N', 'Load applied')
            >>> In [3]: l = PositiveParameter('l', [1,3], 'm', 'Cantilever length')
            >>> In [4]: e = PositiveParameter('e', [60e9,80e9], 'Pa', 'Young Modulus')
            >>> In [5]: d = PositiveParameter('d', [10,60], 'mm', 'Diameter of cross-section')
            >>> In [6]: parameter_set = PositiveParameterSet(u, f, l, e, d)
        
        apply function:
            >>> In [7]: dimensional_matrix = write_dimensional_matrix(parameter_set)
            >>> In [8]: dimensional_matrix.values
            >>> Out[8]: [[1, 0, 0], [1, 1, -2], [1, 0, 0], [-1, 1, -2], [1, 0, 0]]
    
    """
    dimensional_matrix = []
    dimensional_set = []
    logging.captureWarnings(True)
    if isinstance(parameter_set, PositiveParameterSet):
        isnotempty = (parameter_set.dictionary and True) or False
        if isnotempty:
            #First extract all dimensions from saved parameters
            for key1 in parameter_set.dictionary.keys():
                parameter = parameter_set.dictionary[key1]
                dimensions = str(pint.util.ParserHelper.from_string(parameter._dimensionality))
                #ParserHelper return expression of the form: "1.0 {'[length]': -1.0, '[mass]': 1.0, '[time]': -2.0}"
                #Eval part between {} to create dictionary
                dimensions = ast.literal_eval(dimensions[dimensions.find('{'):len(dimensions)])
                for key2 in dimensions.keys():#Go across keys to find new units and register them
                    if not(key2 in dimensional_set):
                        dimensional_set.append(key2)
            #Then construct parameters exponent in matrix    
            for key1 in parameter_set.dictionary.keys():
                parameter = parameter_set.dictionary[key1]
                dimensions = str(pint.util.ParserHelper.from_string(parameter._dimensionality))
                dimensions = ast.literal_eval(dimensions[dimensions.find('{'):len(dimensions)])
                dimensional_vector = []
                for i in range(len(dimensional_set)):
                    if dimensional_set[i] in dimensions.keys():
                        if isinstance(dimensions[dimensional_set[i]], float):
                            dimensional_vector.append(Fraction.from_float(dimensions[dimensional_set[i]]))
                        else:
                            dimensional_vector.append(Fraction(dimensions[dimensional_set[i]]))    
                    else:
                        dimensional_vector.append(Fraction(0))
                dimensional_matrix.append(dimensional_vector)
    else:
        raise TypeError('parameter_set should be PositiveParameterSet')
    logging.captureWarnings(False)
    return pandas.DataFrame(numpy.array(dimensional_matrix,int), columns = dimensional_set, index = parameter_set.dictionary.keys())
                    
#-------[Find echelon form of a given matrix]----------------------------------                    
def compute_echelon_form(in_matrix):
    """Function that computes a matrix into its echelon form.
        
        Parameters
        ----------
        in_matrix: [m*n] numpy.array of float or int
        
        Returns
        -------
        out_matrix: [m*n] list of float or int
                    Matrix with echelon form derived from in_matrix
        
        pivot: [m*m] list of fraction
                Pivot matrix to link out_matrix to in_matrix
        
        pivot_points: [1*k] numpy.array of int
                      Index of pivot points k = rank(in_matrix)<= min(m, n)
            
        Example
        -------
        define dimensional matrix:
            >>> In [1]: in_matrix = numpy.array([[0, 1, 0], [1, 1, -2], [0, 1, 0], [1, -1, -2], [0, 1, 0]], int)
        
        perform echelon function:
            >>> In [2]: out_matrix, pivot, pivot_points = compute_echelon_form(in_matrix)
            >>> In [3]: print([[float(out_matrix[nr][nc]) for nc in range(len(out_matrix[0]))] for nr in range(len(out_matrix))])
                [[1.0, 0.0, -2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            >>> In [4]: print([[float(pivot[nr][nc]) for nc in range(len(pivot[0]))] for nr in range(len(pivot))])
                [[-1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 1.0]]
            >>> In [5]: print(pivot_points)
                [1, 0]
            
        """
    if isinstance(in_matrix, numpy.ndarray):
        # Check values
        if not(numpy.issubdtype(in_matrix.dtype, numpy.integer) or numpy.issubdtype(in_matrix.dtype, numpy.float64)):
            raise TypeError('in_matrix type in index should be integer or float.')
        # Write output matrix as list
        out_matrix = list(in_matrix)
        # Init pivot as identity matrix
        pivot = [[Fraction(1) if nr == nc else Fraction(0) for nc in range(len(out_matrix))] for nr in range(len(out_matrix))]
        out_matrix = [[Fraction(out_matrix[nr][nc]) for nc in range(len(out_matrix[0]))] for nr in range(len(out_matrix))]
        nr = len(out_matrix)
        nc = len(out_matrix[0])
        # Init pivot points, row index and lead
        pivot_points = []
        row_index = list(range(nr))
        lead = 0
        for r in range(nr):
            if lead >= nc:
                return out_matrix, pivot, pivot_points
            #Check if matrix(r,lead) is not zero otherwise swap with matrix(i,lead) i>r: other parameter
            #if not null. If no non-zero factor found switch to new lead (dimension)
            i = r
            while out_matrix[i][lead] == 0:
                i += 1
                if i != nr:
                    continue
                i = r
                lead += 1
                if nc == lead:
                    return out_matrix, pivot, pivot_points
            #Swap the parameters when matrix(r,lead)=0 and matrix(i,lead) with i>r
            if i!=r:
                out_matrix[i], out_matrix[r] = out_matrix[r], out_matrix[i]
                pivot[i], pivot[r] = pivot[r], pivot[i]
                row_index[i], row_index[r] = row_index[r], row_index[i]
            pivot_points.append(row_index[r])
            #Change parameter power to obtain power 1 on lead dimension and adapt identity matrix
            #ex: [0 3 0 1] row on matrix with lead=2 gives [0 1 0 1/3]
            lv = out_matrix[r][lead]
            out_matrix[r] = [mrx / lv for mrx in out_matrix[r]]
            pivot[r] = [mrx / lv for mrx in pivot[r]]
            #Substract from other rows the dimensionality (lead) to have zeros on lead column
            #except on i parameter
            for i in range(nr):
                if i == r:
                    continue
                lv = out_matrix[i][lead]
                out_matrix[i] = [iv - lv*rv for rv, iv in zip(out_matrix[r], out_matrix[i])]
                pivot[i] = [iv - lv*rv for rv, iv in zip(pivot[r], pivot[i])]
            #Switch to new lead once corresponding parameter found
            lead += 1
    else:
        raise TypeError('in_matrix should be numpy array')
    return out_matrix, pivot, pivot_points

#-------[Extract PI set from a given ordered parameter set]--------------------        
def buckingham_theorem(parameter_set, track=False):
    """Function that returns pi_set dimensionless parameters from a set of physical parameters.
        The Pi expression is lower integer exponent i.e. Pi1 = x1**2*x2**-1 and not x1**1*x2**-0.5 or x1**4*x2**-2
        
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        track: bool
               Activates information display (default is False)
        
        Returns
        -------
        pi_set: PositiveParameterSet
                Defines the k (k<n) dimensionless parameters of the problem 
        
        pi_list: [1*k] list of str
                 The dimensionless parameters' expression
        
        Example
        -------
        define a positive set first, see: :func:`~sizinglab.addon.variablepowerlaw.write_dimensional_matrix`
        
        orient repetitive set (using if possible d and l):
            >>> In [7]: parameter_set.first('d', 'l')
        
        search corresponding pi_set:
            >>> In [8]: pi_set = buckingham_theorem(parameter_set, True)
                Chosen repetitive set is: {d, l}
                pi1: pi1 in [1.666666666666667e-08,9.999999999999999e-05], d**-1.0*u**1.0
                pi2: pi2 in [16.666666666666668,300.0], d**-1.0*l**1.0
                pi3: pi3 in [12000.0,1920000.0000000002], d**2.0*e**1.0*f**-1.0
        
        Note
        ----
        Repetitive parameters are elected as first pivot point in the order of arrival in 
        parameter set dictionary keys, therefore user can orient repetitive set applying
        the 'first' method on the parameter set (see example)
    
    """
    pi_set = PositiveParameterSet()
    pi_list = []
    if isinstance(parameter_set, PositiveParameterSet) and isinstance(track, bool):
        #Calculate the dimension matrix
        dimensional_matrix = write_dimensional_matrix(parameter_set)
        problem_rank = numpy.linalg.matrix_rank(dimensional_matrix.values)
        #Calculate the echelon matrix
        dimensional_matrix, pivot_matrix, pivot_points = compute_echelon_form(dimensional_matrix.values)
        #Check that dimensional matrix rank is lower than dimensions number
        if len(dimensional_matrix)>problem_rank:
            #For each PI, make all parameters exponent integer and minimize the number 
            #of negative exponents (since PI**i is dimensionless).
            for r in range(len(dimensional_matrix)):
                if any(el != 0 for el in dimensional_matrix[r]):
                    continue
                max_den = max(f.denominator for f in pivot_matrix[r])
                if sum(f < 0 for f in pivot_matrix[r]) > sum(f > 0 for f in pivot_matrix[r]):
                    for i in range(len(pivot_matrix[r])):
                        pivot_matrix[r][i] = -max_den * pivot_matrix[r][i].numerator / pivot_matrix[r][i].denominator
                else:
                    for i in range(len(pivot_matrix[r])):
                        pivot_matrix[r][i] = max_den * pivot_matrix[r][i].numerator / pivot_matrix[r][i].denominator
                #Compute PI=f(parameters) expression and bounds
                expression = ''
                lower_bound = 1
                upper_bound = 1
                #Sort parameter name and corresponding exponent to have expression unicity
                parameter_list = numpy.array(list(parameter_set.dictionary.keys()))
                exponent_list = numpy.array(pivot_matrix[r])
                exponent_list = exponent_list[numpy.argsort(parameter_list)]
                parameter_list = parameter_list[numpy.argsort(parameter_list)]
                for i in range(len(exponent_list)):
                    parameter = parameter_set[str(parameter_list[i])]
                    if exponent_list[i] > 0:
                        lower_bound = lower_bound * parameter._SI_bounds[0]**exponent_list[i]
                        upper_bound = upper_bound * parameter._SI_bounds[1]**exponent_list[i]
                        expression += (parameter.name + '**' + str(exponent_list[i]) + '*')
                    elif exponent_list[i] < 0:
                        lower_bound = lower_bound * parameter._SI_bounds[1]**exponent_list[i]
                        upper_bound = upper_bound * parameter._SI_bounds[0]**exponent_list[i]
                        expression += (parameter.name + '**' + str(exponent_list[i]) + '*')                          
                if len(expression) != 0:
                    expression = expression[0:len(expression) - 1]
                bounds = [lower_bound, upper_bound]
                #Compute PI name
                pi_name = 'pi' + str(len(pi_set.dictionary.keys()) + 1)
                #Save parameter and expression list
                pi_name = pi_name.upper() if bounds[0]==bounds[1] else pi_name
                bounds = [bounds[0]] if bounds[0]==bounds[1] else bounds
                exec(pi_name+' = PositiveParameter(\''+pi_name+'\',bounds,\'\',\''+expression+'\')')
                pi_set[pi_name]=eval(pi_name)
                pi_list.append(expression)
            #Print results
            if track:
                expression = 'Choosen repetitive set is: {'
                for index in range(len(pivot_points)):
                    if index != len(pivot_points) - 1:
                        expression += parameter_list[pivot_points[index]] + ', '
                    else:
                        expression += parameter_list[pivot_points[index]] + '}'
                print(expression)
                print(pi_set)
    else:
        if not(isinstance(parameter_set, PositiveParameterSet)):
            raise TypeError('parameter_set should be PositiveParameterSet')
        else:
            raise TypeError('track should be boolean')
    return pi_set, pi_list

#-------[Extract all possible PI sets from permuted parameter sets]------------         
def automatic_buckingham(parameter_set, track=False):
    """Function that returns all possible pi_set (with lower exponent) from a set of physical parameters.
        Based on buckingham_theorem function call.
        
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        track: bool
               Activates information display (default is False)
        
        Returns
        -------
        combinatory_pi_set: dict of [1*2] tuples 
                            Stores pi_set at [0] tuple index and Pi expression (str) list at [1] tuple index
                            
        alternative_set_dict: dict of str
                              Stores the alternate expressions for widgets display
        
        Example
        -------
        define a positive set first, see: :func:`~sizinglab.addon.variablepowerlaw.write_dimensional_matrix`
        
        search corresponding pi_set:
            >>> In [7]: combinatory_pi_set = automatic_buckingham(parameter_set, True)
                [AUTO. BUCKINGHAM] Testing repetitive set 1/120: total alternative pi set size is 1
                [AUTO. BUCKINGHAM] Testing repetitive set 2/120: total alternative pi set size is 1
                [AUTO. BUCKINGHAM] Testing repetitive set 3/120: total alternative pi set size is 1
                ...
            >>> In [8]: print(combinatory_pi_set[1][0])
                pi1: pi1 in [1000000.0,2999999999.9999995], l**1.0*u**-1.0
                pi2: pi2 in [1.2e-10,0.0005333333333333334], e**1.0*f**-1.0*u**2.0
                pi3: pi3 in [10000.0,59999999.99999999], d**1.0*u**-1.0            
    
    """
    if isinstance(parameter_set, PositiveParameterSet) and isinstance(track, bool):
        # Extract parameters_list
        parameters_list = []
        for key in parameter_set.dictionary.keys():
            parameters_list.append(parameter_set.dictionary[key].name)
        # Calculate first pi_list to determine the number of repetitive variables
        _, pi_list = buckingham_theorem(parameter_set, False)
        nb_repetitive = len(parameters_list) - len(pi_list)
        # Generate combination_list
        combination_list = list(combinations(parameters_list, nb_repetitive))
        # Save first combination
        new_parameter_set = copy.deepcopy(parameter_set)
        new_parameter_set.first(combination_list[0])
        pi_set, pi_list = buckingham_theorem(new_parameter_set, False)
        saved_set = (pi_set, pi_list)
        combinatory_pi_set = {}
        combinatory_pi_set[1] = saved_set
        # For each combination order parameter_set, apply buckingham and save obtained PI set if different from saved
        for idx in range(len(combination_list)):
            combination = combination_list[idx]
            new_parameter_set.first(combination)
            new_pi_set, new_pi_list = buckingham_theorem(new_parameter_set, False)
            new_pi_list = numpy.array(new_pi_list)
            already_saved = False
            for key in combinatory_pi_set.keys():
                saved_pi_list = combinatory_pi_set[key][1]
                saved_pi_list = numpy.array(list(permutations(saved_pi_list)))
                if numpy.ndarray.sum(numpy.all(saved_pi_list == new_pi_list, axis=1)) >= 1:
                    already_saved = True
                    break
            if not(already_saved):
                saved_set = (new_pi_set, new_pi_list)
                combinatory_pi_set[max(combinatory_pi_set.keys()) + 1] = saved_set
            if track:
                print('[AUTO. BUCKINGHAM] Testing repetitive set {}/{}: total alternative pi set size is {}'.format(idx + 1,\
                      len(combination_list), len(combinatory_pi_set.keys())))
        alternative_set_dict = {}
        for key in combinatory_pi_set.keys():
            expression = ''
            for idx in range(len(combinatory_pi_set[key][1])):
                expression += 'pi' + str(idx + 1) + '=' + combinatory_pi_set[key][1][idx] + '  |  ' 
            expression = expression[0:len(expression) - 5]
            alternative_set_dict[expression] = key
        return combinatory_pi_set, alternative_set_dict
    else:
        if not(isinstance(parameter_set, PositiveParameterSet)):
            raise TypeError('parameter_set should be PositiveParameterSet.')
        else:
            raise TypeError('track should be boolean.')
            
#---[Extract all possible PI sets from permuted parameter sets - multi-thread]- 
#class buckingham_thread(Thread):
#    def __init__(self, parameter_set, combination_list, combinatory_pi_set, combinatory_pi_list_all, thread_name, track):
#        Thread.__init__(self)
#        self.parameter_set = parameter_set
#        self.combination_list = combination_list
#        self.combinatory_pi_set = combinatory_pi_set
#        self.combinatory_pi_list_all = combinatory_pi_list_all
#        self.thread_name = thread_name
#        self.track = track
#    def run(self):
#        for idx1 in range(len(self.combination_list)):
#            if self.track:
#                sys.stdout.write(self.thread_name + ': ' + str(idx1+1) + '/' + str(len(self.combination_list)) + '\n')
#                sys.stdout.flush()
#            new_parameter_set = copy.deepcopy(self.parameter_set)
#            combination = self.combination_list[idx1]
#            new_parameter_set.first(combination)
#            pi_set, pi_list = buckingham_theorem(new_parameter_set, False)
#            #with lock:
#            if len(numpy.shape(numpy.array(self.combinatory_pi_list_all))) == 2:
#                already_saved = True if numpy.ndarray.sum(numpy.all(pi_list == numpy.array(self.combinatory_pi_list_all), axis=1)) >= 1 else False
#            else:
#                already_saved = False
#            if not(already_saved):
#                if len(list(self.combinatory_pi_set.keys())) == 0:
#                    self.combinatory_pi_set[1] = (pi_set, pi_list)
#                else:
#                    self.combinatory_pi_set[max(self.combinatory_pi_set.keys()) + 1] = (pi_set, pi_list)
#                self.combinatory_pi_list_all.extend(numpy.array(list(permutations(pi_list))).tolist())
#                if self.track:
#                    sys.stdout.write('Total alternative pi set size increased to {}\n'.format(len(list(self.combinatory_pi_set.keys()))))
#                    sys.stdout.flush()
#                    sys.stdout.write('With combinatory len {}\n'.format(len(self.combinatory_pi_list_all)))
#                    sys.stdout.flush()
#        
#def automatic_buckingham_mt(parameter_set, threads_number, track=False):
#    if isinstance(parameter_set, PositiveParameterSet) and isinstance(threads_number, int) and isinstance(track, bool):
#        if threads_number <= 0:
#            raise ValueError('threads_number should be >0.')
#        combinatory_pi_set = {}
#        combinatory_pi_list_all = []
#        # Extract parameters_list
#        parameters_list = []
#        for key in parameter_set.dictionary.keys():
#            parameters_list.append(parameter_set.dictionary[key].name)
#        # Generate combination_list
#        combination_list = list(permutations(parameters_list))
#        # Repartition of combination_list over threads
#        if threads_number > len(combination_list):
#            warnings.warn('threads_number greater than available combinations, reduced to {}.'.format(len(combination_list)))
#            threads_number = min(threads_number, len(combination_list))
#        sublist_length = int((len(combination_list)/threads_number) -  (len(combination_list)/threads_number) % 1)
#        thread_combination_list = []
#        for idx in range(threads_number):
#            if idx == threads_number - 1:
#                thread_combination_list.append(combination_list)
#            else:
#                 thread_combination_list.append(combination_list[0:sublist_length])
#                 combination_list = combination_list[sublist_length:len(combination_list)]
#        # Launch threads
#        for idx in range(threads_number):
#            exec('thread_' + str(idx) + ' = buckingham_thread(parameter_set, thread_combination_list[{}], combinatory_pi_set, combinatory_pi_list_all, \'thread_{}\', track)'.format(idx,idx))
#            exec('thread_' + str(idx) + '.start()')
#        # Wait for finishing threads
#        for idx in range(threads_number):
#            exec('thread_' + str(idx) + '.join()')
#        # Write display alternate set
#        alternative_set_dict = {}
#        for key in combinatory_pi_set.keys():
#            expression = ''
#            for idx in range(len(combinatory_pi_set[key][1])):
#                expression += 'pi' + str(idx + 1) + '=' + combinatory_pi_set[key][1][idx] + '  |  ' 
#            expression = expression[0:len(expression) - 5]
#            alternative_set_dict[expression] = key
#        return combinatory_pi_set, alternative_set_dict
#    else:
#        if not(isinstance(parameter_set, PositiveParameterSet)):
#            raise TypeError('parameter_set should be PositiveParameterSet.')
#        elif not(isinstance(track, bool)):
#            raise TypeError('track should be boolean.')            
#        else:
#            raise TypeError('threads_number should be integer.')

#-------[Define manually the PI sets: global checks performed]-----------------         
def force_buckingham(parameter_set, *pi_list):
    """Function used to define manually a dimensionless set of parameters.
        Parameters availability, pi expression and dimension or even pi matrix rank are checked.
        
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        *pi_list: tuple of str
                  Defines the Pi dimensionless parameters expression of the problem
        
        Returns
        -------
        pi_set: PositiveParameterSet
                Defines the k (k<n) dimensionless parameters of the problem
        
        Example
        -------
        define a positive set first, see: :func:`~sizinglab.addon.variablepowerlaw.write_dimensional_matrix`
        
        force pi set:
            >>> In [7]: pi_set = force_buckingham(parameter_set, 'l/u', 'e/f*u^2', 'd/u')
            >>> In [8]: print(pi_set)
                pi1: pi1 in [1000000.0,2999999999.9999995], u**-1.0*l**1.0
                pi2: pi2 in [1.2e-10,0.0005333333333333334], u**2.0*f**-1.0*e**1.0
                pi3: pi3 in [10000.0,59999999.99999999], u**-1.0*d**1.0
        
        Note
        ----
        The analysis is conducted on Pi dimension, rank of Pi-parameter exponents,
        global expression and number of Pi compared to dimensional matrix rank and
        parameters number. 
        The 'understood expression' is visible by printing pi_set.
    
    """
    if isinstance(pi_list, str):
        pi_list = (pi_list)
    if isinstance(parameter_set, PositiveParameterSet) and isinstance(pi_list, tuple):
        pi_list = list(pi_list)
        for idx in range(len(pi_list)):
            if not(isinstance(pi_list[idx], str)):
                raise SyntaxError('pi(s) should be defined using tuple of string expressions.')
        transformations = standard_transformations + (function_exponentiation,) +  (implicit_multiplication,)
        for pi_number in range(len(pi_list)):
            expression = pi_list[pi_number]
            # Raise error if inappropriate operand used in pi definition
            if ('=' in expression) or ('<' in expression) or ('>' in expression) or ('+' in expression):
                raise SyntaxError('pi(s) expression contains inapropriate operand: \'=\', \'<\', \'>\', or \'+\'.')
            # Replace exponent expression
            expression = expression.replace('^','**')
            parameter_list = []
            list_change = True
            # Parse expression until no decomposition found
            while list_change:
                list_change = False
                if len(parameter_list) == 0:
                    sympy_expr = parse_expr(expression, evaluate=False, transformations=transformations)
                    for argument in list(sympy_expr.args):
                        if not(str(argument) in parameter_list):
                            list_change = True
                            parameter_list.append(str(argument))
                else:
                    indices = range(len(parameter_list))
                    for idx in indices:
                        sympy_expr = parse_expr(parameter_list[idx], evaluate=False, transformations=transformations)
                        try: 
                            list(sympy_expr.args)
                        except:# Most probable, parameter name is sympy function name: hypothesis of no simplification
                            sympy_expr.args = ()
                        if not(len(list(sympy_expr.args)) == 0):
                            list_change = True
                            del parameter_list[idx]
                            for argument in list(sympy_expr.args):
                                # Check that argument is not a value
                                try:
                                    float(str(argument))
                                except:                               
                                    parameter_list.append(str(argument))
                # When parsing is stuck because 1/expression is found, replace 1/expression by expression with all terms power changed
                for idx in range(len(parameter_list)):
                    parameter = parameter_list[idx]
                    if parameter[0:2] == '1/':
                        old_parameter = '/' + parameter[2:len(parameter)]
                        parameter = parameter[2:len(parameter)]
                        if parameter[0] == '(' and parameter[len(parameter) - 1] == ')':
                            parameter = parameter[1:len(parameter) - 1]
                        # Get the parameter sublist
                        sympy_expr = parse_expr(parameter, evaluate=False, transformations=transformations)
                        try: 
                            if len(list(sympy_expr.args)) == 0:
                                raise ValueError                            
                            subparameter_list = numpy.array([])
                            for subparameter in list(sympy_expr.args):
                                subparameter_list = numpy.append(subparameter_list, str(subparameter))
                            # Write the multiple expressions for old_parameter because parser can modify terms oder
                            parenthesis = True if old_parameter[1] == '(' else False
                            combination_list = list(permutations(subparameter_list.tolist()))
                            for combination in combination_list:
                                old_parameter = '/(' if parenthesis else '/'
                                for subparameter in combination:
                                    if subparameter[0:2] == '1/':
                                        old_parameter = old_parameter[0:len(old_parameter) - 1] + subparameter[3:len(subparameter)] + '*'
                                    else:
                                        old_parameter += subparameter + '*'
                                old_parameter = old_parameter[0:len(old_parameter) - 1]
                                old_parameter = old_parameter + ')' if parenthesis else old_parameter
                                if old_parameter in expression:
                                    break
                            subparameter_length = numpy.array([]).astype(int)
                            for subparameter in subparameter_list:
                                subparameter_length = numpy.append(subparameter_length, len(str(subparameter)))
                            subparameter_list = subparameter_list[numpy.argsort(-1 * subparameter_length)].tolist()
                            for sub_idx in range(len(subparameter_list)):
                                subparameter = subparameter_list[sub_idx]
                                if subparameter[0:2] == '1/':
                                    subparameter = subparameter[2:len(subparameter)]
                                    subparameter_list[sub_idx] = subparameter
                                    reverse_exponent = False
                                else:
                                    reverse_exponent = True
                                if subparameter[0] == '(' and subparameter[len(subparameter) - 1] == ')':
                                    parameter = parameter.replace(subparameter, '{' + str(sub_idx) + '}' + '**-1') if reverse_exponent else parameter.replace('/' + subparameter, '{' + str(sub_idx) + '}')
                                else:
                                    exp_idx = subparameter.rfind('**')
                                    if exp_idx == -1:
                                        parameter = parameter.replace(subparameter, '{' + str(sub_idx) + '}' + '**-1') if reverse_exponent else parameter.replace('/' + subparameter, '{' + str(sub_idx) + '}')
                                    else:
                                        exponent = subparameter[exp_idx + 2:len(subparameter)]
                                        subparameter_list[sub_idx] = subparameter[0:exp_idx]
                                        parameter = parameter.replace(subparameter, '{' + str(sub_idx) + '}**' + str(-1 * float(exponent))) if reverse_exponent else parameter.replace('/' + subparameter, '{' + str(sub_idx) + '}**' + exponent)
                            parameter_expression = '\'' + parameter + '\'.format('
                            for subparameter in subparameter_list:
                                parameter_expression += '\'' + subparameter + '\','
                            parameter_expression = parameter_expression[0:len(parameter_expression) - 1] + ')'
                            new_parameter = eval(parameter_expression)
                            parameter = new_parameter
                            new_parameter = '*' + new_parameter
                        except:
                            # Only one parameter found with no exponent, case .../u-> 1/u -> parameter u
                            new_parameter = '*' + parameter + '**-1'
                        expression = expression.replace(old_parameter, new_parameter)
                        parameter_list[idx] = parameter
            # Overwritte pi expression
            pi_list[pi_number] = expression
            # Check that parsed values are saved parameters
            for parameter in parameter_list:
                if not(parameter in parameter_set.dictionary.keys()):
                    raise SyntaxError('from pi{} expression, {} not in parameter_set.'.format(pi_number, parameter))
        # Calculate the dimension matrix of the variables
        dimensional_matrix = write_dimensional_matrix(parameter_set)
        # Extract parameters' coefficient from the PI expression (extracting first bigger parameters'names) and write dimension
        parameter_list = numpy.array(list(parameter_set.dictionary.keys()))
        parameter_length = numpy.array([])
        for parameter in parameter_list:
            parameter_length = numpy.append(parameter_length, len(parameter))
        parameter_list = parameter_list[numpy.argsort(-1 * parameter_length)].tolist()
        bounds_list = numpy.zeros([2, len(pi_list)]).astype(float)
        pi_parameters = numpy.zeros([len(pi_list), numpy.shape(dimensional_matrix)[0]]).astype(int)
        for pi_number in range(len(pi_list)):
            expression = pi_list[pi_number]
            pi_exponents = numpy.zeros(numpy.shape(dimensional_matrix)[1]).astype(float)
            bounds = numpy.array([]).astype(float)
            new_expression = ''
            for parameter in parameter_list:
                if parameter in expression:
                    index_start = expression.index(parameter) + len(parameter)
                    # Switch between 3 possibilities *parameter, *parameter*other_parameter, *parameter**exponent[*other_parameter...]
                    if index_start == len(expression):
                        exponent = 1.0
                        expression = expression.replace(parameter, '#')
                    else:
                        if expression[index_start + 1] != '*':
                            exponent = 1.0
                            expression = expression.replace(parameter, '#')
                        else:
                            index_start += 2
                            index_stop = len(expression) if expression.find('*', index_start) == -1 else expression.find('*', index_start)
                            exponent = float(expression[index_start:index_stop])
                            expression = expression.replace(parameter + '**' + expression[index_start:index_stop], '#')
                    pi_exponents += exponent * dimensional_matrix.loc[parameter, :].values
                    if len(new_expression) == 0:
                        new_expression = parameter + '**' + str(exponent)
                    else:
                        new_expression += '*' + parameter + '**' + str(exponent)
                    pi_parameters[pi_number, parameter_list.index(parameter)] = exponent
                    if len(bounds) == 0:
                        if exponent < 0:
                            bounds = numpy.array([parameter_set[parameter]._SI_bounds[1], parameter_set[parameter]._SI_bounds[0]])**exponent
                        else:
                            bounds = numpy.array(parameter_set[parameter]._SI_bounds)**exponent
                    else:
                        if exponent < 0:
                            bounds = bounds * (numpy.array([parameter_set[parameter]._SI_bounds[1], parameter_set[parameter]._SI_bounds[0]])**exponent)
                        else:
                            bounds = bounds * (numpy.array(parameter_set[parameter]._SI_bounds)**exponent)
            bounds_list[:, pi_number] = bounds
            pi_list[pi_number] = new_expression
            # Check pi is dimensionless
            pi_exponents = pandas.DataFrame(numpy.array([pi_exponents,pi_exponents]).astype(int), columns=dimensional_matrix.columns.values.tolist())
            try:
                pi_exponents = pi_exponents.drop(columns=['dimensionless'])
            except:
                pass
            pi_exponents = pi_exponents.values.tolist()[0]
            if numpy.sum(numpy.abs(pi_exponents)) != 0:
                raise ValueError('at least pi{} is not dimensionless.'.format(pi_number))
        # Check pi are independent on parameter space
        if numpy.linalg.matrix_rank(pi_parameters) < len(pi_list):
            raise ValueError('pi set does not cover the dimension set.')
        # Check that pi number is equal to parameter number minus dimensions number
        if not(len(pi_list) == len(parameter_list) - numpy.linalg.matrix_rank(dimensional_matrix)):
            raise ValueError('pi set dimension should be {}.'.format(len(parameter_list) - numpy.shape(dimensional_matrix)[1]))
        # Save pi set
        bounds_list = numpy.transpose(bounds_list)
        pi_set = PositiveParameterSet()
        for pi_number in range(len(pi_list)):
            #Compute PI name
            pi_name = 'pi' + str(pi_number + 1)
            #Save parameter and expression list
            exec(pi_name + ' = PositiveParameter(\'' + pi_name + '\',bounds_list[{},:].tolist(),\'\',\''.format(pi_number) + pi_list[pi_number] + '\')')
            pi_set[pi_name]=eval(pi_name)
        return pi_set
    else:
        if not(isinstance(parameter_set, PositiveParameterSet)):
            raise TypeError('parameter_set should be PositiveParameterSet.')
        else:
            raise SyntaxError('pi(s) should be defined using tuple of string expressions.')

#-------[Define function translating x into pi]--------------------------------
def declare_func_x_to_pi(parameters_set, pi_set):
    """Functions that declare pi=f(x) to transform parameters set values into pi set values.
        
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        pi_set: PositiveParameterSet
                Defines the k (k<n) dimensionless parameters of the problem
        
        Returns
        -------
        f: function
            a function of X, **X** being a [m*n] numpy.array of float representing physical parameters values 
            which returns a [m*k] numpy.array of float corresponding to the dimensionless parameters values
        
        Example
        -------
        define a positive set first, see: :func:`~sizinglab.addon.variablepowerlaw.write_dimensional_matrix`
        
        define pi set using buckingham:
            >>> In [7]: pi_set, _ = buckingham_theorem(parameter_set, False)
        
        set x values:
            >>> In [8]: x = [[1, 1.5, 2, 3, 5],[0.1, 2, 1, 2, 1],[2, 1, 3, 1.5, 2]]
        
        declare function and compute y values:
            >>> In [9]: func_x_to_pi = declare_func_x_to_pi(parameter_set, pi_set)
            >>> In [10]: func_x_to_pi(x)
                array([[ 2.  ,  2.  ,  5.  ],
                       [10.  ,  0.01, 10.  ],
                       [ 1.5 ,  6.  ,  1.  ]])
            
    """
    if isinstance(parameters_set, PositiveParameterSet) and isinstance(pi_set, PositiveParameterSet):
        def f(X):
            Y = []
            X_T = numpy.transpose(X)
            # Get pi equations
            for pi_parameter in pi_set.dictionary.keys():
                equation = pi_set[pi_parameter].description
                V = []
                # Create parameters list and index in parameter_set (for column handling on doe)
                parameter_list = numpy.array(list(parameters_set.dictionary.keys()))
                parameter_index = numpy.array(list(range(len(parameters_set.dictionary.keys()))))
                # Sort parameter by length in order to extract first bigger parameter names (that cannot be included in others...)
                parameter_length = numpy.array([]).astype(int)
                for parameter in parameter_list.tolist():
                    parameter_length = numpy.append(parameter_length, len(parameter))
                parameter_list = parameter_list[numpy.argsort(-1*parameter_length)].tolist()
                parameter_index = parameter_index[numpy.argsort(-1*parameter_length)].tolist()
                # Search for parameters included in the equations and remove it
                for index in range(len(parameter_list)):
                    x_parameter = parameters_set[parameter_list[index]].name
                    if equation.find(x_parameter) != -1:
                        # Find parameter index in equation
                        idx_start = equation.find(x_parameter)
                        # Then find end of exponent expression
                        if equation.find("*", idx_start + len(x_parameter) + 2)==-1:
                            idx_end = len(equation)
                        else:
                            idx_end = equation.find("*", idx_start + len(x_parameter) + 2)
                        # Extract exponent value
                        exponent = float(equation[idx_start + len(x_parameter) + 2:idx_end])
                        # Remove from expression
                        if idx_start == 0:
                            if idx_end == len(equation):
                                equation = ''
                            else:
                                equation = equation[idx_end + 1:len(equation)]
                        elif idx_end == len(equation):
                            equation = equation[0: idx_start - 1]
                        else:
                            equation = equation[0: idx_start - 1] + equation[idx_end :len(equation)]
                        # Copy parameters'values not to overwrite X_T
                        X_values = numpy.copy(X_T[parameter_index[index]])
                        # Overwrite 0 values if parameter exponent is negative (does not happen when used in VPLM with xi>0)
                        if exponent<0:
                            X_values[X_values==0] = float('nan')
                        value = numpy.transpose(numpy.power(numpy.transpose(X_values), abs(exponent)))
                        value = 1/value if exponent<0 else value
                        V = value if len(V)==0 else numpy.multiply(V, value)
                Y.append(V.tolist())
            Y = numpy.transpose(Y)
            return Y
        return f

#-------[Define function extracting regression model with increased complexity]
def regression_models(doe, elected_pi0, order, **kwargs):
    """Functions that calculate the regression model coefficient with increasing model complexity.
        The added terms for complexity increase are sorted depending on their regression coefficient value on standardized pi.
        For more information on regression see Scipy linalg method :func:`~scipy.linalg.lstsq`
        
        Parameters
        ----------
        doe: [m*k] numpy.array of float or int
             Represents the elected feasible constrained sets of m experiments values expressed over the k dimensionless parameters
        
        elected_pi0: str
                     Selected pi for regression: syntax is 'pin' with n>=1 and n<=k
        
        order: int
               * Model order >=1: as an example, order 2 in log_space=True is :
                   log(pi0) = log(cst) + a1*log(pi1) + a11*log(pi1)**2 + a12*log(pi1)*log(pi2) + a2*log(pi2) + a22*log(pi2)**2
               * Model order >=1: as an example, order 2 in log_space=False is :
                   pi0 = cst + a1*pi1 + a11*pi1**2 + a12*pi1*pi2 + a2*pi2 + a22*pi2
        
        **kwargs: additional argumens 
                  * **ymax_axis** (*float*): set y-axis maximum value representing relative error, 100=100% (default value)
                  * **pi_list** (*list* of *str*): set particular name for the f(pi1, pi2, ..., piN)
                  * **log_space** (*bool*): define if polynomial regression should be performed within logarithmic space (True) or linear (False)
                  * **latex** (*bool*): define if graph legend font should be latex (default is False) - may cause some issues if used
        
        Returns
        -------
        models: dict of [1*4] tuple
                Stores the different models regression information as for model 'i':
                    * dict[i][0]: str of the model expression
                    * dict[i][1]: numpy.array of the regression coefficients
                    * dict[i][2]: pandas.DataFrame of the trained set (**max abs(e)**, **average abs(e)**, **average e** and **sigma e**)
                    * dict[i][3]: pandas.DataFrame of the tested set (**max abs(e)**, **average abs(e)**, **average e** and **sigma e**)
                    
                    Where **e** represents the relative error on elected_pi0>0 in %
                    
                Additional data is saved in:
                    * dict['max abs(e)']: (1*2) tuple containing trained and test sets max absolute relative error on the k-models
                    * dict['ave. abs(e)']: (1*2) tuple containing trained and test sets average absolute relative error on the k-models
                    * dict['ave. e']: (1*2) tuple containing trained and test sets average relative error on the k-models
                    * dict['sigma e']: (1*2) tuple containing trained and test sets standard deviation on relative error on the k-models
        
        Example
        -------
        to define the parameter and pi sets and generate DOE: to define the parameter and pi sets refer to: :func:`~sizinglab.addon.variablepowerlaw.reduce_parameter_set`
        
        generate mathematic relation between Pi parameters knowing pi1=l/u and following formulas:  
            >>> In [14]: PI2 = doeX['e']*(1/doeX['f'])* doeX['u']**2
            >>> In [15]: PI3 = doeX['d']*(1/doeX['u'])
            >>> In [16]: PI1 = numpy.zeros(len(PI2))
            >>> In [17]: for idx in range(len(PI1)):
                    ...:         PI1[idx] = (10**2.33)*(PI2[idx]**(1.35+0.17*numpy.log10(PI2[idx])+0.042*numpy.log10(PI3[idx])))*(PI3[idx]**-0.25) + (random.random()-0.5)/1000000
            >>> In [18]: l_values = PI1 * doeX['u']
            >>> In [19]: doeX['l'] = l_values   
            >>> In [20]: doeX = doeX[list(parameter_set.dictionary.keys())]
            >>> In [21]: func_x_to_pi = declare_func_x_to_pi(parameter_set, pi_set)
            >>> In [22]: doePI = func_x_to_pi(doeX.values)
            >>> In [23]: models = regression_models(doePI, 'pi1', 3)
                
            .. image:: ../source/_static/Pictures/variablepowerlaw_regression_models1.png
  
    """
    if isinstance(doe, numpy.ndarray) and isinstance(elected_pi0, str) and isinstance(order, int):
        # Check values
        if elected_pi0[0:2] != 'pi':
            raise SyntaxError('elected_pi0 should be of the form pik with k an int.')
        elected_pi0 = int(elected_pi0[2:len(elected_pi0)])
        if (elected_pi0 < 1) or (elected_pi0 > len(doe)):
            raise ValueError('elected_pi0 should be >=1 and <={}.'.format(len(doe)))
        if order < 1:
            raise ValueError('order should be >=1.')
        if not(numpy.issubdtype(doe.dtype, numpy.integer) or numpy.issubdtype(doe.dtype, numpy.float64)):
            raise TypeError('doe type in index should be integer or float.')
        log_space=True
        ymax_axis=100
        pi_list = []
        test_mode = False
        latex = False
        for key, value in kwargs.items():
            if not(key in ['log_space','ymax_axis','pi_list','test_mode', 'latex']):
                raise KeyError('unknown argument ' + key)
            elif key == 'pi_list':
                if isinstance(value, list):
                   if len(value) != numpy.shape(doe)[1]-1:
                       raise ValueError('defined pi_list mismatch doe size')
                else:
                    raise TypeError('pi_list should be a list of string')
            elif key == 'ymax_axis':
                if isinstance(value, int) or isinstance(value, float):
                    if (value <=0):
                        ValueError('ymax_axis should be >0')
                    else:
                        ymax_axis = float(value)
                else:
                    raise TypeError('order should be a float')
            elif key == 'log_space':
                if isinstance(value, bool):
                    log_space = value
                else:
                    raise ValueError('log_space should be a boolean')
            elif key == 'test_mode':
                if isinstance(value, bool):
                    test_mode = value
                else:
                    raise ValueError('test_mode should be a boolean')
            elif key == 'latex':
                if isinstance(value, bool):
                    latex = value
                else:
                    raise ValueError('latex should be a boolean')   
        # Adapt X if necessary and calculate normal centered values
        X_doe = numpy.log10(doe) if log_space else doe
        # Extract chosen Y values
        Y = numpy.copy(X_doe[:, elected_pi0 - 1])
        X_doe = numpy.delete(X_doe, elected_pi0 - 1, 1)
        # Create first labels corresponding to order 1
        poly_feature = PolynomialFeatures(degree=1, include_bias = False)
        X = poly_feature.fit_transform(X_doe)
        term_names = poly_feature.get_feature_names()
        labels = ['']
        for index_of_term, term in enumerate(term_names):
            for k in range(numpy.shape(X)[1] + 1):
                if k == elected_pi0 - 1:
                    continue
                elif k > elected_pi0 - 1:
                    delta = 1
                else:
                    delta = 0
                if log_space:
                    term = term.replace("x" + str(k - delta), 'log(pi' + str(k + 1) + ')')
                else:
                    term = term.replace("x" + str(k - delta), 'pi' + str(k + 1))
            term = term.replace("^","**")
            term = term.replace(" ","*")
            labels.append(term)
        # Extract constant to set it first
        X_c = pandas.DataFrame(numpy.c_[numpy.ones(numpy.shape(X)[0]), X], columns=labels)
        X = X_c[labels[1:len(labels)]]
        # Calculate the regression coefficient (on standardized values) to rank coefficients (constant been extracted)
        X_norm = 2 * ((X.values - numpy.amin(X.values, axis=0)) / (numpy.amax(X.values, axis=0) - numpy.amin(X.values, axis=0))) - 1
        Y_norm = 2 * ((Y - numpy.amin(Y, axis=0)) / (numpy.amax(Y, axis=0) - numpy.amin(Y, axis=0))) - 1
        coeff, _, _, _ = scipy.linalg.lstsq(X_norm, Y_norm)
        initial_order = numpy.array(X.columns.tolist())
        final_order = initial_order[numpy.argsort(-1 * numpy.absolute(coeff))]
        labels1 = [labels[0]] + final_order.tolist()
        X = X_c[labels1]
        # Consider ordering order>1 parameters
        if order > 1:
            poly_feature = PolynomialFeatures(degree=order, include_bias = False)
            X = poly_feature.fit_transform(X_doe)
            term_names = poly_feature.get_feature_names()
            labels = ['']
            for index_of_term, term in enumerate(term_names):
                for k in range(numpy.shape(X)[1] + 1):
                    if k == elected_pi0 - 1:
                        continue
                    elif k > elected_pi0 - 1:
                        delta = 1
                    else:
                        delta = 0
                    if log_space:
                        term = term.replace("x" + str(k - delta), 'log(pi' + str(k + 1) + ')')
                    else:
                        term = term.replace("x" + str(k - delta), 'pi' + str(k + 1))
                term = term.replace("^","**")
                term = term.replace(" ","*")
                labels.append(term)
            # Extract constant to set it first
            X_c = pandas.DataFrame(numpy.c_[numpy.ones(numpy.shape(X)[0]), X], columns=labels)
            X = X_c[labels[1:len(labels)]]
            # Calculate the regression coefficient (on standardized values) to rank coefficients (constant been extracted)
            X_norm = 2 * ((X.values - numpy.amin(X.values, axis=0)) / (numpy.amax(X.values, axis=0) - numpy.amin(X.values, axis=0))) - 1
            Y_norm = 2 * ((Y - numpy.amin(Y, axis=0)) / (numpy.amax(Y, axis=0) - numpy.amin(Y, axis=0))) - 1
            coeff, _, _, _ = scipy.linalg.lstsq(X_norm, Y_norm)
            initial_order = numpy.array(X.columns.tolist())
            final_order = initial_order[numpy.argsort(-1 * numpy.absolute(coeff))]
            final_order = final_order.tolist()
            # Delete from final_order order 1 labels
            for label_name in labels1:
                if label_name != '':
                    final_order.remove(label_name)  
            labels2 = labels1 + final_order
            X = X_c[labels2]
        # Calculate the regression models considering parameters by order of decreasing correlation
        models = {}
        for idx in range(numpy.shape(X.values)[1]):
            error_test = numpy.array([])
            values = X.values[:, 0:idx + 1]
            for test_idx in range(numpy.shape(values)[0]):
                coeff, _, _, _ = scipy.linalg.lstsq(numpy.delete(values, test_idx, 0), numpy.delete(Y, test_idx, 0))
                # Calculate error on trained set when selecting last point for cross validation (saved model expression)
                if test_idx == (numpy.shape(values)[0] - 1):
                    y_pred = numpy.dot(numpy.delete(values, test_idx, 0), coeff)
                    if log_space:
                        y_pred = 10.0**y_pred
                        y_data = 10.0**numpy.delete(Y, test_idx, 0)
                    else:
                        y_data = numpy.delete(Y, test_idx, 0)
                    error_train = (y_pred - y_data) * (1 / y_data) * 100
                # Calculate and save the error on the tested point (cross-validation)
                y_pred = numpy.dot(values[test_idx, :], coeff)
                if log_space:
                    y_pred = 10.0**y_pred
                    y_data = 10.0**Y[test_idx]
                else:
                    y_data = Y[test_idx]
                error_test = numpy.append(error_test, (y_pred - y_data) * (1 / y_data) * 100)
            # Write model expression
            labels =  numpy.array(X.columns.tolist())
            if log_space:
                expression = 'log(pi' + str(elected_pi0) + ') = '
            else:
                expression = 'pi' + str(elected_pi0) + ' = '
            for i in range(0, idx + 1):
                if coeff[i] < 0:
                    if labels[i] == '':
                        expression = expression[0:len(expression) - 1] + '{:.5f}'.format(coeff[i]) + '+'
                    else:
                        expression = expression[0:len(expression) - 1] + '{:.5f}*'.format(coeff[i]) + labels[i] + '+'
                else:
                    if labels[i] == '':
                        expression += '{:.5f}'.format(coeff[i]) + labels[i] + '+'
                    else:
                        expression += '{:.5f}*'.format(coeff[i]) + labels[i] + '+'
            expression = expression[0:len(expression) - 1]
            # Calculate max and average absolute error and average and sigma error on train data
            error_average = numpy.mean(error_train)
            error_sigma = numpy.std(error_train)
            abs_error_average = numpy.mean(numpy.absolute(error_train))
            abs_error_max = numpy.amax(numpy.absolute(error_train))
            error_train = numpy.array([abs_error_max, abs_error_average, error_average, error_sigma])
            error_train = pandas.DataFrame(error_train, index=['max |e|','ave. |e|', 'ave. e', 'sigma e'])
            # Calculate max and average absolute error and average and sigma error on test data
            error_average = numpy.mean(error_test)
            error_sigma = numpy.std(error_test)
            abs_error_average = numpy.mean(numpy.absolute(error_test))
            abs_error_max = numpy.amax(numpy.absolute(error_test))
            error_test = numpy.array([abs_error_max, abs_error_average, error_average, error_sigma])
            error_test = pandas.DataFrame(error_test, index=['max |e|','ave. |e|', 'ave. e', 'sigma e'])
            # Save data
            models[len(list(models.keys())) + 1] = (expression, coeff, error_train, error_test)
        # Extract the 4 indicators results stored in models
        abs_error_max_train = []
        abs_error_max_test = []
        abs_error_average_train = []
        abs_error_average_test = []
        error_average_train = []
        error_average_test = []
        error_sigma_train = []
        error_sigma_test = []
        for key in models.keys():
            abs_error_max_train.append(float(models[key][2].values[0]))
            abs_error_max_test.append(float(models[key][3].values[0]))
            abs_error_average_train.append(float(models[key][2].values[1]))
            abs_error_average_test.append(float(models[key][3].values[1]))
            error_average_train.append(float(models[key][2].values[2]))
            error_average_test.append(float(models[key][3].values[2]))
            error_sigma_train.append(float(models[key][2].values[3]))
            error_sigma_test.append(float(models[key][3].values[3]))
        # Set latex render on plot
        if latex:
            plot.rc('text', usetex=True)
            plot.rc('font', family='serif')
        # Start to plot the graph with indicators
        x = numpy.array(range(len(models.keys()))).astype(int) + 1
        fig, axs = plot.subplots(4, sharex=True, gridspec_kw={'hspace':0.05}, figsize=(8, 10))
        # Plot maximum absolute relative error
        axs[0].plot(x, numpy.array(abs_error_max_train), 'k-*', label='Fitting set')
        axs[0].plot(x, numpy.array(abs_error_max_test), 'r-*', label='Validation set')
        y_max = min(ymax_axis, max(max(abs_error_max_train), max(abs_error_max_test)))
        axs[0].axis([numpy.amin(x), numpy.amax(x), 0, y_max])
        axs[0].set_ylabel(r'(1): $\max \mid \epsilon \mid$',fontsize=18)
        axs[0].grid(True)
        axs[0].legend(fontsize=16)
        # Plot average |relative error| 
        axs[1].plot(x, numpy.array(abs_error_average_train), 'k-*', label='Fitting set')
        axs[1].plot(x, numpy.array(abs_error_average_test), 'r-*', label='Validation set')
        y_max = min(ymax_axis, max(max(abs_error_average_train), max(abs_error_average_test)))
        axs[1].axis([numpy.amin(x), numpy.amax(x), 0, y_max])
        axs[1].set_ylabel(r'(2): $\frac{1}{n} \cdot \sum_{i=1}^n \mid \epsilon \mid$',fontsize=18)
        axs[1].grid(True)
        # Plot the |average relative error|
        axs[2].plot(x, numpy.absolute(error_average_train), 'k-*', label='Fitting set')
        axs[2].plot(x, numpy.absolute(error_average_test), 'r-*', label='Validation set')
        y_max = min(ymax_axis, max(max(map(abs, error_average_train)), max(map(abs, error_average_test))))
        axs[2].axis([numpy.amin(x), numpy.amax(x), 0, y_max])
        axs[2].set_ylabel(r'(3): $\mid \overline{\epsilon} \mid = \mid \mu \mid$',fontsize=18)
        axs[2].grid(True)
        # Plot the standard deviation
        axs[3].plot(x, numpy.absolute(error_sigma_train), 'k-*', label='Fitting set')
        axs[3].plot(x, numpy.absolute(error_sigma_test), 'r-*', label='Validation set')
        y_max = min(ymax_axis, max(max(map(abs, error_sigma_train)), max(map(abs, error_sigma_test))))
        axs[3].axis([numpy.amin(x), numpy.amax(x), 0, y_max])
        axs[3].set_ylabel(r'(4): $\mid \sigma_{\epsilon} \mid$',fontsize=18)
        axs[3].set_xlabel('Model terms number',fontsize=16)
        axs[3].grid(True)
        majors = range(len(x)+1)
        majors = ['$n^o' + str(majors[i]) +'$' for i in range(len(majors))]
        axs[3].xaxis.set_major_locator(ticker.MultipleLocator(1))
        axs[3].xaxis.set_major_formatter(ticker.FixedFormatter(majors))
        try:
            plot.savefig('../_temp/regression_models_fig1.pdf', dpi=1200, format='pdf')
        except:
            pass
        if not(test_mode):
            plot.show()
        # De-activate latex render on plot
        plot.rc('text', usetex=False)
        plot.rc('font', family='sans-serif')
        # Save data directly into models
        models['max |e|'] = [abs_error_max_train, abs_error_max_test]
        models['ave. |e|'] = [abs_error_average_train, abs_error_average_test]
        models['ave. e'] = [error_average_train, error_average_test]
        models['sigma e'] = [error_sigma_train, error_sigma_test]
        # Display model expression and error
        error_average_test = numpy.absolute(error_average_test)
        error_average_test = error_average_test.tolist()
        best_model = abs_error_average_test.index(min(abs_error_average_test))
        best_model = best_model[0] if isinstance(best_model, list) else best_model
        if not(test_mode):
            print('Model elected on criteria (2) is n°{}:'.format(best_model + 1))
        expression = models[best_model + 1][0]
        # Concatenate model formula and change to latex
        default_list = False
        if len(pi_list) != numpy.shape(doe)[1]:
            default_list = True
        else:
            for i in range(len(pi_list)):
                if not(isinstance(pi_list(i),str)):
                    default_list = True
                    break
        if default_list:
            pi_list = []
            for i in range(numpy.shape(doe)[1]):
                pi_list.append('\pi_{' + str(i + 1) + '}')
        try:
            expression_latex = concatenate_expression(expression, pi_list)
            if not(test_mode):
                display(Latex(expression_latex))
        except:
            if not(test_mode):
                print(expression + '\n')
        # Save expression
        try:
            hs = open("../_temp/latex_formula.txt","a")
            hs.write(expression_latex + '\n')
            hs.close() 
        except:
            pass
        return models
    else:
        if not(isinstance(doe, numpy.ndarray)):
            raise TypeError('doe should be numpy array.')
        elif not(isinstance(elected_pi0, str)):
            raise TypeError('elected_pi0 should be a string.')
        else:
            raise TypeError('order should be an integer.')
            
#-------[Define function to concatenate regression formula]--------------------
def concatenate_expression(expression, pi_list):
    """Function that transform regression model expression into latex form with concatenation (only for power-laws).
        
        Parameters
        ----------
        expression: Str
                    Expression of the regression model
        
        pi_list: List of str
                 Defines the pi expressions (default is pi1, pi2...)

        Returns
        -------
        new_expression: Str 
                        Represents the new model formula in latex form
        
        Example
        -------
        define expression and list:
            >>> In [1]: expression = 'log(pi1) = 2.33011+1.35000*log(pi2)-0.25004*log(pi3)+0.04200*log(pi2)*log(pi3)+0.17000*log(pi2)**2'
            >>> In [2]: pi_list = ['\pi_1','\pi_2','\pi_3']
        adapt expression:
            concatenate_expression(expression, pi_list)
            
    """
    if expression[0:4] == 'log(':
        expression = expression.split('=')
        new_expression = expression[0]
        new_expression = new_expression.replace(' ','')
        expression = expression[1]
        expression = expression.replace(' ','')
        expression = expression.replace('+','#+')
        expression = expression.replace('-','#-')
        if expression[0] == '#':
            expression = expression[1:len(expression)]
        expression = expression.split('#')
        power_list = {}                              
        for i in range(len(pi_list)):
            if len(expression) == 0:
                break
            pi_name= 'log(pi'+ str(i+1) + ')'
            if new_expression == pi_name:
                new_expression = '$' + pi_list[i] + '='
                continue
            keep = numpy.ones(len(expression)).astype(bool)
            for idx in range(len(expression)):
                local_expression = expression[idx]
                if local_expression.find(pi_name) != -1:
                    keep[idx] = False
                    idx = local_expression.index(pi_name)
                    if idx + len(pi_name) != len(local_expression):
                        if  local_expression[idx + len(pi_name):idx + len(pi_name) + 2] == '**':
                            try:
                                exponent = int(local_expression[idx + len(pi_name) + 2:local_expression.index('*', idx + len(pi_name) + 2)])
                            except:
                                exponent = int(local_expression[idx + len(pi_name) + 2:len(local_expression)])
                        else:
                            exponent = 1 
                    else:
                        exponent = 1
                    if exponent > 2:
                        local_expression = local_expression.replace(pi_name + '**' + str(exponent),'log(' + pi_list[i] + ')^' + str(exponent-1))
                    elif exponent == 2 :
                        local_expression = local_expression.replace(pi_name + '**' + str(exponent),'log(' + pi_list[i] + ')')
                    else:
                        local_expression = local_expression.replace(pi_name + '*','')
                        local_expression = local_expression.replace('*' + pi_name,'')
                    local_expression = local_expression.replace('**','^')
                    local_expression = local_expression.replace('*',' \cdot ')
                    for k in range(len(pi_list)):
                        local_expression = local_expression.replace('pi' + str(k+1),pi_list[k])
                    if pi_name in power_list.keys():
                        power_list[pi_name] += local_expression
                    else:
                        power_list[pi_name] = local_expression
            expression = numpy.array(expression)
            expression = expression[keep]
            expression = expression.tolist()
        new_expression += '10^{' + str(expression[0]) + '}'
        for i in range(len(pi_list)):
            pi_name= 'log(pi'+ str(i+1) + ')'
            if pi_name in power_list.keys():
                power_list[pi_name] = power_list[pi_name].replace('+-','-')
                new_expression += ' \cdot ' + pi_list[i] + '^{' + power_list[pi_name] + '}'
        new_expression += '$'
    else:
        new_expression = expression
        for i in range(len(pi_list)):
            pi_name= 'pi'+ str(i+1)
            new_expression = new_expression.replace(pi_name, pi_list[i])
        new_expression = new_expression.replace('**', '^')
        new_expression = new_expression.replace('*', ' \cdot ')
        new_expression = '$' + new_expression + '$'
    return new_expression

#-------[Define function to change xi set definition after FEM calculation]----
def adapt_parameter_set(parameter_set, pi_set, doeX, replaced_parameter, new_parameter, expression, description):
    """Function that transform physical parameters problem (and corresponding DOE) after FEM calculation.
        
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        pi_set: PositiveParameterSet
                Defines the k (k<n) dimensionless parameters of the problem
        
        doeX: pandas.DataFrame 
              DOE of the parameter_set in SI units (column names should be of the form 'parameter_i')
        
        replaced_parameter: str
                            Name of the replaced parameter (should not be a repetitive term, i.e. present in only one PI (PI0) term to ensure proper PI spacing)
        
        new_parameter: str
                       Name of new parameter (units has to be identical to replaced parameter)
        
        expression: str 
                    Relation between old and new parameter if x_new = 2*x_old+3*other_parameter, write '2*x_old+3*other_parameter'
        
        description: str
                     Saved description for new parameter
        
        Returns
        -------
        new_parameter_set: PositiveParamaterSet 
                           Represents the new physical problem
        
        new_pi_set: PositiveParameterSet 
                    Dimensionless parameters set derived from pi_set replacing parameter
        
        new_doeX: pandas.DataFrame 
                  Computed DOE from doeX using expression (relation between parameters)
        
        Example
        -------
        to define the parameter, pi sets and calculate DOE refer to: :func:`~sizinglab.addon.variablepowerlaw.regression_models`
        
        save DOE into dataframe:
            >>> In [9]: labels = list(parameter_set.dictionary.keys())
            >>> In [10]: doeX = pandas.DataFrame(doeX, columns=labels)
        
        then imagine you want to replace one parameter (after FEM simulation):
            >>> In [11]: new_parameter_set, new_pi_set, new_doeX = adapt_parameter_set(parameter_set, pi_set, doeX, 'd', 'd_out', 'd+2*e', 'outer diameter')
            
        you are able to perform new regression calculation!
        
    """
    if isinstance(parameter_set, PositiveParameterSet) and isinstance(pi_set, PositiveParameterSet) and isinstance(doeX, pandas.DataFrame)\
        and isinstance(replaced_parameter, str) and isinstance(expression, str) and isinstance(new_parameter, str) and isinstance(description, str):
        # Check that replaced parameter is in the set
        if not(replaced_parameter in parameter_set.dictionary.keys()):
            raise KeyError('replaced_parameter not in parameter_set dictionary key(s).')
        # Check that doeX and parameter_set match
        parameter_list = doeX.columns.values.tolist()
        for parameter in parameter_list:
            if not(parameter in parameter_set.dictionary.keys()):
                raise ValueError('doeX column names and parameter_set key(s) mismatch.')
        if len(list(parameter_set.dictionary.keys())) != len(parameter_list):
            raise ValueError('doeX column and parameter_set key(s) size mismatch.')
        # Validate expression syntax (including that parameters are in the set) by replacing parameter by first encountered value
        if ('=' in expression) or ('<' in expression) or ('>' in expression):
            raise SyntaxError('expression syntax not correct, =/</> operand should not be used.')
        parameter_list =  numpy.array(parameter_list)
        parameter_index = numpy.array(range(len(parameter_list))).astype(int)
        parameter_length = numpy.array([]).astype(int).astype(int)
        for parameter in parameter_list.tolist():
            parameter_length = numpy.append(parameter_length, len(parameter))
        parameter_list = parameter_list[numpy.argsort(-1*parameter_length)].tolist()
        parameter_index = parameter_index[numpy.argsort(-1*parameter_length)].tolist()
        expression_1 = expression
        for idx in range(len(parameter_list)):
            parameter_name = parameter_list[idx]
            if parameter_name in expression_1:
                expression_1 = expression_1.replace(parameter_name, '[][:, {}]'.format(parameter_index[idx]))
        expression_1 = expression_1.replace('[]','doeX.values')
        try:
            values = eval(expression_1)
        except:
            raise SyntaxError('expression syntax is not correct.')
        # Check new parameter values
        if numpy.amin(values)<= 0:
            raise ValueError('{} parameter tend to have <=0 values cannot be saved in new PositiveParameterSet.'.format())
        # Check expression dimension
        ureg = pint.UnitRegistry()
        Q_ = ureg.Quantity
        for idx in range(len(parameter_list)):
            parameter_name = parameter_list[idx]
            exec(parameter_name + '=' + 'Q_(doeX.values[0, {}],\''.format(parameter_index[idx]) + '{}\')'.format(parameter_set[parameter_name]._SI_units))
        try:
            value = eval(expression)
        except:
            raise SyntaxError('expression syntax is not correct: units not homogenous.')
        if value.units != parameter_set[replaced_parameter]._SI_units:
            raise ValueError('replaced and new parameters should have the same dimension.')
        # Save new parameter and parameter set
        new_parameter_set = copy.deepcopy(parameter_set)
        bounds = [numpy.amin(values), numpy.amax(values)]
        exec(new_parameter + '=PositiveParameter(\'{}\',bounds,\'{}\',description)'.format(new_parameter, value.units))
        if eval(new_parameter + '.name') in parameter_set.dictionary.keys():
            raise KeyError('Parameter name has same name as existing parameter: {}'.format(eval(new_parameter + '.name')))
        exec('new_parameter_set['+ new_parameter +'.name] =' + new_parameter)
        del new_parameter_set[replaced_parameter]
        # Save new doeX
        new_doeX = pandas.DataFrame.copy(doeX)
        new_doeX = new_doeX.drop(replaced_parameter, axis=1)
        new_doeX[new_parameter] = values
        # Change pi expression
        nb_expression_changed = 0
        new_pi_set = copy.deepcopy(pi_set)
        for pi in pi_set.dictionary.keys():
            expression = pi_set[pi].description
            expression_1 = expression
            for idx in range(len(parameter_list)):
                parameter_name = parameter_list[idx]
                if parameter_name == replaced_parameter:
                    if parameter_name in expression_1:
                        idx_start = expression_1.find(parameter_name)
                        idx_end = idx_start + len(parameter_name) - 1
                        if idx_start != 0:
                            new_expression = expression[0:idx_start-1]
                        else:
                            new_expression = ''
                        new_expression += new_parameter
                        if idx_end != len(expression):
                            new_expression += expression[idx_end+1:len(expression)]
                        new_pi_set[pi].description = new_expression
                        nb_expression_changed += 1
                        break
                else:
                    if parameter_name in expression_1:
                        erase_name = ''
                        for i in range(len(parameter_name)):
                            erase_name += '*'
                        expression_1 = expression_1.replace(parameter_name, erase_name)
        if nb_expression_changed > 1:
            warnings.warn('parameter {} involved in different PI expressions, PI space may be badly covered'.format(replaced_parameter))
        # Adapt pi bounds with new expressions
        func_x_to_pi = declare_func_x_to_pi(new_parameter_set, new_pi_set)
        doePI = func_x_to_pi(new_doeX.values)
        pi_list = list(new_pi_set.dictionary.keys())
        for idx in range(numpy.shape(doePI)[1]):
            pi_name = pi_list[idx]
            new_pi_set[pi_name].defined_bounds = [numpy.amin(doePI[:,idx]), numpy.amax(doePI[:,idx])]
        return new_parameter_set, new_pi_set, new_doeX
    elif not(isinstance(parameter_set, PositiveParameterSet)):
        raise TypeError('parameter_set should be PositiveParameterSet.')
    elif not(isinstance(pi_set, PositiveParameterSet)):
        raise TypeError('pi_set should be PositiveParameterSet.')
    elif not(isinstance(doeX, pandas.DataFrame)):
        raise TypeError('doeX should be DataFrame.')
    elif not(isinstance(replaced_parameter, str)):
        raise TypeError('replaced_parameter should be string.')
    elif not(isinstance(expression, str)):
        raise TypeError('expression should be string.')
    elif not(isinstance(new_parameter, str)):
        raise TypeError('new_parameter should be string.')
    else:
        raise TypeError('description should be string.')

#-------[Define function to reduce problem set extracting FEM output parameter]
def reduce_parameter_set(parameter_set, pi_set, elected_output):
    """Function that reduce physical parameters and Pi set extracting output physical parameter and Pi0.
        
        Parameters
        ----------
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        pi_set: PositiveParameterSet
                Defines the k (k<n) dimensionless parameters of the problem
               
        elected_output: str
                        Parameter that represents FEM output
        
        Returns
        -------
        reduced_parameter_set: PositiveParamaterSet 
                               Parameter set reduced by elected_output
        
        reduced_pi_set: PositiveParameterSet 
                        Pi set reduced by Pi0 (dimensionless parameter countaining output parameter)
        
        Example
        -------
        to define the parameter and pi sets refer to: :func:`~sizinglab.addon.variablepowerlaw.buckingham_theorem`
        
        reduce sets considering 'l' is the output:
            >>> In [9]: reduced_parameter_set, reduced_pi_set = reduce_parameter_set(parameter_set, pi_set, 'l')
        
        then declare transformation function and create DOE:
            >>> In [10]: func_x_to_pi = declare_func_x_to_pi(reduced_parameter_set, reduced_pi_set)
            >>> In [11]: from pixdoe import create_const_doe
            >>> In [12]: doeX, _ = create_const_doe(reduced_parameter_set, reduced_pi_set, func_x_to_pi, 50, numpy.ones(4).astype(int), [], [], 3, 5, True, False)
            >>> In [13]: doeX = pandas.DataFrame(doeX, columns=list(reduced_parameter_set.dictionary.keys()))
    
    """
    if isinstance(parameter_set, PositiveParameterSet) and isinstance(pi_set, PositiveParameterSet) and isinstance(elected_output, str):
        if not(elected_output in parameter_set.dictionary.keys()):
            raise KeyError('elected_output not in parameter_set keys.')
        reduced_parameter_set = copy.deepcopy(parameter_set)
        reduced_pi_set = copy.deepcopy(pi_set)
        # Look into PI expression to see if parameter appears only into one of them
        parameter_list =  numpy.array(list(parameter_set.dictionary.keys()))
        parameter_index = numpy.array(range(len(parameter_list))).astype(int)
        parameter_length = numpy.array([]).astype(int).astype(int)
        for parameter in parameter_list.tolist():
            parameter_length = numpy.append(parameter_length, len(parameter))
        parameter_list = parameter_list[numpy.argsort(-1*parameter_length)].tolist()
        parameter_index = parameter_index[numpy.argsort(-1*parameter_length)].tolist()
        pi_list = list(pi_set.dictionary.keys())
        parameter_in_expression = numpy.zeros(len(pi_list)).astype(bool)
        for idx in range(len(pi_list)):
            expression = pi_set[pi_list[idx]].description
            for parameter in parameter_list:
                if (parameter in expression) and (parameter == elected_output):
                    parameter_in_expression[idx] = True
                    break
                elif parameter in expression:
                   expression = expression.replace(parameter, '')
        if numpy.sum(parameter_in_expression) != 1:
            print('Parameter {} appears in different Pi expression, set have not been reduced.'.format(elected_output))
            return reduced_parameter_set, reduced_pi_set
        # Extract Pi and physical parameter
        idx = int(numpy.argwhere(parameter_in_expression))
        del reduced_pi_set[pi_list[idx]]
        del reduced_parameter_set[elected_output]
        return reduced_parameter_set, reduced_pi_set
    elif not(isinstance(parameter_set, PositiveParameterSet)):
        raise TypeError('parameter_set should be PositiveParameterSet.')
    elif not(isinstance(pi_set, PositiveParameterSet)):
        raise TypeError('pi_set should be PositiveParameterSet.')
    else:
        raise TypeError('elected_output should be string.')

#-------[Define function to import saved doe Dataframe]------------------------
def import_csv(file_name, parameter_set):
    """Function to import .CSV with column label syntax as 'param_name' or 'param_name [units]'.
        Auto-adaptation to SI-units is performed and parameters out of set are ignored/deleted.
        
        Parameters
        ----------
        file_name: str
                   Name of the saved file with path (example: file_name = './subfolder/name')
        
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
    """
    if isinstance(parameter_set, PositiveParameterSet):
        # Load file
        try:
            doeX = pandas.read_csv(file_name, sep=';')
        except:
            raise SyntaxError('Unable to load file!')
        # Get parameter name and units: column name is either 'parameter_name' or parameter_name [units]'
        parameter_list = list(doeX.columns.values)
        units_list = []
        for idx in range(len(parameter_list)):
            parameter = parameter_list[idx]
            if parameter.find(' [') != -1:
                idx_start = parameter.find('[') + 1
                idx_end = parameter.find(']')
                units_list.append(parameter[idx_start:idx_end])
                parameter_list[idx] = parameter[0:parameter.find(' [')]
            else:
                units_list.append('SI')
        # Check parameter and units and adapt values to SI if necessary
        ureg = pint.UnitRegistry()
        ureg.default_system = 'mks'
        Q_ = ureg.Quantity
        for idx in range(len(parameter_list)):
            parameter = parameter_list[idx]
            if parameter in parameter_set.dictionary.keys():
                if units_list[idx] != 'SI':
                    try:
                        value = Q_(1, units_list[idx]).to_base_units()
                        if str(value.units) != parameter_set[parameter]._SI_units:
                            raise ValueError('dimensions mismatch for parameter {}, {} found instead of {}.'.format(parameter, str(value.units),\
                                             parameter_set[parameter]._SI_units))
                        else:
                            # Overwrite parameter column with SI units values
                            values = doeX[parameter + ' [' + units_list[idx] +']']
                            for idx in range(len(values)):
                                value = Q_(values[idx], units_list[idx]).to_base_units()
                                values[idx] = value.magnitude
                            doeX[parameter] = values
                            doeX = doeX.drop(parameter + ' [' + units_list[idx] +']', axis=1)
                    except:
                        warnings.warn('parameter {} units defined in file are unreadable, SI units are applied!'.format(parameter))
            else:
                doeX = doeX.drop(parameter, axis=1)
                warnings.warn('parameter {} not defined in the parameter set, value erased from imported doe.'.format(parameter))
        for parameter in parameter_set.dictionary.keys():
            if not(parameter in parameter_list):
                raise KeyError('parameter {} not in the doe.'.format(parameter))
        # Change column order to match parameter_set definition
        doeX = doeX.reindex(columns=list(parameter_set.dictionary.keys()))
        return doeX
    else:
        raise TypeError('parameter_set should be a PositiveParameterSet')
    
#-------[Define function to save doe Dataframe]--------------------------------
def save_csv(doeX, file_name, parameter_set, is_SI):
    """Function to save .CSV with column label syntax as 'param_name [units]'.
        With units either defined by user or SI (is_SI, True by default).
        
        Parameters
        ----------
        doeX: numpy.array 
              DOE of the parameter_set either in defined_units or SI units
        
        file_name: str
                   Name of the saved file with path (example: file_name = './subfolder/name')
        
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        is_SI: bool
               Define if parameters values are expressed in SI units or units defined by user
        
    """ 
    if isinstance(doeX, numpy.ndarray) and isinstance(file_name, str) and isinstance(parameter_set, PositiveParameterSet) and isinstance(is_SI, bool):
        # Check that data array and parameter set have same size
        if numpy.shape(doeX)[1] != len(list(parameter_set.dictionary.keys())):
            raise ValueError('data dimension mismatch parameter_set keys\'number')
        # Check that values are in defined bounds
        key_list = list(parameter_set.dictionary.keys())
        for idx in range(numpy.shape(doeX)[1]):
            max_value = numpy.amax(doeX[:, idx])
            min_value = numpy.amin(doeX[:, idx])
            bounds = parameter_set[key_list[idx]]._SI_bounds if is_SI else parameter_set[key_list[idx]].defined_bounds
            if (min_value < bounds[0]) or (max_value > bounds[1]):
                warnings.warn('for parameter {} saved values are out of bounds!'.format(key_list[idx]))
        # Write labels and create dataframe
        labels = []
        for key in parameter_set.dictionary.keys():
            if is_SI:
                labels.append(str(key) + ' [' + parameter_set[key]._SI_units + ']')
            else:
                labels.append(str(key) + ' [' + parameter_set[key].defined_units + ']')
        doeX = pandas.DataFrame(doeX, columns=labels)
        # Try to save .CSV file
        try:
            file_name += '.csv'
            doeX.to_csv(file_name, sep=';', index=False)
            print('\n' + file_name + ' file created with success...')
        except:
            print(file_name + ' file not created, check file_name syntax')
    else:
        if not(isinstance(doeX,  numpy.ndarray)):
            raise TypeError('data should be numpy array')
        elif not(isinstance(file_name, str)):
            raise TypeError('file_name should be a string')
        elif not(isinstance(parameter_set, PositiveParameterSet)):
            raise TypeError('parameter_set should be a PositiveParameterSet')
        else:
            raise TypeError('is_SI should be boolean')
 
#-------[Define function to save doe Dataframe]--------------------------------
def perform_regression(doePI, models, choosen_model, **kwargs):
    """Function to perform regresion using models expression form (with replaced coefficients).
        
        Parameters
        ----------
        doePI: numpy.array 
               DOE of the pi_set
        
        models: specific 
                Output of :func:`~sizinglab.addon.variablepowerlaw`.
        
        choosen_model: int
                       The elected regression model number
        
        **kwargs: additional argumens 
                  * **pi_list** (*list* of *str*): the name/expression of pi (default is pi1, pi2, pi3...)
                  * **latex** (*bool*): define if graph legend font should be latex (default is False) - may cause some issues if used
        
        Example
        -------
        to define regression models refer to: :func:`~sizinglab.addon.variablepowerlaw.regression_models`
        
        then perform regression on model n°8 to show detailed results on model fit and error:
                >>> In[24]: perform_regression(doePI, models, choosen_model=8)
                
                .. image:: ../source/_static/Pictures/variablepowerlaw_perform_regression1.png
        
    """ 
    if isinstance(doePI, numpy.ndarray) and isinstance(choosen_model, int):
        test_mode = False
        pi_list=[]
        latex = False
        for i in range(numpy.shape(doePI)[1]):
                pi_list.append('\pi_{' + str(i + 1) + '}')
        for key, value in kwargs.items():
            if not(key in ['pi_list','test_mode','latex']):
                raise KeyError('unknown argument ' + key)
            elif key == 'pi_list':
                if isinstance(value, list):
                    if len(value) != numpy.shape(doePI)[1]:
                       raise ValueError('defined pi_list mismatch doePI size.')
                    else:
                        for pi_name in value:
                            if not(isinstance(pi_name,str)):
                                raise ValueError('pi_list should be a list of string.')
                        pi_list = value
                else:
                    raise TypeError('pi_list should be a list of string.')
            elif key == 'test_mode':
                if isinstance(value, bool):
                    test_mode = value
                else:
                    raise TypeError('test_mode should be a boolean.')
            elif key == 'latex':
                if isinstance(value, bool):
                    latex = value
                else:
                    raise TypeError('latex should be a boolean.')
        # Check that choosen model is available
        if choosen_model <= 0:
            raise ValueError('choosen_model should be >=1.')
        max_value = 0
        for key in models.keys():
            try:
                max_value = max(max_value, int(key))
            except:
                break
        if choosen_model > max_value:
            raise KeyError('choosen_model should be <={}.'.format(max_value))
        # Print alternative model expression and error repartition
        if not(test_mode):
            print('Elected model for regression is n°{}:'.format(choosen_model))
        expression = str(models[choosen_model][0])
        try:
            expression_latex = concatenate_expression(expression, pi_list)
            if not(test_mode):
                display(Latex(expression_latex))
        except:
            if not(test_mode):
                print(expression + '\n')
        # Save expression
        try:
            hs = open("../_temp/latex_formula.txt","a")
            hs.write(expression_latex + '\n')
            hs.close()
        except:
            pass
        # Adapt expression for calculation
        elected_pi0 = expression[0:expression.find('=')]
        elected_pi0 = elected_pi0.replace('log(','')
        elected_pi0 = elected_pi0.replace(')','')
        elected_pi0 = elected_pi0.replace(' ','')
        # Disable warnings
        logging.captureWarnings(True)
        # Set latex render on plot
        if latex:
            plot.rc('text', usetex=True)
            plot.rc('font', family='serif')
        # Plot regression values in pi0 vs. f(pi1, pi2,...) graph with y=x reference and error repartition histogram
        idx = 0
        for coeff in models[choosen_model][1]:
            idx = expression.find('{:.5f}'.format(coeff), idx)
            expression = expression[0:idx] + str(coeff) + expression[idx + len('{:.5f}'.format(coeff)) - 1:len(expression)]
            idx = idx + len('{:.5f}'.format(coeff)) - 1
        expression1 = expression[expression.find('=') + 1:len(expression)]
        expression2 = expression[0:expression.find('=')]
        log_space = True if expression2.find('log') != -1 else False
        expression1 = expression1.replace('log','numpy.log10')
        expression2 = expression2.replace('log','numpy.log10')
        fig, axs = plot.subplots(1, 2, tight_layout=True)
        for idx in range(numpy.shape(doePI)[1]):
            expression1 = expression1.replace('pi' + str(idx + 1), 'doePI[:,{}]'.format(idx))
            expression2 = expression2.replace('pi' + str(idx + 1), 'doePI[:,{}]'.format(idx))
        try:
            Y_reg = 10**eval(expression1) if log_space else eval(expression1)
            Y = 10**eval(expression2) if log_space else eval(expression2)
            # Adapt Y_reg vector if only constant is considered 
            if isinstance(Y_reg, float):
                Y_reg = Y_reg * numpy.ones(numpy.shape(Y)).astype(float)
        except:
            raise ValueError('possibly doePI and model expression mismatch on pi number.')
        xmin = min(numpy.amin(Y), numpy.amin(Y_reg))
        xmax = max(numpy.amax(Y), numpy.amax(Y_reg))
        axs[0].plot([xmin, xmax], [xmin, xmax],  'b-')
        axs[0].plot(Y, Y_reg, 'r.')
        axs[0].axis([xmin, xmax, xmin, xmax])
        axs[0].grid(True)
        axs[0].set_title('Regression model',fontsize=18)
        elected_pi0 = int(elected_pi0.replace('pi',''))-1
        axs[0].set_xlabel('$' + pi_list[elected_pi0] + '$',fontsize=16)
        y_label = '$' + pi_list[elected_pi0] + ' \simeq f('
        for i in range(len(pi_list)):
            if i != elected_pi0:
                y_label += pi_list[i] + ','
        y_label = y_label[0:len(y_label)-1] + ')$'
        axs[0].set_ylabel(y_label,fontsize=18)
        error = ((numpy.array(Y_reg) - numpy.array(Y))*(1/numpy.array(Y))*100).tolist()
        n_bins = max(1, int(len(error)/5))
        N, bins, patches = axs[1].hist(error, bins=n_bins)
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = plot.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)
        axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=len(error)))
        axs[1].grid(True)
        axs[1].set_title(r'$\epsilon$ repartition',fontsize=18)
        if latex:
            expression = '$\overline{\epsilon}$='+'{:.1f}\% '.format(numpy.mean(error))+'$\sigma_{\epsilon}$='+'{:.1f}\%'.format(numpy.std(error))
        else:
            expression = '$\overline{\epsilon}$='+'{:.1f}% '.format(numpy.mean(error))+'$\sigma_{\epsilon}$='+'{:.1f}%'.format(numpy.std(error))
        axs[1].set_xlabel(expression,fontsize=16)
        axs[1].set_ylabel(r'Probability',fontsize=18)
        axs[1].set_xlim([-3*numpy.std(error)+numpy.mean(error), 3*numpy.std(error)+numpy.mean(error)])
        try:
            plot.savefig('../_temp/regression_models_fig2.pdf', dpi=1200, format='pdf')
        except:
            pass
        if not(test_mode):
            plot.show()
        # De-activate latex render on plot
        plot.rc('text', usetex=False)
        plot.rc('font', family='sans-serif')
        # Unable warnings
        logging.captureWarnings(False)
    else:
        if not(isinstance(doePI, numpy.ndarray)):
            raise TypeError('doePI should be numpy array')
        else:
            raise TypeError('choosen_model should be an integer')

#-------[Define function to PI sensitivity to design drivers]------------------
def pi_sensitivity(pi_set, doePI, useWidgets, **kwargs):
    """Function to perform sensitivity analysis on dimensionless parameters according to specific performance.
        
        Parameters
        ----------           
        pi_set: PositiveParameterSet 
                Set of dimensionless parameters
                 
        doePI: numpy.array 
               DOE of the complete pi_set (except pi0)
        
        useWidgets: bool
                    Boolean to choose if widgets displayed (set to True within Jupyther Notebook)
        
        **kwargs: additional argumens 
                  * **pi0** (*list* of *str*): name of the different pi0 = f(pi...) considered as design drivers
                  * **piN** (*list* of *str*): name of the f(pi1, pi2, ..., piN) considered as secondary parameters
                  * **latex** (*bool*): display in latex format
                  * **figwidth** (*int*): change figure width (default is 16 in widgets mode)
                  * **zero_ymin** (*bool*): set y-axis minimum value to 0 (default is False)
                  * **xlabel_size** (*int*): set x-axis label font size (default is 18)
               
        Example
        -------
        to load a doe example:
                >>> In [1]: doePI = pandas.read_excel('./pi_analysis_example.xls')
                >>> In [2]: doePI = doe[['pj','pfe','pi2','pi3','pi4','pi5','pi6']].values
                >>> In [3]: pi1 = PositiveParameter('pi1',[0.1,1],'','p_j')
                >>> In [4]: pi2 = PositiveParameter('pi2',[0.1,1],'','p_fe')
                >>> In [5]: pi3 = PositiveParameter('pi3',[0.1,1],'','d_i*d_e**-1')
                >>> In [6]: pi4 = PositiveParameter('pi4',[0.1,1],'','e_tooth*d_e**-1*n')
                >>> In [7]: pi5 = PositiveParameter('pi5',[0.1,1],'','e_yoke*d_e**-1*n')
                >>> In [8]: pi6 = PositiveParameter('pi6',[0.1,1],'','w_pm*d_e**-1')
                >>> In [9]: pi7 = PositiveParameter('pi7',[0.1,1],'','r_i*d_e**-1')
                >>> In [10]: pi_set = PositiveParameterSet(pi1, pi2, pi3, pi4, pi5, pi6, pi7)
        
        then perform sensitivity analysis:
                >>> In [11]: pi_sensitivity(pi_set, doePI, False, pi0=['pi1', 'pi2'], piN=['pi3', 'pi4', 'pi5', 'pi6', 'pi7'])
                
                .. image:: ../source/_static/Pictures/variablepowerlaw_pi_sensitivity.png
                
        Note
        ----
        Within Jupyter Notebook, rendering will be slightly different with compressed size in X-axis to be printed within one page width and labels adapted consequently.
        
    """ 
    if isinstance(useWidgets,bool):
        test_mode = False
        for key, value in kwargs.items():
            if key == 'test_mode':
                if isinstance(value,bool):
                    test_mode = value
                else:
                    raise TypeError('test_mode should be boolean')
        if useWidgets:
            pi_list = []
            for pi in pi_set.dictionary.keys():
                pi_list.append(pi.replace('pi','$\pi_{') + '}$')
            pi0_list = [list(pi_set.dictionary.keys())[0]]
            piN_list = list(pi_set.dictionary.keys())[1:len(list(pi_set.dictionary.keys()))]
            axes, plot, _, _ = pi_sensitivity_sub(pi_set, doePI, pi0=pi0_list, piN=piN_list, figwidth=16)
            checkboxes1 = []
            checkboxes2 = []
            cb_container1 = widgets.HBox()
            cb_container2 = widgets.HBox()
            container3 = widgets.HBox()
            for pi_parameter in range(numpy.shape(doePI)[1]):
                if pi_parameter == 0:
                    checkboxes1.append(widgets.Checkbox(description = pi_list[pi_parameter], value=True, width=90))
                    checkboxes2.append(widgets.Checkbox(description = pi_list[pi_parameter], value=False, width=90))
                else:
                    checkboxes1.append(widgets.Checkbox(description = pi_list[pi_parameter], value=False, width=90))
                    checkboxes2.append(widgets.Checkbox(description = pi_list[pi_parameter], value=True, width=90))
            cb_container1.children=[i for i in checkboxes1]
            cb_container2.children=[i for i in checkboxes2]
            label_slider = widgets.FloatSlider(value=18, min=5, max=24, step=1,  description='X-label FontSize:')
            zero_y = widgets.Checkbox(description = 'y-axis intersect 0', value=False, width=90)
            container3.children = [label_slider, zero_y]
            tab = widgets.Tab(children=[cb_container1, cb_container2, container3])
            button = widgets.Button(description="Apply")
            tab.set_title(0, 'Primary PI')
            tab.set_title(1, 'Secondary PI')
            tab.set_title(2, 'Graph parameters')
            menu = VBox(children=[tab, button])
            display(menu)
            if not(test_mode):
                plot.show()
            reference_list = list(pi_set.dictionary.keys())
            def on_button_clicked(b):
                clear_output()
                display(menu)
                idx = 0
                pi0_list = []
                for c in cb_container1.children:
                    if c.value:
                        pi0_list.append(reference_list[idx])
                    idx += 1
                idx = 0
                piN_list = []
                for c in cb_container2.children:
                    if c.value:
                        piN_list.append(reference_list[idx])
                    idx += 1
                for c in container3.children:
                    if isinstance(c.value,bool):
                        zero_ymin_value = c.value
                    else:
                        fontsize = int(c.value)
                axes, plot, _, _ = pi_sensitivity_sub(pi_set, doePI, pi0=pi0_list, piN=piN_list, figwidth=16, zero_ymin=zero_ymin_value, xlabel_size=fontsize)
                if not(test_mode):
                    plot.show()
            button.on_click(on_button_clicked)
        else:
            _, plot, _, _ = pi_sensitivity_sub(pi_set, doePI, **kwargs)
            try:
                plot.savefig('../_temp/pi_sensitivity.pdf', dpi=1200, format='pdf')
            except:
                pass
            if not(test_mode):
                plot.show()
                print('MCC - Maximum Correlation Coefficient between Pearson and Spearman')
                print('alpha - Relative standard deviation (on dimensionless parameter)')
                print('IF - Impact factor IF=MCC*alpha')
            # De-activate latex render on plot
            plot.rc('text', usetex=False)
            plot.rc('font', family='sans-serif')
    else:
        raise TypeError('useWidgets should be a boolean')
    
def pi_sensitivity_sub(pi_set, doePI, **kwargs):
    """Sub-function of :func:`~sizinglab.addon.variablepowerlaw.pi_sensitivity`
    """
    if isinstance(pi_set, PositiveParameterSet) and isinstance(doePI, numpy.ndarray):
        # Check data and define default when widgets option chosen
        if numpy.shape(doePI)[1] != len(list(pi_set.dictionary.keys())):
            raise ValueError('doePI and pi_set dimensions mismatch')
        pi_list = list(pi_set.dictionary.keys())
        latex = False
        figwidth = float('Inf')
        x_index = list(range(1, len(pi_list)))
        y_index = [0]
        zero_ymin = False
        xlabel_size = 18
        for key, value in kwargs.items():
            if not(key in ['pi0','piN', 'latex', 'figwidth', 'zero_ymin', 'xlabel_size','test_mode']):
                raise KeyError('unknown argument ' + key)
            elif key == 'test_mode':
                pass
            elif key == 'pi0':
                if isinstance(value,str):
                    try:
                        y_index = pi_list.index(value)
                    except:
                         raise ValueError('pi0 not in pi_set')
                elif isinstance(value, list):
                    y_index = []
                    for pi0_value in value:
                        try:
                            y_index.append(pi_list.index(pi0_value))
                        except:
                            raise ValueError('some pi0 values not in pi_set')
                else:
                    raise TypeError('pi0 should be a string or a list of string')
            elif key == 'piN':
                if isinstance(value,str):
                    try:
                        x_index = pi_list.index(value)
                    except:
                         raise ValueError('piN not in pi_set')
                elif isinstance(value, list):
                    x_index = []
                    for piN_value in value:
                        try:
                            x_index.append(pi_list.index(piN_value))
                        except:
                            raise ValueError('some piN values not in pi_set')
                else:
                    raise TypeError('piN should be a string or a list of string')
            elif key == 'figwidth':
                if isinstance(value,float) or isinstance(value,int):
                    figwidth = float(value)
                else:
                    raise ValueError('figwidth should be float or int')
            elif key == 'zero_ymin':
                if isinstance(value,bool):
                    zero_ymin = value
                else:
                    raise ValueError('zero_ymin should be boolean')
            elif key == 'xlabel_size':
                if isinstance(value,int):
                    if (value<7) or (value >24):
                        raise ValueError('xlabel_size should be in [7, 24]')
                    xlabel_size = value
                else:
                    raise ValueError('xlabel_size should be integer')
            else:
                if isinstance(value,bool):
                    latex = value
                else:
                    raise TypeError('latex should be boolean')
        # Calculate alpha coefficient: standard_deviation/average_value
        alpha = []
        for x_i in range(len(x_index)):
            alpha.append(numpy.std(doePI[:, x_index[x_i]], ddof=1)/numpy.mean(doePI[:, x_index[x_i]]))
        # Calculate correlation coefficient correl in [-1;+1]
        pearson_correl_matrix = numpy.zeros((len(y_index),len(x_index)))
        spearman_correl_matrix = numpy.zeros((len(y_index),len(x_index)))
        impact_matrix = numpy.zeros((len(y_index),len(x_index)))
        for y_i in range(len(y_index)):
            for x_i in range(len(x_index)):                
                pearson_correl_matrix[y_i, x_i] = scipy.stats.pearsonr(doePI[:, y_index[y_i]], doePI[:, x_index[x_i]])[0]
                spearman_correl_matrix[y_i, x_i] = scipy.stats.spearmanr(doePI[:, y_index[y_i]], doePI[:, x_index[x_i]])[0]
                max_correl_coeff = numpy.array([pearson_correl_matrix[y_i, x_i], spearman_correl_matrix[y_i, x_i]])
                max_correl_coeff = max_correl_coeff[numpy.argmax(numpy.absolute(max_correl_coeff))]
                impact_matrix[y_i, x_i] = max_correl_coeff * alpha[x_i]
        # Construct labels and pi_axis limits
        latex_pi_list, problem = latex_pi_expression(pi_set, [])
        pi_axis = [float('Inf'), -float('Inf')]
        for x_i in x_index:
            pi_axis[0] = min(pi_axis[0], numpy.amin(doePI[:, x_i]/numpy.mean(doePI[:, x_i])))
            pi_axis[1] = max(pi_axis[1], numpy.amax(doePI[:, x_i]/numpy.mean(doePI[:, x_i])))
        # Set latex render on plot
        if latex:
            plot.rc('text', usetex=True)
            plot.rc('font', family='serif')
        # Plot graphs
        fig, axes = plot.subplots(nrows=len(y_index), ncols=len(x_index), sharex=False, sharey=False)
        fig.set_size_inches(min(figwidth, 3*len(x_index)), 3*len(y_index))
        for y_i in range(len(y_index)):
            for x_i in range(len(x_index)):
                if len(y_index)==1:
                    axis_name = x_i
                elif len(x_index)==1:
                    axis_name = y_i
                else:
                    axis_name = (y_i, x_i)
                axes[axis_name].plot(doePI[:, x_index[x_i]]/numpy.mean(doePI[:, x_index[x_i]]), doePI[:, y_index[y_i]], 'ob')
                axes[axis_name].set_xlim(pi_axis)
                if zero_ymin:
                    axes[axis_name].set_ylim([0, numpy.amax(doePI[:, y_index[y_i]])])
                else:
                    axes[axis_name].set_ylim([numpy.amin(doePI[:, y_index[y_i]]), numpy.amax(doePI[:, y_index[y_i]])])
                if x_i == 0:
                    axes[axis_name].set_ylabel(latex_pi_list[y_index[y_i]], fontsize=18)
                else:
                    axes[axis_name].set_yticklabels([])
                if y_i == y_index[-1]:
                    expression = '$\\frac{' + pi_list[x_index[x_i]].replace('pi','\pi_{') + '}}{\\overline{' + pi_list[x_index[x_i]].replace('pi','\pi_{') + '}}}'
                    if not(problem):
                        expression += ' \\mid ' + latex_pi_list[x_index[x_i]].replace('$','') + '$'
                    axes[axis_name].set_xlabel(expression, fontsize=xlabel_size)
                else:
                    axes[axis_name].set_xticklabels([])
                axes[axis_name].annotate('MCC={:.2f}'.format(impact_matrix[y_i, x_i]/alpha[x_i]), (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
                axes[axis_name].annotate('$\\alpha$={:.2f}'.format(alpha[x_i]), (0.8, 0.7), xycoords='axes fraction', ha='center', va='center')
                if latex:
                    axes[axis_name].annotate('IF={:.0f}\%'.format(impact_matrix[y_i, x_i]*100), (0.8, 0.6), xycoords='axes fraction', ha='center', va='center')
                else:
                    axes[axis_name].annotate('IF={:.0f}%'.format(impact_matrix[y_i, x_i]*100), (0.8, 0.6), xycoords='axes fraction', ha='center', va='center')
                axes[axis_name].grid(False)
        # Color figure with decreasing order of impact
        max_IF = numpy.amax(numpy.absolute(impact_matrix))
        min_IF = numpy.amin(numpy.absolute(impact_matrix))
        for x_i in range(len(x_index)):
            for y_i in range(len(y_index)):
                current_IF =numpy.absolute(impact_matrix[y_i, x_i])
                blue = max(0, min(1, 1 - (current_IF-min_IF)/(max_IF-min_IF)))
                if len(y_index)==1:
                    axis_name = x_i
                elif len(x_index)==1:
                    axis_name = y_i
                else:
                    axis_name = (y_i, x_i)
                axes[axis_name].set_facecolor((1, 1, blue))
        return axes, plot, impact_matrix, latex_pi_list
    else:
        if not(isinstance(pi_set, PositiveParameterSet)):
            raise TypeError('pi_set should be PositiveParameterSet')
        elif not(isinstance(doePI, numpy.ndarray)):
            raise TypeError('doePI should be numpy array')
        else:
            raise TypeError('useWidgets should be a boolean')
 
#-------[Define function to return PI expression in latex format]--------------           
def latex_pi_expression(pi_set, parameter_set):
    """Function to write pi description in latex form: ***internal*** to :func:`~sizinglab.addon.variablepowerlaw.pi_sensitivity` and :func:`~sizinglab.addon.variablepowerlaw.pi_dependency`
    """
    greek_list = ['alpha','beta','gamma','delta','epsilon','varepsilon','zeta','eta','theta','vartheta','gamma','kappa','lambda',\
                      'mu','nu','xi','pi','varpi','rho','varrho','sigma','varsigma','tau','upsilon','phi','varphi','chi','psi','omega']
    latex_pi_list = []
    try:
        parameter_list = list(parameter_set.dictionary.keys())
    except:
        parameter_list = []
    problem = False
    for pi_name in pi_set.dictionary.keys():
        expression = pi_set[pi_name].description
        expression_list_init = expression.replace('**','^')
        expression_list_init = expression_list_init.split('*')
        expression_list = []
        for expression in expression_list_init:
            expression_list.extend(expression.split('^'))
        for expression in expression_list:
            try:
                float(expression)
            except:
                if len(parameter_list) != 0:
                    if not(expression in parameter_list):
                        problem = True
                        break
        if problem:
            break
        else:
            upper_expression_list = []
            lower_expression_list = []
            for expression in expression_list_init:
                if len(expression.split('^')) == 1:
                    exponent = 1
                else:
                    exponent = float(expression.split('^')[1])
                    expression = expression.split('^')[0]
                if len(expression.replace('_','')) == (len(expression) - 1):
                    terms = expression.split('_')
                    if terms[0].lower() in greek_list:
                        expression = '\\' + terms[0]
                        if terms[1].lower() in greek_list:
                            expression += '_{\\' + terms[1] + '}'
                        else:
                            expression += '_{' + terms[1].upper() + '}'
                    else:
                        expression = terms[0].upper()
                        if terms[1].lower() in greek_list:
                            expression += '_{\\' + terms[1] + '}'
                        else:
                            expression += '_{' + terms[1].upper() + '}'
                else:
                    expression = expression.replace('_','')
                    expression = expression.upper()
                if abs(exponent) != 1.0:
                    expression += '^{' + str(abs(exponent)) + '}'
                if exponent > 0:
                    upper_expression_list.append(expression)
                else:
                    lower_expression_list.append(expression)
            if len(lower_expression_list) != 0:
                expression = '$' + pi_name.replace('pi','\pi_{') + '} = \\frac{'
                if len(upper_expression_list) == 0:
                    expression += '1}{'
            else:
                expression = '$' + pi_name.replace('pi','\pi_{') + '} = '
            for idx in range(len(upper_expression_list)):
                if idx == 0:
                    expression += upper_expression_list[idx]
                else:
                    expression += ' \\cdot ' + upper_expression_list[idx]
            if len(lower_expression_list) != 0:
                expression += '}{'
                for idx in range(len(lower_expression_list)):
                    if idx == 0:
                        expression += lower_expression_list[idx]
                    else:
                        expression += ' \\cdot ' + lower_expression_list[idx]
                expression += '}$'
            else:
                expression += '$'
        latex_pi_list.append(expression)
    if problem:
        latex_pi_list = []
        for pi_name in pi_set.dictionary.keys():
            latex_pi_list.append('$' + pi_name.replace('pi','\pi_{') + '}$')
    return latex_pi_list, problem

#-------[Define function to PI sensitivity to design drivers]------------------
def pi_dependency(pi_set, doePI, useWidgets, **kwargs):
    """Function to perform dependency analysis on dimensionless parameters.
        
        Parameters
        ----------
        pi_set: PositiveParameterSet 
                Set of dimensionless parameters
                 
        doePI: numpy.array 
               DOE of the complete pi_set (except pi0)
        
        useWidgets: bool
                    Boolean to choose if widgets displayed (set to True within Jupyther Notebook)
        
        **kwargs: additional argumens 
                  * **x_list** (*list* of *str*): name of the different pi1, pi2,... to be defined as x-axis (default is all)
                  * **y_list** (*list* of *str*): name of the different pi1, pi2,... to be defined as y-axis (default is all)
                  * **order** (*int*): the order choosen for power-law or polynomial regression model (default is 2)
                  * **threshold** (*float*): in ]0,1[ the lower limit to consider regression for plot  (default is 0.9)
                  * **figwidth** (*int*): change figure width (default is 16 in widgets mode)
                  * **xlabel_size** (*int*): set x-axis label font size (default is 16)
                  
        Example
        -------
        to load a doe example: see :func:`~sizinglab.addon.variablepowerlaw.pi_sensitivity`
        
        then perform dependency analysis:
                >>> In [11]: pi_dependency(pi_set, doePI, useWidgets=False)
                
                .. image:: ../source/_static/Pictures/variablepowerlaw_pi_dependency.png
        
    """
    if isinstance(useWidgets,bool):
        test_mode = False
        for key, value in kwargs.items():
            if key == 'test_mode':
                if isinstance(value,bool):
                    test_mode = value
                else:
                    raise TypeError('test_mode should be boolean')            
        if useWidgets:
            pi_list = []
            for pi in pi_set.dictionary.keys():
                pi_list.append(pi.replace('pi','$\pi_{') + '}$')
            x_list = pi_list
            y_list = pi_list
            _, _, _, plot = pi_dependency_sub(pi_set, doePI, order=2, threshold=0.9, figwidth=16)
            checkboxes1 = []
            checkboxes2 = []
            cb_container1 = widgets.HBox()
            cb_container2 = widgets.HBox()
            container3 = widgets.HBox()
            for pi_parameter in range(numpy.shape(doePI)[1]):
                checkboxes1.append(widgets.Checkbox(description = x_list[pi_parameter], value=True, width=90))
                checkboxes2.append(widgets.Checkbox(description = y_list[pi_parameter], value=True, width=90))
            cb_container1.children=[i for i in checkboxes1]
            cb_container2.children=[i for i in checkboxes2]
            label_slider = widgets.FloatSlider(value=18, min=5, max=24, step=1,  description='X-label FontSize:')
            order_slider = widgets.FloatSlider(value=2, min=1, max=4, step=1,  description='Order:')
            threshold_slider = widgets.FloatSlider(value=0.9, min=0.0, max=1.0, step=0.01,  description='Thresold:')
            container3.children = [label_slider, order_slider, threshold_slider]
            tab = widgets.Tab(children=[cb_container1, cb_container2, container3])
            button = widgets.Button(description="Apply")
            tab.set_title(0, 'Primary PI')
            tab.set_title(1, 'Secondary PI')
            tab.set_title(2, 'Graph parameters')
            menu = VBox(children=[tab, button])
            display(menu)
            if not(test_mode):
                plot.show()
            reference_list = list(pi_set.dictionary.keys())
            def on_button_clicked(b):
                clear_output()
                display(menu)
                idx = 0
                xlist = []
                for c in cb_container1.children:
                    if c.value:
                        xlist.append(reference_list[idx])
                    idx += 1
                idx = 0
                ylist = []
                for c in cb_container2.children:
                    if c.value:
                        ylist.append(reference_list[idx])
                    idx += 1
                idx = 0
                for c in container3.children:
                    if idx == 0:
                        fontsize = int(c.value)
                    elif idx == 1:
                        order_value = int(c.value)
                    else:
                        threshold_value = float(c.value)
                    idx += 1
                _, _, _, plot = pi_dependency_sub(pi_set, doePI, x_list=xlist, y_list=ylist, order=order_value, threshold=threshold_value, figwidth=16, xlabel_size=fontsize)
                if not(test_mode):
                    plot.show()
            button.on_click(on_button_clicked)
        else:
            _, _, _, plot = pi_dependency_sub(pi_set, doePI, **kwargs)
            try:
                plot.savefig('../_temp/pi_dependency.pdf', dpi=1200, format='pdf')
            except:
                pass
            if not(test_mode):
                plot.show()
    else:
        raise TypeError('useWidgets should be a boolean')
    
def pi_dependency_sub(pi_set, doePI, **kwargs):
    """Sub-function of :func:`~sizinglab.addon.variablepowerlaw.pi_dependency`
    """
    if isinstance(pi_set, PositiveParameterSet) and isinstance(doePI, numpy.ndarray):
        # Check data and define default when widgets option chosen
        if numpy.shape(doePI)[1] != len(list(pi_set.dictionary.keys())):
            raise ValueError('doePI and pi_set dimensions mismatch')
        if numpy.amin(doePI) <= 0:
            raise ValueError('doePI values should be strictly positive')
        x_list = list(pi_set.dictionary.keys())
        y_list = list(pi_set.dictionary.keys())
        pi_list = list(pi_set.dictionary.keys())
        order = 2
        threshold = 0.9
        figwidth = float('Inf')
        xlabel_size = 16
        for key, value in kwargs.items():
            if not(key in ['x_list','y_list','order', 'threshold', 'figwidth','xlabel_size','test_mode']):
                raise KeyError('unknown argument ' + key)
            elif key == 'test_mode':
                pass
            elif key == 'x_list':
                if isinstance(value, list):
                   for pi_name in value:
                       if not(pi_name in x_list):
                           raise KeyError('defined list key {} not in pi_set'.format(pi_name))
                   x_list = list(set(value))
                else:
                    raise TypeError('pi_list should be a list of string')
            elif key == 'y_list':
                if isinstance(value, list):
                   for pi_name in value:
                       if not(pi_name in y_list):
                           raise KeyError('defined list key {} not in pi_set'.format(pi_name))
                   y_list = list(set(value))
                else:
                    raise TypeError('pi_list should be a list of string')
            elif key == 'order':
                if isinstance(value, int):
                    if (value <=0) or (value >= numpy.shape(doePI)[0]):
                        ValueError('order should be >0 and <{}'.format(numpy.shape(doePI)[0]))
                    else:
                        order = value
                else:
                    raise TypeError('order should be a float')
            elif key == 'threshold':
                if isinstance(value, float):
                    if (value <= 0) or (value > 1):
                        raise ValueError('threshold should be in ]0,1[')
                    else:
                        threshold = value
                else:
                    raise ValueError('threshold should be float')
            elif key == 'figwidth':
                if isinstance(value,float) or isinstance(value,int):
                    figwidth = float(value)
                else:
                    raise ValueError('figwidth should be float or int')
            elif key == 'xlabel_size':
                if isinstance(value,int):
                    if (value<7) or (value >24):
                        raise ValueError('xlabel_size should be in [7, 24]')
                    xlabel_size = value
                else:
                    raise ValueError('xlabel_size should be integer')
        xy_identical = True
        if len(x_list) == len(y_list):
            for element in x_list:
                if not(element in y_list):
                    xy_identical = False
        else:
            xy_identical = False
        # Plot scatter matrix
        r2_matrix = numpy.zeros((len(y_list),len(x_list)))
        regType_matrix = numpy.zeros((len(y_list),len(x_list))).astype(str)
        coeff_matrix = numpy.zeros((len(y_list),len(x_list)),dtype=(float,order + 1))
        fig, axes = plot.subplots(nrows=len(y_list), ncols=len(x_list), sharex=False, sharey=False)
        fig.set_size_inches(min(figwidth,3*len(x_list)), 3*len(y_list))
        # Write pi_list and x-y index
        xaxis_index = []
        latex_pi_list, problem = latex_pi_expression(pi_set, [])
        for idx in range(len(pi_list)):
            pi_list[idx] = pi_list[idx].replace('pi','$\\pi_{') + '}$'
        for idx in range(len(x_list)):
            xaxis_index.append(int(x_list[idx].replace('pi','')) - 1)
        yaxis_index = []
        for idx in range(len(y_list)):
            yaxis_index.append(int(y_list[idx].replace('pi','')) - 1)
        for yaxis_i in range(len(yaxis_index)):
            for xaxis_i in range(len(xaxis_index)):
                xaxis_value = xaxis_index[xaxis_i]
                yaxis_value = yaxis_index[yaxis_i]
                x = doePI[:, xaxis_value]
                y = doePI[:, yaxis_value]
                # Try to fit order x power form
                x_log = numpy.log10(x)
                x_log = x_log.reshape(-1, 1)
                y_log = numpy.log10(y)
                poly_feature = PolynomialFeatures(degree=order, include_bias = True)
                X = poly_feature.fit_transform(x_log)
                coeff, _, _, _ = scipy.linalg.lstsq(X, y_log)
                coeff_matrix[yaxis_i, xaxis_i] = tuple(coeff)
                y_pred = 10**numpy.dot(X, coeff)
                r2_log = 1 - numpy.sum((y-y_pred)**2) / numpy.sum((y-numpy.mean(y))**2)
                x_pred = numpy.linspace(numpy.amin(doePI[:, xaxis_value]), numpy.amax(doePI[:, xaxis_value]), 20)
                y_pred = 10**numpy.dot(poly_feature.fit_transform(numpy.log10(x_pred).reshape(-1, 1)), coeff)
                # Construct relation expression y/10^a0 = x^(a1+a2*log(x)+a3*log(x)^2+.....)
                if coeff[0]/10 >= 1:
                    current_coeff = '{:.0f}'.format(coeff[0])
                elif coeff[0] >= 1:
                    current_coeff = '{:.1f}'.format(coeff[0])
                else:
                    current_coeff = '{:.2f}'.format(coeff[0])
                expression = '$\\frac{' + pi_list[yaxis_value].replace('$','') + '}{10^{' + current_coeff + '}}= ' + pi_list[xaxis_value].replace('$','') + '^{'
                for idx in range(len(coeff)-1):
                    if coeff[idx + 1]/10 >= 1:
                        current_coeff = '{:.0f}'.format(coeff[idx + 1])
                    elif coeff[idx + 1] >= 1:
                        current_coeff = '{:.1f}'.format(coeff[idx + 1])
                    else:
                        current_coeff = '{:.2f}'.format(coeff[idx + 1])
                    if idx == 0:
                        expression += current_coeff
                    elif idx == 1:
                        if coeff[idx + 1] < 0:
                            expression += current_coeff + ' \cdot log(' + pi_list[xaxis_value].replace('$','') + ') '
                        else:
                            expression += '+' + current_coeff + ' \cdot log(' + pi_list[xaxis_value].replace('$','') + ') '
                    else:
                        if coeff[idx + 1] < 0:
                            expression += current_coeff + ' \cdot log(' + pi_list[xaxis_value].replace('$','') + ')^{' + str(idx) + '} '
                        else:
                            expression += '+' + current_coeff + ' \cdot log(' + pi_list[xaxis_value].replace('$','') + ')^{' + str(idx) + '} '
                expression += '}$'
                # Try to fit order x polynomial form
                poly_feature = PolynomialFeatures(degree=order, include_bias = True)
                X = poly_feature.fit_transform(x.reshape(-1, 1))
                coeff, _, _, _ = scipy.linalg.lstsq(X, y)
                r2_lin = 1 - numpy.sum((y-numpy.dot(X, coeff))**2) / numpy.sum((y-numpy.mean(y))**2)
                if r2_lin > r2_log:
                    regType_matrix[yaxis_i, xaxis_i] = 'lin'
                    r2_matrix[yaxis_i, xaxis_i] = r2_lin
                    x_pred = numpy.linspace(numpy.amin(doePI[:, xaxis_value]), numpy.amax(doePI[:, xaxis_value]), 20)
                    y_pred = numpy.dot(poly_feature.fit_transform(x_pred.reshape(-1, 1)), coeff)
                    # Construct relation expression y = a0 + a1*x + a2*x^2 + a3*x^3....
                    if coeff[0]/10 >= 1:
                        current_coeff = '{:.0f}'.format(coeff[0])
                    elif coeff[0] >= 1:
                        current_coeff = '{:.1f}'.format(coeff[0])
                    else:
                        current_coeff = '{:.2f}'.format(coeff[0])
                    expression = '$' + pi_list[yaxis_value].replace('$','') + '=' + current_coeff
                    for idx in range(len(coeff)-1):
                        if coeff[idx + 1]/10 >= 1:
                            current_coeff = '{:.0f}'.format(coeff[idx + 1])
                        elif coeff[idx + 1] >= 1:
                            current_coeff = '{:.1f}'.format(coeff[idx + 1])
                        else:
                            current_coeff = '{:.2f}'.format(coeff[idx + 1])
                        if idx == 0:
                            if coeff[idx + 1] < 0:
                                expression += current_coeff + ' \cdot ' + pi_list[xaxis_value].replace('$','')
                            else:
                                expression += '+' + current_coeff  + ' \cdot ' + pi_list[xaxis_value].replace('$','')
                        else:
                            if coeff[idx + 1] < 0:
                                expression += current_coeff + ' \cdot ' + pi_list[xaxis_value].replace('$','') + '^{' + str(idx+1) + '} '
                            else:
                                expression += '+' + current_coeff + ' \cdot ' + pi_list[xaxis_value].replace('$','') + '^{' + str(idx+1) + '} '
                    expression += '$'
                    label_size = 10
                else:
                    regType_matrix[yaxis_i, xaxis_i] = 'log'
                    r2_matrix[yaxis_i, xaxis_i] = r2_log
                    label_size = 12
                if (yaxis_i == xaxis_i) and xy_identical :
                    axes[yaxis_i, xaxis_i].annotate(latex_pi_list[xaxis_value], (0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=xlabel_size)
                else:
                    # Plot fitting curve and data
                    axes[yaxis_i, xaxis_i].plot(x, y, 'ob')
                    if r2_matrix[yaxis_i, xaxis_i] >= threshold:
                        axes[yaxis_i, xaxis_i].plot(x_pred, y_pred, '-r')
                        axes[yaxis_i, xaxis_i].annotate('$R^2$={:.2f}'.format(r2_matrix[yaxis_i, xaxis_i]), (0.8, 0.6), xycoords='axes fraction', ha='center', va='center')
                        axes[yaxis_i, xaxis_i].annotate(expression, (0.5, 0.8), xycoords='axes fraction', ha='center', va='center', fontsize=label_size)
                # Highlight best model up/down diag in scatter plot (if scatter plot choosen)
                if xy_identical and (yaxis_i > xaxis_i) and (r2_matrix[yaxis_i, xaxis_i] >= threshold or r2_matrix[xaxis_i, yaxis_i] >= threshold):
                    if r2_matrix[yaxis_i, xaxis_i] < r2_matrix[xaxis_i, yaxis_i]:
                        axes[xaxis_i, yaxis_i].set_facecolor('xkcd:light beige')
                    else:
                        axes[yaxis_i, xaxis_i].set_facecolor('xkcd:light beige')
                # Highlight best model in x-y plot (if scatter plot not choosen)
                if not(xy_identical) and xaxis_i == len(xaxis_index)-1:
                    if numpy.amax(r2_matrix[yaxis_i, :]) >= threshold:
                        axes[yaxis_i,numpy.argsort(-1*r2_matrix[yaxis_i, :])[0]].set_facecolor('xkcd:light beige')
                if xaxis_i == 0 and (yaxis_i % 2) == 0:
                    axes[yaxis_i, xaxis_i].set_yticklabels([])
                elif xaxis_i == len(xaxis_index)-1 and (yaxis_i % 2) == 1:
                    axes[yaxis_i, xaxis_i].set_yticklabels([])
                elif xaxis_i == len(xaxis_index)-1 and (yaxis_i % 2) == 0:
                    axes[yaxis_i, xaxis_i].yaxis.set_ticks_position('right')
                elif xaxis_i > 0 and xaxis_i < len(xaxis_index)-1:
                    axes[yaxis_i, xaxis_i].set_yticklabels([])
                if yaxis_i == 0 and (xaxis_i % 2) == 0:
                    axes[yaxis_i, xaxis_i].set_xticklabels([])
                elif yaxis_i == 0 and (xaxis_i % 2) == 1:
                    axes[yaxis_i, xaxis_i].xaxis.set_ticks_position('top')
                elif yaxis_i == len(yaxis_index)-1 and (xaxis_i % 2) == 1:
                    axes[yaxis_i, xaxis_i].set_xticklabels([])
                elif yaxis_i > 0 and yaxis_i < len(yaxis_index)-1:
                    axes[yaxis_i, xaxis_i].set_xticklabels([])
                if not(xy_identical):
                    if xaxis_i == 0:
                        axes[yaxis_i, xaxis_i].set_ylabel(latex_pi_list[yaxis_value],fontsize=16)
                    if yaxis_i == len(yaxis_index)-1:
                        axes[yaxis_i, xaxis_i].set_xlabel(latex_pi_list[xaxis_value],fontsize=xlabel_size)
        return r2_matrix, coeff_matrix, regType_matrix, plot
    elif not(isinstance(doePI, numpy.ndarray)):
        raise TypeError('doePI should be numpy array')
    else:
        raise TypeError('pi_set should be a PositiveParameterSet')
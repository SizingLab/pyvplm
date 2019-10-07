# -*- coding: utf-8 -*-
"""
Specific module to interface variablepowerlaw module with comsol FEM software

"""

#-------[Import necessary packages]--------------------------------------------
import numpy
import warnings
import re
import pandas
import pint
from pyvplm.core.definition import PositiveParameterSet

#-------[Define function to save doe Dataframe]--------------------------------
def save_file(doeX, file_name, parameter_set, is_SI, **kwargs):
    """Function to save .txt file within COMSOL input format.
        Values can be either expressed with SI units or user defined units (is_SI, True by default).
        
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
        test_mode = False
        for key, value in kwargs.items():
            if not(key in ['test_mode']):
                raise KeyError('unknown argument ' + key)
            elif key == 'test_mode':
                if isinstance(value, bool):
                    test_mode = value
                else:
                    raise TypeError('test_mode should be a boolean.')
        # Check that data array and parameter set have same size
        if numpy.shape(doeX)[1] != len(list(parameter_set.dictionary.keys())):
            raise ValueError('data dimension mismatch parameter_set keys\'number')
        # Check that values are in defined bounds with 0.01% relative error flexibility
        key_list = list(parameter_set.dictionary.keys())
        for idx in range(numpy.shape(doeX)[1]):
            max_value = numpy.amax(doeX[:, idx])
            min_value = numpy.amin(doeX[:, idx])
            bounds = parameter_set[key_list[idx]]._SI_bounds if is_SI else parameter_set[key_list[idx]].defined_bounds
            if (min_value < bounds[0]) and (abs(min_value - bounds[0])/bounds[0]) < 0.0001:
                doeX[:, idx] = numpy.maximum(doeX[:, idx], bounds[0])
            elif (max_value > bounds[1]) and (abs(max_value - bounds[1])/bounds[1]) < 0.0001:
                doeX[:, idx] = numpy.minimum(doeX[:, idx], bounds[1])
            elif (min_value < bounds[0]) or (max_value > bounds[1]):
                warnings.warn('for parameter {} saved values are out of bounds with more than 0.01% flexibility!'.format(key_list[idx]))
        # Write each paramater value on a line
        try:
            file = open(file_name + '.txt', 'w')
            if not(test_mode):
                print('\n REMINDER: while importing parameters\'values to COMSOL use following units')
            idx = 0
            for key in parameter_set.dictionary.keys():
                expression = str(key) + ' '
                for value in doeX[:, idx]:
                    expression += str(value) + ','
                file.write(expression[0:len(expression) - 1] + '\n') 
                if not(test_mode):
                    print(key + ':' + parameter_set[key]._SI_units) if is_SI else print(key + ':' + parameter_set[key].defined_units)
                idx += 1
            file.close()
            if not(test_mode):
                print('\n '+ file_name + '.txt file created with success...')
        except:
            raise SyntaxError('file name inapropriate, unable to open file for writing purpose.')
    else:
        if not(isinstance(doeX,  numpy.ndarray)):
            raise TypeError('data should be numpy array')
        elif not(isinstance(file_name, str)):
            raise TypeError('file_name should be a string')
        elif not(isinstance(parameter_set, PositiveParameterSet)):
            raise TypeError('parameter_set should be a PositiveParameterSet')
        else:
            raise TypeError('is_SI should be boolean')
            
def import_file(file_name, parameter_set, units):
    """Function to import .txt file generated by COMSOL (output format).
        Values can be either expressed within SI units, user defined units or specified units in the parameter name : 'parameter_name (units)'.
        
        Parameters
        ----------        
        file_name: str
                   Name of the saved file with path (example: file_name = './subfolder/name')
        
        parameter_set: PositiveParameterSet
                       Defines the n physical parameters for the studied problem
        
        units: str
               Define what units should be considered for parameters from set
               * 'SI': means parameter is expressed within SI units, no adaptation needed and '(units)' in column name is ignored.
               * 'defined': means parameter is expressed within defined units written in parameter and '(units)' in column name is ignored, adaptation may be performed.
               * 'from_file': means parameter is expressed with file units and adaptation may be performed.
               
               If units are unreadable and **from_file** option is choosen, it is considered to be expressed in SI units.
               For parameters not in the defined set, if units are unreadable, there are no units or **SI** option is choosen, it is considered to be SI otherwise adaptation is performed.
        
    """ 
    if isinstance(file_name, str) and isinstance(parameter_set, PositiveParameterSet) and isinstance(units, str):
        # Test units value
        if not(units == 'SI' or units == 'defined' or units == 'from_file'):
            raise ValueError('units should be \'SI\' or \'defined\' or \'from_file\'.')
        # Read file lines
        try:
            doeX = numpy.array([]).astype(float)
            file_name = file_name + '.txt'
            empty_file = True
            labels = []
            with open(file_name, 'r') as file:
                line = file.readline()
                previous_line = line
                while len(line) != 0 :
                    if line[0] != '%':
                        empty_file = False
                        # Extract parameters list
                        if len(labels) == 0:
                            s = previous_line[2:len(previous_line)]
                            labels = eval('[\'' + re.sub("\s+", "\',\'", s.strip()) +'\']')
                            non_units = numpy.ones(len(labels)).astype(bool)
                            for idx in range(len(labels)):
                                parameter = str(labels[idx])
                                if parameter[0] == '(':
                                    if idx !=0:
                                        labels[idx-1] = labels[idx-1] + ' ' + labels[idx]
                                    non_units[idx] = False
                            labels = numpy.array(labels)
                            labels = labels[non_units].tolist()
                        # Extract values
                        s = line
                        values = numpy.array(eval('[' + re.sub("\s+", ",", s.strip()) +']'))
                        if len(doeX) == 0:
                            doeX = values
                        else:
                            doeX = numpy.c_[doeX, values]
                    previous_line = line
                    line = file.readline()
            if empty_file:
                doeX = []
            else:
                doeX = numpy.transpose(doeX)
                if numpy.shape(doeX)[1] != len(labels):
                    raise SyntaxError
                # Modify units name depending on choosen option
                for idx in range(len(labels)):
                    label = labels[idx]
                    parameter = label[0:label.find('(')] if label.find('(') != -1 else label
                    if parameter in list(parameter_set.dictionary.keys()):
                        if units == 'SI':
                            labels[idx] = parameter
                        elif units == 'defined':
                            labels[idx] = parameter + ' [' + parameter_set[parameter].defined_units + ']'
                        elif units == 'from_file':
                            if label.find('(') == -1:
                                labels[idx] = parameter
                            else:
                                file_units = label[label.find('(') + 1: len(label) - 1]
                                ureg = pint.UnitRegistry()
                                ureg.default_system = 'mks'
                                Q_ = ureg.Quantity
                                try:
                                    value = Q_(1, file_units).to_base_units()
                                    if str(value.units) != parameter_set[parameter]._SI_units:
                                        raise ValueError('dimensions mismatch for parameter {}, {} found instead of {}.'.format(parameter, str(value.units),\
                                                         parameter_set[parameter]._SI_units))
                                except:
                                    labels[idx] = parameter
                                    warnings.warn('parameter {} units defined in file are unreadable, SI units are applied!'.format(parameter))
                    else:
                        if label.find('(') != -1:
                            parameter = label[0:label.find('(')]
                            file_units = label[label.find('(') + 1: len(label) - 1]
                            ureg = pint.UnitRegistry()
                            ureg.default_system = 'mks'
                            Q_ = ureg.Quantity
                            try:
                                value = Q_(1, file_units).to_base_units()
                                if units == 'SI':
                                    labels[idx] = parameter + ' [' + str(value.units) + ']'
                                else:
                                    labels[idx] = parameter + ' [' + file_units + ']'
                            except:
                                labels[idx] = parameter
                                warnings.warn('parameter {} units defined in file are unreadable, it is supposed to be SI units!'.format(parameter))
                # Adapt values with units and errase in name if in parameter set
                ureg = pint.UnitRegistry()
                ureg.default_system = 'mks'
                Q_ = ureg.Quantity
                for idx in range(len(labels)):
                    label = labels[idx]
                    if label.find(' [') != -1:
                        parameter = label[0:label.find('[')]
                        parameter = parameter.replace(' ','')
                        file_units = label[label.find('[') + 1: len(label) - 1]
                        file_units = file_units.replace(' ','')
                        value = Q_(1, file_units)
                        SI_value = value.to_base_units()
                        if (value.magnitude == SI_value.magnitude) and (parameter in list(parameter_set.dictionary.keys())):
                            labels[idx] = parameter
                        elif (value.magnitude != SI_value.magnitude) and (parameter in list(parameter_set.dictionary.keys())):
                            labels[idx] = parameter
                            for nr in range(len(doeX[:,idx])):
                                value = Q_(doeX[nr, idx], file_units).to_base_units()
                                doeX[nr, idx] = value.magnitude
                        elif (value.magnitude != SI_value.magnitude):
                            labels[idx] = parameter + ' ['+ str(SI_value.units) + ']'
                            for nr in range(len(doeX[:,idx])):
                                value = Q_(doeX[nr, idx], file_units).to_base_units()
                                doeX[nr, idx] = value.magnitude
                        else:
                            labels[idx] = parameter + ' ['+ file_units + ']'
                doeX = pandas.DataFrame(doeX, columns=labels)
                # Adapt column order considering parameter_set order (add unknown parameter at the end)
                new_labels = []
                parameters_list = list(parameter_set.dictionary.keys())
                for parameter in parameters_list:
                    if parameter in labels:
                        new_labels.extend([parameter])
                    else:
                        warnings.warn('parameter {} not found in imported file!'.format(parameter))
                for label in labels:
                    if not(label in new_labels):
                        new_labels.extend([label])
                doeX = doeX[new_labels]
            file.close()
            return doeX
        except SyntaxError:
            raise SyntaxError('parameter number and values mismatch, check that you have no spaces in parameters\' names')
        else:
            raise SyntaxError('file name inapropriate, unable to open file for writing purpose.')
    else:
        if not(isinstance(file_name, str)):
            raise TypeError('file_name should be a string')
        elif not(isinstance(parameter_set, PositiveParameterSet)):
            raise TypeError('parameter_set should be a PositiveParameterSet')
        else:
            raise TypeError('is_SI should be a string')
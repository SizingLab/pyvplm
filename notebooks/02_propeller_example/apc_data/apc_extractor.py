# -*- coding: utf-8 -*-
"""
APC specific extractor
"""

import os
import re
import numpy
import pandas

file_list = os.listdir("./")
comp = 0
parameter_names = []
X = numpy.array([])
for file_name in file_list:
    if file_name[len(file_name) - 4 : len(file_name)] == ".dat":
        with open("./" + file_name, "r") as file:
            comp += 1
            line = file.readline()
            data = line.split("x")
            diameter = data[0].replace(" ", "")
            data = data[1].split(" ")
            pitch = data[0]
            blades_nb = "2"
            if pitch.find("MR") != -1:
                pitch = pitch.replace("MR", "")
                prop_type = "MR"
            elif pitch.find("EP") != -1:
                pitch = pitch.replace("EP", "")
                prop_type = "EP"
            elif pitch.find("E") != -1:
                pitch = pitch.replace("E", "")
                prop_type = "E"
            elif pitch.find("SF") != -1:
                pitch = pitch.replace("SF", "")
                prop_type = "SF"
            elif pitch.find("N") != -1:
                pitch = pitch.replace("N", "")
                prop_type = "N"
            elif pitch.find("W") != -1:
                pitch = pitch.replace("W", "")
                prop_type = "W"
            elif pitch.find("C") != -1:
                pitch = pitch.replace("C", "")
                prop_type = "C"
            elif pitch.find("-4blades") != -1:
                pitch = pitch.replace("-4blades", "")
                prop_type = ""
                blades_nb = "4"
            elif pitch.find("-3blades") != -1:
                pitch = pitch.replace("-3blades", "")
                prop_type = ""
                blades_nb = "3"
            else:
                prop_type = "NULL"
            save_data = False
            while len(line) != 0:
                if line.find("PROP RPM =") != -1:
                    n = line.split("=")
                    n = n[1].replace(" ", "")
                    n = n.replace("\n", "")
                    line = file.readline()
                    line = file.readline()
                    save_data = True
                    if len(parameter_names) == 0:
                        parameter_names = numpy.array(line.split(" "), dtype=numpy.dtype(("U", 30)))
                        to_be_deleted = numpy.zeros(len(parameter_names)).astype(bool)
                        for idx in range(len(parameter_names)):
                            parameter = parameter_names[idx].replace(" ", "")
                            parameter = parameter.replace("\t", "")
                            parameter = parameter.replace("\n", "")
                            if (
                                len(parameter) == 0
                                or len(parameter) == 1
                                and (parameter != "V" and parameter != "J")
                            ):
                                to_be_deleted[idx] = True
                        parameter_names = parameter_names[to_be_deleted == False]
                        for idx in range(len(parameter_names)):
                            parameter = parameter_names[idx]
                            parameter = parameter.replace(" ", "")
                            if parameter == "V":
                                parameter_names[idx] = parameter + " (MPH)"
                            elif parameter == "PWR":
                                parameter_names[idx] = parameter + " (HP)"
                            elif parameter == "Torque":
                                parameter_names[idx] = parameter + " (IN.LBF)"
                            elif parameter == "Thrust":
                                parameter_names[idx] = parameter + " (LBF)"
                        parameter_names = parameter_names.tolist()
                        parameter_names.append("DIAMETER (IN)")
                        parameter_names.append("PITCH (IN)")
                        parameter_names.append("TYPE")
                        parameter_names.append("BLADE(nb)")
                        parameter_names.append("COMP")
                        parameter_names.append("RPM")
                        line = file.readline()
                        line = file.readline()
                    else:
                        line = file.readline()
                        line = file.readline()
                if save_data:
                    if line.find(".") == -1:
                        save_data = False
                    else:
                        if line.find("NaN") == -1:
                            s = line
                            values = numpy.array(
                                eval("[" + re.sub("\s+", ",", s.strip()) + "]"),
                                dtype=numpy.dtype(("U", 30)),
                            )
                            values = numpy.append(values, diameter)
                            values = numpy.append(values, pitch)
                            values = numpy.append(values, prop_type)
                            values = numpy.append(values, blades_nb)
                            values = numpy.append(values, comp)
                            values = numpy.append(values, n)
                            if len(X) == 0:
                                X = values
                            else:
                                X = numpy.c_[X, values]
                line = file.readline()
            file.close()
            print("Data extracted from file: ./" + file_name)
X = numpy.transpose(X)
X = pandas.DataFrame(X, columns=parameter_names)
X.to_csv(".\APC_summary_file.csv", sep=";", index=False)

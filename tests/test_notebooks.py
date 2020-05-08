# -*- coding: utf-8 -*-
"""
Code embedded into Notebooks n°1 to n°4 tested using pytest
"""
# -------[Extend pyVPLM directories]------------------------------------------
import os
import pandas
import numpy
from numpy import log10
import matplotlib.pyplot as plot
from pyvplm.core.definition import PositiveParameter, PositiveParameterSet
from pyvplm.addon.variablepowerlaw import (
    buckingham_theorem,
    automatic_buckingham,
    reduce_parameter_set,
    declare_func_x_to_pi,
    regression_models,
    perform_regression,
    force_buckingham,
    pi_sensitivity,
    pi_dependency,
)
from pyvplm.addon.pixdoe import create_const_doe
from pyvplm.addon.comsoladdon import save_file, import_file
from pyDOE2 import lhs


# -------[Import necessary packages]--------------------------------------------


def test_notebook1():
    print("\n Notebook1 tested...")
    # Declare physical variables and set
    rth = PositiveParameter("rth", [0.01, 100], "K/W", "thermal resistance hot spot/base")
    d = PositiveParameter("d", [10e-3, 150e-3], "m", "pot external diameter")
    e = PositiveParameter("e", [0.1e-3, 10e-3], "m", "airgap thickness")
    LAMBDA_FERRITE = PositiveParameter(
        "LAMBDA_FERRITE", [5], "W/m/K", "thermal conductivity for ferrite"
    )
    lambda_wind = PositiveParameter(
        "lambda_wind", [0.1, 3], "W/(m*K)", "equivalent thermal conductivity for winding"
    )
    parameter_set = PositiveParameterSet(rth, d, e, lambda_wind, LAMBDA_FERRITE)
    # Define d and lambda_wind to be first elected as repetitive set
    parameter_set.first("d", "lambda_wind")
    # Calculate pi parameters with first repetitive set found in parameters' order
    pi_set, _ = buckingham_theorem(parameter_set, track=False)
    # Compute automatic search of pi [step that may take a while, so be patient]
    combinatory_pi_set, exp_dict = automatic_buckingham(parameter_set, track=False)
    # Add manually the e<=d/10 constraint <=> e/d=pi2<=0.1
    pi_set["pi2"].defined_bounds = [pi_set["pi2"].defined_bounds[0], 0.1]
    # Extract Pi0 and Rth before calculating DOE
    reduced_parameter_set, reduced_pi_set = reduce_parameter_set(
        parameter_set, pi_set, elected_output="rth"
    )
    # Save function to translate x parameters to pi parameters
    func_x_to_pi = declare_func_x_to_pi(reduced_parameter_set, reduced_pi_set)
    # Generate at least 50 points DOE on physical parameters to have proper fullfact doe on PI (distance criteria is 1/5 and 3 points elected)
    doeX, _ = create_const_doe(
        reduced_parameter_set, reduced_pi_set, func_x_to_pi, whished_size=50, test_mode=True
    )
    plot.close("all")
    # Save .txt file compatible with COMSOL
    file_name = "input_01"
    save_file(doeX, file_name, reduced_parameter_set, is_SI=True, test_mode=True)
    os.remove("input_01.txt")
    # Import FEM thermal simulations results and plot 3 first points
    doeX_FEM = import_file(
        "notebooks/01_inductor_example/output_01", parameter_set, units="from_file"
    )
    # Make little post-processing by replacing power loss and temperature drop by Rth:
    doeX_FEM["rth"] = (doeX_FEM["Temperature [K]"] - 300) / doeX_FEM["Total_heat_source [W]"]
    doeX_FEM = doeX_FEM.drop(columns=["Total_heat_source [W]", "Temperature [K]"])
    doeX_FEM = doeX_FEM[list(parameter_set.dictionary.keys())]
    # Define transformation function with full set that time
    func_x_to_pi = declare_func_x_to_pi(parameter_set, pi_set)
    # Fit with 2nd order model the obtained Pi DOE
    doePI = func_x_to_pi(doeX_FEM.values)
    models = regression_models(doePI, elected_pi0="pi1", order=2, test_mode=True)
    perform_regression(doePI, models, choosen_model=5, test_mode=True)


def test_notebook2():
    print("\n Notebook2 tested...")
    # Declare physical variables and set (limits have no impact since DOE is not constructed)
    t = PositiveParameter("t", [0.01, 100], "N", "trust")
    RHO = PositiveParameter("RHO", [1.184], "kg/m^3", "air density")
    n = PositiveParameter("n", [0.1, 1500], "rad/s", "rotational speed")
    d = PositiveParameter("d", [3, 50], "cm", "propeller diameter")
    pitch = PositiveParameter("pitch", [1, 10], "inch", "propeller pitch")
    v = PositiveParameter("v", [0.01, 10], "m/s", "air speed")
    BETA = PositiveParameter("BETA", [101000], "Pa", "air bulk modulus")
    parameter_set = PositiveParameterSet(t, RHO, n, d, pitch, v, BETA)
    # Define d and lambda_wind to be first elected as repetitive set
    parameter_set.first("n", "d", "RHO")
    # Define d and lambda_wind to be first elected as repetitive set
    parameter_set.first("n", "d", "RHO")
    # Calculate pi parameters with first repetitive set found in parameters' order
    pi_set, _ = buckingham_theorem(parameter_set, False)
    # Force expressions
    pi_set = force_buckingham(
        parameter_set, "t/(RHO*n**2*d**4)", "pitch/d", "v/(n*d)", "d**2*n**2*RHO/BETA"
    )
    # Define power parameter and create new set to higligh power relation
    p = PositiveParameter("p", [0.0001, 100], "W", "mechanical power")
    parameter_set2 = PositiveParameterSet(RHO, n, d, pitch, v, BETA, p)
    parameter_set2.first("RHO", "n", "d")
    # Read the .csv file
    df = pandas.read_csv("notebooks/02_propeller_example/APC_STATIC-data-all-props.csv", sep=";")
    # Make little datasheet post-processing keeping only multy-rotor with N.D<=105000 and extract PI-DOE
    df_MR = df[df["TYPE"] == "MR"]
    df_MR = df_MR[df_MR["N.D"] <= 105000]
    pi1 = df_MR["Ct"]
    pi2 = df_MR["PITCH(IN)"] / df_MR["DIAMETER(IN)"]
    pi3 = (
        float(RHO)
        / float(BETA)
        * (0.0254 * df_MR["DIAMETER(IN)"]) ** 2
        * (2 * 3.14 / 60 * df_MR["RPM"]) ** 2
    )
    doePI = numpy.c_[numpy.array(pi1), numpy.array(pi2), numpy.array(pi3)]
    doePI = pandas.DataFrame(doePI, columns=["pi1", "pi2", "pi3"])
    # Fit with 5th order model the obtained Pi DOE pi0=pi1
    models = regression_models(doePI.values, elected_pi0="pi1", order=5, test_mode=True)
    # Plot advanced result for model n°7
    perform_regression(doePI.values, models, choosen_model=7, test_mode=True)
    # Fit with 3rd order polynomia model the obtained Pi DOE
    models3 = regression_models(
        doePI.values, elected_pi0="pi1", order=3, log_space=False, test_mode=True
    )
    # Plot advanced result for model n°4
    perform_regression(doePI.values, models3, choosen_model=4, test_mode=True)
    # Read the .csv file
    df2 = pandas.read_csv("notebooks/02_propeller_example/APC_summary_file.csv", sep=";")
    # Remove all <=0 terms (not to have infinite values in log space) and extract data for regression
    df2_reduced = df2[df2["Ct"] >= 0.01]
    df2_reduced = df2_reduced[df2_reduced["J"] != 0]
    df2_reduced = df2_reduced[(df2_reduced["RPM"] * df2_reduced["DIAMETER (IN)"]) <= 105000]
    pi1 = df2_reduced["Ct"]
    pi2 = df2_reduced["PITCH (IN)"] / df2_reduced["DIAMETER (IN)"]
    pi3 = df2_reduced["J"]
    pi4 = (
        float(RHO)
        / float(BETA)
        * (0.0254 * df2_reduced["DIAMETER (IN)"]) ** 2
        * (2 * 3.14 / 60 * df2_reduced["RPM"]) ** 2
    )
    doePI2 = numpy.c_[numpy.array(pi1), numpy.array(pi2), numpy.array(pi3), numpy.array(pi4)]
    doePI2 = pandas.DataFrame(doePI2, columns=["pi1", "pi2", "pi3", "pi4"])
    # Fit with 3rd order power-law model the obtained Pi DOE
    models4 = regression_models(doePI2.values, elected_pi0="pi1", order=3, test_mode=True)
    # Plot advanced result for model n°8
    perform_regression(doePI2.values, models4, choosen_model=8, test_mode=True)
    # Fit with 2nd order polynomia model the obtained Pi DOE
    models5 = regression_models(
        doePI2.values, elected_pi0="pi1", order=2, log_space=False, test_mode=True
    )
    # Plot advanced result for model n°6
    perform_regression(doePI2.values, models5, choosen_model=6, test_mode=True)


def test_notebook3():
    print("\n Notebook3 tested...")
    # Declare physical variables and set
    E_S = PositiveParameter("E_S", [210e9], "Pa", "steel Young modulus")
    d_rs = PositiveParameter("d_rs", [10e-3, 100e-3], "m", "roller-screw nut diameter")
    e1 = PositiveParameter("e1", [5e-3, 40e-3], "m", "smaller diameter housing thikness")
    e2 = PositiveParameter("e2", [1e-3, 10e-3], "m", "bigger diameter housing thikness")
    RHO = PositiveParameter("RHO", [8.05], "kg/m^3", "steel density")
    L_a = PositiveParameter("L_a", [0.1, 1.2], "m", "actuator length")
    L_rs = PositiveParameter("L_rs", [3e-3, 100e-3], "m", "roller-screw nut length")
    omega0 = PositiveParameter("omega0", [5, 2000], "rad/s", "angular resonance pulsation")
    a = PositiveParameter("a", [0.1, 100], "m/s^2", "acceleration rate")
    sigma = PositiveParameter("sigma", [1e6, 1e9], "Pa", "maximal constraint")
    Q_M = PositiveParameter("Q_M", [2], "", "quality factor")
    parameter_set = PositiveParameterSet(E_S, d_rs, e1, e2, L_a, L_rs, RHO, omega0, a, sigma, Q_M)
    # Calculate pi parameters with first repetitive set found in parameters' order
    pi_set, _ = buckingham_theorem(parameter_set, track=False)
    # Read the .csv file
    df = pandas.read_csv("notebooks/03_housing_example/dataPI0_carter.csv", sep=";")
    # Fit with 2nd order model the obtained Pi DOE
    doePI = df.values
    models = regression_models(doePI, elected_pi0="pi1", order=2, test_mode=True)
    # Plot advanced result for model n°4
    perform_regression(doePI, models, choosen_model=4, test_mode=True)
    # Define geometry an material
    d_rs = 0.09  # [m] roller-screw nut diameter
    la = 0.9  # [m] actuator length
    e1 = 0.01  # [m] Road thickness
    e2 = 0.005  # [m] Housing thickness
    Qm = 30  # [-] Mechanical quality coef
    rho = 7800  # [kg/m^3] Volumic mass
    a = 20 * 9.8  # [m/s^2] Acceleration
    # Pi calculation
    pi2 = la / d_rs
    pi3 = e1 / d_rs
    pi4 = e2 / d_rs
    expression = "10**(1.28187-0.60263*log10(pi3)+0.80237*log10(pi4)+0.83024*log10(pi2)+0.40250*log10(pi2)**2+0.07760*log10(pi2)*log10(pi4)+0.08495*log10(pi2)*log10(pi3)+0.27216*log10(pi4)**2+0.13195*log10(pi3)**2-0.21442*log10(pi3)*log10(pi4))"
    pi1 = eval(expression)
    sigma = pi1 * Qm * d_rs * rho * a


def test_notebook_4():
    print("\n Notebook4 tested...")
    # Define SPM design parameters
    d_e = PositiveParameter("d_e", [50, 500], "mm", "External stator diameter")
    d_i = PositiveParameter("d_i", [20, 300], "mm", "Internal stator diameter")
    e_tooth = PositiveParameter("e_tooth", [3, 60], "mm", "Tooth thikness")
    e_yoke = PositiveParameter("e_yoke", [2, 20], "mm", "Yoke thikness")
    w_pm = PositiveParameter("w_pm", [2, 20], "mm", "Permanent magnet width")
    r_i = PositiveParameter("r_i", [5, 100], "mm", "Rotor internal radius")
    j = PositiveParameter("j", [0.1, 1000], "A/m**2", "Winding current density")
    B_R = PositiveParameter("B_R", [1.1], "tesla", "Permanent magnet remanence")
    B_SAT = PositiveParameter("B_SAT", [0.02], "tesla", "Iron induction saturation")
    MU_0 = PositiveParameter("MU_0", [1.26e-6], "H/m", "Vacuum permeability")
    t_l = PositiveParameter("t_l", [0.01, 100], "N", "Linear torque")
    parameter_set = PositiveParameterSet(
        d_e, d_i, e_tooth, e_yoke, w_pm, r_i, j, B_R, B_SAT, MU_0, t_l
    )
    # Perform dimensional analysis on linear torque
    pi_set, _ = buckingham_theorem(parameter_set, False)
    # Define new parameters for joule losses definition
    p_jl = PositiveParameter("p_jl", [0.01, 1000], "W/m", "Linear joule losses")
    RHO_WIND = PositiveParameter("RHO_WIND", [17000], "ohm*m", "Linear winding resistivity")
    s_wind = PositiveParameter("s_wind", [1, 100], "mm**2", "Winding total cross section")
    parameter_set = PositiveParameterSet(
        d_e, d_i, e_tooth, e_yoke, w_pm, r_i, j, B_R, B_SAT, MU_0, RHO_WIND, s_wind, p_jl
    )
    # Perform dimensional analysis on joule losses
    pi_set, _ = buckingham_theorem(parameter_set, False)
    # Calculate levels
    bounds = numpy.array([[30, 150], [10, 100], [750, 3000]])
    doe_levels = lhs(3, samples=27, criterion="maximin", random_state=42)
    doe = bounds[:, 0] + doe_levels / doe_levels.max(axis=0) * (bounds[:, 1] - bounds[:, 0])
    # Show matrix
    doe_data = pandas.DataFrame(doe, columns=["d_e", "h", "omega_max"])
    doe_data.to_excel("output.xls")
    os.remove("output.xls")
    doe_data.head(n=numpy.shape(doe)[0])
    # Plot 3D figure
    fig = plot.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(doe[:, 0], doe[:, 1], doe[:, 2])
    ax.set_xlabel("$D_e$")
    ax.set_ylabel("$h$")
    ax.set_zlabel("$\omega_{max}$")
    doe = pandas.read_excel("notebooks/04_motor_example/output.xls")
    # Declare directly the pi_set problem
    doePI = doe[["pi01", "pi02", "pi03", "pi1", "pi2", "pi3", "pi4", "pi5", "pi6"]].values
    pi1 = PositiveParameter("pi1", [0.1, 1], "", "t_l*b_r**-1*j**-1*d_e**-3")
    pi2 = PositiveParameter("pi2", [0.1, 1], "", "p_j*rho_win**-1*d_e**-2*j**-2")
    pi3 = PositiveParameter(
        "pi3", [0.1, 1], "", "p_fe*delta_p**-1*omega_max**1.5*b_r**-2*d_iron**-1*d_e**-2"
    )
    pi4 = PositiveParameter("pi4", [0.1, 1], "", "mu_0*j*d_e*b_r**-1")
    pi5 = PositiveParameter("pi5", [0.1, 1], "", "d_i*d_e**-1")
    pi6 = PositiveParameter("pi6", [0.1, 1], "", "e_tooth*d_e**-1*n")
    pi7 = PositiveParameter("pi7", [0.1, 1], "", "e_yoke*d_e**-1*n")
    pi8 = PositiveParameter("pi8", [0.1, 1], "", "w_pm*d_e**-1")
    pi9 = PositiveParameter("pi9", [0.1, 1], "", "r_i*d_e**-1")
    pi_set = PositiveParameterSet(pi1, pi2, pi3, pi4, pi5, pi6, pi7, pi8, pi9)
    # Perform sensitivity analysis
    pi_sensitivity(pi_set, doePI, useWidgets=False, test_mode=True)
    # Perform dependency analysis
    pi_dependency(pi_set, doePI, useWidgets=False, test_mode=True)

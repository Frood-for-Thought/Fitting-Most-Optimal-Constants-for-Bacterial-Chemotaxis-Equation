import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import the Tumble Angle Module
from Tumble_Angle import Angle_Generator
# Initialize the angle generator class to select from probability distribution.
angle_generator = Angle_Generator()


# Initialization and Food Concentration Calculation
nl = 101
Grad = 0.000405  # µm^-1
Max_Food_Conc = 60000  # µM
DL = 310  # µm
Food_Function = np.zeros(nl)
for Food_Pos in range(nl):
    Food_Function[Food_Pos] = np.exp(Grad * Food_Pos * DL)
Ini_Food_Const = Max_Food_Conc / Food_Function[-1]
xbias = Ini_Food_Const * Food_Function


# Generate a new tumble angle
next_tumble_angle = angle_generator.tumble_angle_function()


# The theoretical parameters have been pre-calculated to fit onto.
# These parameters are for:
#     nl = 101
#     Grad = 0.000405  # µm^-1
#     Max_Food_Conc = 60000  # µM
#     DL = 310  # µm
input_parameters = '../input_parameters.xlsx'
parameter_df = pd.read_excel(input_parameters)
vd_chemotaxis = parameter_df.loc[:, 'drift_velocity']  # The theoretical drift velocity per deme.
c_df_over_dc = parameter_df.loc[:, 'c_x_df_l_dc']  # Concentration*df/dc.
vo_max = parameter_df.loc[1, 'Vo_max']  # The run speed.
# Timed rate of change of the amount of receptor protein bound.
Rtroc = vd_chemotaxis*Grad*c_df_over_dc  # This numpy vector is calculated from the above constant and pandas series.

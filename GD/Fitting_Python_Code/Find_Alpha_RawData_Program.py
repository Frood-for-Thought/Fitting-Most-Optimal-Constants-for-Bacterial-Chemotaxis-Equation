import numpy as np
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

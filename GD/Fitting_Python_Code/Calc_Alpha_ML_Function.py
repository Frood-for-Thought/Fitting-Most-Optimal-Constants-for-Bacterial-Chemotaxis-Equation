import numpy as np
import pandas as pd
import math
# Import the Tumble Angle Module
from Tumble_Angle import Angle_Generator

# class Norm_Mean_Match_ML:
#     def __init__(self):
#         self.Rtroc = Rtroc
#         self.next_tumble_angle = next_tumble_angle
#         self.vd_chemtaxis = vd_chemtaxis


class NormMeanMatchDataGenerator:
    def __init__(self, Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt, max_iter):
        self.Rtroc = Rtroc  # Time rate of change of the fractional amount of receptor (protein) bound.
        self.alpha = alpha  # The alpha constant to be found.
        self.Angle = Angle  # Bacterial orientation angle.
        self.Vo_max = Vo_max  # Run Speed
        self.DL = DL  # Deme length [µM]
        self.nl = nl  # Total number of demes.
        self.deme_start = deme_start  # Deme starting position.
        self.d = diff  # Diffusion constant
        self.dt = dt  # Time step
        self.pos = DL * deme_start  # Position variable [µM]
        self.pos_ini = DL * deme_start  # Starting position [µM]
        self.angle_generator = Angle_Generator.tumble_angle_function()  # Generating new orientation angle from PDF.
        self.max_iter = max_iter  # Number of data points generated.

    def simulate_bacterial_movement(self):
        pos = self.pos
        Angle = self.Angle
        Calculated_Ave_Vd_Array = []
        iter = 1

        while iter < self.max_iter:
            for t in np.arange(1, 1000, self.dt):
                i = int(pos // self.DL)
                if (90 <= Angle) and (Angle < 270):
                    Ptum = self.dt * np.exp(-self.d + self.alpha * self.Rtroc[i])
                else:
                    Ptum = self.dt * np.exp(-self.d - self.alpha * self.Rtroc[i])

                R_rt = np.random.rand()
                if R_rt < Ptum:
                    Next_Angle = self.angle_generator.tumble_angle_function()
                    Angle = (Angle + Next_Angle) % 360
                else:
                    Dot_Product = np.cos(np.radians(Angle))
                    pos += self.dt * self.Vo_max * Dot_Product

                pos = max(0, min(pos, self.nl * self.DL))

                if pos == 0 or pos >= self.nl * self.DL:
                    break

            Calculated_Ave_Vd = (pos - self.pos_ini) / t
            Calculated_Ave_Vd_Array.append(Calculated_Ave_Vd)
            pos = self.DL * self.deme_start
            iter += 1

        return np.array(Calculated_Ave_Vd_Array)

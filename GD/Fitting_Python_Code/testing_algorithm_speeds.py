import numpy as np
import pandas as pd
import torch
import time
# Import the Tumble Angle Module
from Tumble_Angle import Angle_Generator


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
        self.angle_generator = angle_generator  # Generating new orientation angle from PDF.
        self.max_iter = max_iter  # Number of data points generated.

    def simulate_bacterial_movement_cpu(self):
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

    def simulate_bacterial_movement_cuda(self):
        pos = torch.tensor(self.pos, device='cuda')
        Angle = torch.tensor(self.Angle, device='cuda')
        Rtroc_cuda = torch.tensor(self.Rtroc.values, device='cuda')
        Calculated_Ave_Vd_Array = []
        iter = 1

        while iter < self.max_iter:
            for t in torch.arange(1, 1000, self.dt, device='cuda'):
                i = (pos // self.DL).long()
                if (90 <= Angle) and (Angle < 270):
                    Ptum = self.dt * torch.exp(-self.d + self.alpha * Rtroc_cuda[i.item()])
                else:
                    Ptum = self.dt * torch.exp(-self.d - self.alpha * Rtroc_cuda[i.item()])

                R_rt = torch.rand(1, device='cuda')
                if R_rt < Ptum:
                    Next_Angle = self.angle_generator.tumble_angle_function()
                    Angle = (Angle + Next_Angle) % 360
                else:
                    Dot_Product = torch.cos(torch.radians(Angle))
                    pos += self.dt * self.Vo_max * Dot_Product

                pos = torch.clamp(pos, 0, self.nl * self.DL)

                if pos == 0 or pos >= self.nl * self.DL:
                    break

            Calculated_Ave_Vd = (pos - self.pos_ini) / t
            Calculated_Ave_Vd_Array.append(Calculated_Ave_Vd.item())
            pos = torch.tensor(self.DL * self.deme_start, device='cuda')
            iter += 1

        return torch.tensor(Calculated_Ave_Vd_Array, device='cuda')

    def time_execution(self):
        # Measure CPU execution time
        start_time = time.time()
        cpu_data = self.simulate_bacterial_movement_cpu()
        cpu_time = time.time() - start_time

        # Measure CUDA execution time
        start_time = time.time()
        cuda_data = self.simulate_bacterial_movement_cuda()
        cuda_time = time.time() - start_time

        print(f"CPU execution time: {cpu_time:.4f} seconds")
        print(f"CUDA execution time: {cuda_time:.4f} seconds")

        # Verify that both implementations produce similar results
        print("CPU data:", cpu_data)
        print("CUDA data:", cuda_data.cpu().numpy())  # Convert CUDA tensor to numpy array for comparison

# Parameters for the simulation
# Initialization and Food Concentration Calculation.
nl = 101
Grad = 0.000405  # µm^-1
Max_Food_Conc = 60000  # µM
DL = 310  # µm
Food_Function = np.zeros(nl)
for Food_Pos in range(nl):
    Food_Function[Food_Pos] = np.exp(Grad * Food_Pos * DL)
Ini_Food_Const = Max_Food_Conc / Food_Function[-1]
xbias = Ini_Food_Const * Food_Function

# Generate a new tumble angle.
angle_generator = Angle_Generator()
next_tumble_angle = angle_generator.tumble_angle_function()
Start_Angle = 90  # degrees
Angle = Start_Angle

# The theoretical parameters have been pre-calculated to fit onto.
# These parameters are for:
#     nl = 101
#     Grad = 0.000405  µm^-1
#     Max_Food_Conc = 60000  µM
#     DL = 310  µm
input_parameters = '../input_parameters.xlsx'
parameter_df = pd.read_excel(input_parameters)
vd_chemotaxis = parameter_df.loc[:, 'drift_velocity']  # The theoretical drift velocity per deme.
c_df_over_dc = parameter_df.loc[:, 'c_x_df_l_dc']  # Concentration*df/dc.
Vo_max = parameter_df.loc[1, 'Vo_max']  # The run speed.
# Timed rate of change of the amount of receptor protein bound.
Rtroc = vd_chemotaxis * Grad * c_df_over_dc  # This numpy vector is calculated from the above constant and pandas series.

alpha = 500
diff = 1.16
dt = 0.1
max_iter = 1000
deme_start = 30

# Initialize the data generator
data_generator = NormMeanMatchDataGenerator(Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt, max_iter)

# Time the execution of both algorithms
data_generator.time_execution()
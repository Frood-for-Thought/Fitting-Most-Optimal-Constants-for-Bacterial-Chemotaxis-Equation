import numpy as np
import pandas as pd
import torch
import time
import logging
# Import the Tumble Angle Module
from Tumble_Angle import Angle_Generator
from Tumble_Angle import AngleGenerator_cuda

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        self.max_iter = max_iter  # Number of data points generated.

        logging.info(f"Initialized NormMeanMatchDataGenerator with: alpha={alpha}, Angle={Angle}, Vo_max={Vo_max}, "
                     f"DL={DL}, nl={nl}, deme_start={deme_start}, diff={diff}, dt={dt}, max_iter={max_iter}")

    def simulate_bacterial_movement_cpu(self):
        pos = self.pos
        Angle = self.Angle
        Calculated_Ave_Vd_Array = []
        angle_generator = Angle_Generator()
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
                    Next_Angle = angle_generator.tumble_angle_function()
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
        Calculated_Ave_Vd_Array = []

        # Initializing total time steps.
        time_steps = torch.arange(0, 1000, self.dt, device='cuda')
        num_steps = time_steps.size(0)

        # Initialize tensors for pos and Angle for all iterations.
        position = torch.zeros((num_steps, self.max_iter), device='cuda')
        ang = torch.zeros((num_steps, self.max_iter), device='cuda')
        # Starting Position.
        position[0, :] = torch.full((self.max_iter,), self.pos, device='cuda')
        # Starting Angle.
        ang[0, :] = torch.randint(0, 360, (self.max_iter,), device='cuda').float()  # Random angles

        # Calculating the Timed Rate of Change Tensor
        Rtroc_tensor = torch.tensor(self.Rtroc, device='cuda')  # Convert Rtroc to tensor
        Rtroc_tensor_size = Rtroc_tensor.size(0)

        # Initialize the angle generator class to select from probability distribution.
        angle_generator = AngleGenerator_cuda()
        # Calculating the Total Angles needed for the Algorithm.
        total_angles = num_steps * self.max_iter
        Next_Angle = angle_generator.tumble_angle_function_cuda(size=total_angles).view(num_steps, self.max_iter)
        Next_Angle_Size = Next_Angle.size()

        # The random numbers to be used.
        R_rt = torch.rand(total_angles, device='cuda').view(num_steps, self.max_iter)
        R_rt_size = R_rt.size()

        # Initialize Ptum as a 2D tensor with dimensions [num_steps, max_iter].
        Ptum = torch.zeros((num_steps, self.max_iter), device='cuda')
        Ptum_size = Ptum.size()

        for t_idx, t in enumerate(time_steps):
            # Tensors inside the for loop are vectorized and in parallel.

            # The .long() method ensures the tensor is of integer type, which is necessary for indexing.
            i = (position[t_idx, :] // self.DL).long()

            # Direction for moving up or down gradient.
            # The boolean is true if it is moving down the gradient.
            direction_condition = (90 <= ang[t_idx]) & (ang[t_idx] < 270)

            # Calculate Ptum using torch.where
            Ptum[t_idx] = torch.where(direction_condition,
                               self.dt * torch.exp(-self.d + self.alpha * Rtroc_tensor[i]),
                               self.dt * torch.exp(-self.d - self.alpha * Rtroc_tensor[i]))

            # Tumbling condition
            tumble_mask = R_rt[t_idx] < Ptum[t_idx]

            # Update angles based on tumbling condition
            Next_Angle = self.angle_generator.tumble_angle_function_cuda(size=max_iter).to('cuda')
            ang[tumble_mask, t_idx] = (ang[tumble_mask, t_idx] + Next_Angle[tumble_mask]) % 360

            next_angles = self.angle_generator.tumble_angle_function_cuda(tumble_mask.sum().item())
            Angle[tumble_mask] = (Angle[tumble_mask] + next_angles) % 360
            Dot_Product = torch.cos(torch.deg2rad(Angle[run_mask]))
            pos[run_mask] += self.dt * self.Vo_max * Dot_Product

            pos = torch.clamp(pos, 0, self.nl * self.DL)
            if (pos == 0).any() or (pos >= self.nl * self.DL).any():
                break

        Calculated_Ave_Vd = (pos - self.pos_ini) / t
        Calculated_Ave_Vd_Array.append(Calculated_Ave_Vd.mean().item())

        return torch.tensor(Calculated_Ave_Vd_Array, device='cuda')

    def time_execution(self):
        # # Measure CPU execution time
        # logging.info("Starting CPU execution")
        # start_time = time.time()
        # cpu_data = self.simulate_bacterial_movement_cpu()
        # cpu_time = time.time() - start_time
        # logging.info(f"CPU execution time: {cpu_time:.4f} seconds")

        # Measure CUDA execution time
        logging.info("Starting CUDA execution")
        start_time = time.time()
        cuda_data = self.simulate_bacterial_movement_cuda()
        cuda_time = time.time() - start_time
        logging.info(f"CUDA execution time: {cuda_time:.4f} seconds")

        # Verify that both implementations produce similar results
        logging.info("Verifying results")
        logging.info(f"CPU data: {cpu_data}")
        logging.info(f"CUDA data: {cuda_data.cpu().numpy()}")  # Convert CUDA tensor to numpy array for comparison

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
max_iter = 20000
deme_start = 30

# Initialize the data generator
data_generator = NormMeanMatchDataGenerator(Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt, max_iter)

# torch.autograd provides classes and functions implementing automatic differentiation of arbitrary
# scalar valued functions. It requires minimal changes to the existing code
# - you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword.
# Autograd includes a profiler that lets you inspect the cost of different operators inside your model
# - both on the CPU and GPU.
# The record_shapes=True option is used to record the shapes of the tensors involved in the operations being profiled.
# This can be helpful in understanding how tensor dimensions change throughout your operations and
# identify potential inefficiencies related to tensor shape manipulations.
with torch.autograd.profiler.profile(record_shapes=True) as prof:
    with torch.autograd.profiler.record_function("simulate_bacterial_movement_cuda"):
        cuda_data = data_generator.simulate_bacterial_movement_cuda()

print(prof.key_averages().table(sort_by="cuda_time_total"))

# # Time the execution of both algorithms
# data_generator.time_execution()

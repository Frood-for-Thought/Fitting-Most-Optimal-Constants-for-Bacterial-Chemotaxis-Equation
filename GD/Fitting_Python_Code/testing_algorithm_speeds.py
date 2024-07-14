import numpy as np
import torch
import time
from Tumble_Angle import Angle_Generator


class NormMeanMatchDataGenerator:
    def __init__(self, Rtroc, F, alpha, Angle, Vo_max, DL, nl, deme_start):
        self.Rtroc = Rtroc
        self.F = F
        self.alpha = alpha
        self.Angle = Angle
        self.Vo_max = Vo_max
        self.DL = DL
        self.nl = nl
        self.deme_start = deme_start
        self.d = 1.16  # Diffusion constant
        self.dt = 0.1  # Time step
        self.pos = DL * deme_start
        self.pos_ini = DL * deme_start
        self.angle_generator = New_Angle()

    def simulate_bacterial_movement_cpu(self):
        pos = self.pos
        Angle = self.Angle
        Calculated_Ave_Vd_Array = []

        max_iter = 1000
        iter = 1

        while iter < max_iter:
            for t in np.arange(1, 1000, self.dt):
                i = int(pos // self.DL)
                if (90 <= Angle) and (Angle < 270):
                    Ptum = self.dt * np.exp(-self.d + self.alpha * self.Rtroc[i])
                else:
                    Ptum = self.dt * np.exp(-self.d - self.alpha * self.Rtroc[i])

                R_rt = np.random.rand()
                if R_rt < Ptum:
                    Next_Angle = self.angle_generator.tumble_angle_function(self.F)
                    Angle = (Angle + Next_Angle) % 360
                else:
                    Dot_Product = np.cos(np.radians(Angle))
                    pos += self.dt * self.Vo_max * Dot_Product

                pos = np.clip(pos, 0, self.nl * self.DL)

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
        Calculated_Ave_Vd_Array = []

        max_iter = 1000
        iter = 1

        while iter < max_iter:
            for t in torch.arange(1, 1000, self.dt, device='cuda'):
                i = (pos // self.DL).long()
                if (90 <= Angle) and (Angle < 270):
                    Ptum = self.dt * torch.exp(-self.d + self.alpha * self.Rtroc[i])
                else:
                    Ptum = self.dt * torch.exp(-self.d - self.alpha * self.Rtroc[i])

                R_rt = torch.rand(1, device='cuda')
                if R_rt < Ptum:
                    Next_Angle = self.angle_generator.tumble_angle_function(self.F)
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


# Example usage
# Define the necessary parameters
Rtroc = [0.1] * 100  # Placeholder values
F = [0.1] * 100  # Placeholder values
alpha = 0.5
Angle = 90
Vo_max = 1.0
DL = 1.0
nl = 100
deme_start = 10

# Initialize the data generator
data_generator = NormMeanMatchDataGenerator(Rtroc, F, alpha, Angle, Vo_max, DL, nl, deme_start)

# Measure execution time for CPU implementation
start_time = time.time()
cpu_data = data_generator.simulate_bacterial_movement_cpu()
cpu_duration = time.time() - start_time

# Measure execution time for CUDA implementation
start_time = time.time()
cuda_data = data_generator.simulate_bacterial_movement_cuda()
cuda_duration = time.time() - start_time

print(f"CPU Duration: {cpu_duration:.4f} seconds")
print(f"CUDA Duration: {cuda_duration:.4f} seconds")

# Verify that the data matches
print("CPU Data:", cpu_data)
print("CUDA Data:", cuda_data.cpu().numpy())  # Convert CUDA tensor to numpy array for comparison

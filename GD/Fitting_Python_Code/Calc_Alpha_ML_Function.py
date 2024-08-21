# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
import torch
import logging
from Generate_Dynamic_Data_Points import Norm_Vd_Mean_Data_Generator


class Dynamic_Data_Evolving_Mean_Estimator:
    """
    Dynamic_Data_Evolving_Mean_Estimator, pronounced 'deme'.
    """
    def __init__(self, Rtroc, alpha, Angle, Vo_max, DL, nl, deme_start, diff, dt, max_iter):
        self.Rtroc = Rtroc  # Time rate of change of the fractional amount of receptor (protein) bound.
        self.alpha = alpha  # The alpha constant to be found.
        self.Angle = Angle  # Bacterial orientation angle.
        self.Vo_max = Vo_max  # Run Speed.
        self.DL = DL  # Deme length [µM].
        self.nl = nl  # Total number of demes.
        self.deme_start = deme_start  # Deme starting position.
        self.d = diff  # Diffusion constant.
        self.dt = dt  # Time step.
        self.pos = DL * deme_start  # Position variable [µM].
        self.pos_ini = DL * deme_start  # Starting position [µM].
        self.max_iter = max_iter  # Number of data points generated.

    def generate_data(self):
        norm_data_gen = Norm_Vd_Mean_Data_Generator(self.Rtroc, self.alpha, self.Angle, self.Vo_max, self.DL, self.nl,
                                                    self.deme_start, self.d, self.dt, self.max_iter)


def train_alpha_model(Rtroc, Vo_max, DL, pos_ini, dt, Theory_Vel, num_iterations=80):
    initial_alpha = 0.1  # Starting value for alpha
    model = AlphaModel(initial_alpha)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for n in range(num_iterations):
        optimizer.zero_grad()

        # Set max_iter and TV based on the current n
        if n <= 20:
            max_iter = 1000
            TV = 1 / (100 * Rtroc[30])  # Example value based on deme_start=30
        elif n <= 40:
            max_iter = 2000
            TV = 1 / (200 * Rtroc[30])
        elif n <= 60:
            max_iter = 4000
            TV = 1 / (300 * Rtroc[30])
        else:
            max_iter = 10000
            TV = 1 / (400 * Rtroc[30])

        Ave_Vel_Diff = model(Rtroc, Vo_max, DL, pos_ini, dt, max_iter, Theory_Vel)

        # Standard error for the current velocity array
        stderror = torch.std(Ave_Vel_Diff) / torch.sqrt(torch.tensor(max_iter, dtype=torch.float32, device='cuda'))

        # Compute the gradient step `h`
        h = -2 * TV * Ave_Vel_Diff
        TV_2_SE = 2 * TV * stderror

        # Update alpha
        model.alpha.data += h

        # Ensure alpha does not go below 0
        if model.alpha.data < 0:
            model.alpha.data = torch.tensor(0.0, device='cuda')

        if n % 20 == 0:
            print(f"Iteration {n}, Alpha: {model.alpha.item()}, Ave_Vel_Diff: {Ave_Vel_Diff.item()}")

        # Adjust learning rate every 20 iterations
        if n % 20 == 0 and n > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= (n // 20 + 1)

    return model.alpha.item()

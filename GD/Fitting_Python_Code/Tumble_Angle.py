import numpy as np
from scipy.stats import rv_continuous
import torch


class Tumble_Angle_Distribution(rv_continuous):
    def _pdf(self, x):
        """
        Override the _pdf describe bacterial tumble angle from "rv_continuous" in "scipy.stats".
        :param x: Angle in Rad.
        :return: P(x) = 0.5*(1+cos(x))*sin(x)
        """
        return 0.5 * (1 + np.cos(x)) * np.sin(x)


class Angle_Generator(Tumble_Angle_Distribution):
    def __init__(self):
        # Initialize the base class with angles from 0 to pi radians.
        super().__init__(a=0, b=np.pi, name='custom_angle_distribution')

    def tumble_angle_function(self):
        """
        Randomly select a new tumble angle from the probability distribution, P(x) = 0.5*(1+cos(x))*sin(x).
        :return: The repositioned tumble angle in degrees.
        """
        random_angle_rad = self.rvs()
        return np.degrees(random_angle_rad)


# Recalculating for CUDA


class TumbleAngleDistribution_cuda:
    def __init__(self, a=0, b=1, rnd_num_range=1):
        self.a = a
        self.b = b
        self.rnd_num_range = rnd_num_range  # From 0 to rnd_num_range

    def pdf(self, x):
        """
        Probability Density Function for the tumble angle.
        :param x: Angle in Rad.
        :return: P(x) = 0.5*(1+cos(x))*sin(x)
        """
        return 0.5 * (1 + torch.cos(x)) * torch.sin(x)

    def sample(self):
        """
        Sample from the custom probability distribution using direct sampling.
        :return: Tensor of samples in radians.
        """
        u = torch.rand(self.rnd_num_range, device='cuda') * (self.b - self.a) + self.a
        return self.pdf(u)  # Evaluate the PDF at the sampled angles


class AngleGenerator_cuda(TumbleAngleDistribution_cuda):
    def __init__(self):
        super().__init__(a=0, b=np.pi, rnd_num_range=1)

    def tumble_angle_function_cuda(self):
        random_angles_rad = self.pdf()
        return torch.rad2deg(random_angles_rad)


# Example usage
angle_generator = AngleGenerator_cuda()
angles = angle_generator.tumble_angle_function_cuda()
print(angles)
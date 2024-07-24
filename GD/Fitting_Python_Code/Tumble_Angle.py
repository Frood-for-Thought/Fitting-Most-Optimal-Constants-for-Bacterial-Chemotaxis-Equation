import numpy as np
from scipy.stats import rv_continuous
import torch
import time


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


class CustomTumbleAngleDistribution:
    def __init__(self, a=0, b=np.pi):
        self.a = a
        self.b = b
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_pdf = self.compute_max_pdf()

    def pdf(self, x):
        """
        Probability Density Function for the tumble angle.
        :param x: Angle in Rad.
        :return: P(x) = 0.5*(1+cos(x))*sin(x)
        """
        return 0.5 * (1 + torch.cos(x)) * torch.sin(x)

    def compute_max_pdf(self):
        """
        Compute the maximum value of the PDF for use in rejection sampling.
        """
        x_vals = torch.linspace(self.a, self.b, 100000, device=self.device)
        pdf_vals = self.pdf(x_vals)
        return torch.max(pdf_vals).item()

    def rvs(self, size=1):
        """
        Random variate generation for the custom distribution using rejection sampling.
        :param size: Number of samples to generate.
        :return: Tensor of samples in radians.
        """
        samples = torch.empty(size, device=self.device)
        count = 0

        while count < size:
            # Generate uniform random numbers between a and b
            x = torch.rand(size - count, device=self.device) * (self.b - self.a) + self.a
            # Generate uniform random numbers between 0 and the maximum of the PDF
            y = torch.rand(size - count, device=self.device) * self.max_pdf
            # Accept the samples that are under the PDF curve
            accept = y < self.pdf(x)
            num_accept = torch.sum(accept).item()

            if num_accept > 0:
                samples[count:count + num_accept] = x[accept]
                count += num_accept

        return samples


class AngleGenerator_cuda:
    def __init__(self):
        self.distribution = CustomTumbleAngleDistribution(a=0, b=np.pi)

    def tumble_angle_function_cuda(self, size=1):
        random_angle_rad = self.distribution.rvs(size)
        return torch.rad2deg(random_angle_rad).to(self.distribution.device)


# Function to test multiple angle generations
def test_multiple_generations(num_generations):
    angle_generator = AngleGenerator_cuda()
    start_time = time.time()

    angles = angle_generator.tumble_angle_function_cuda(num_generations)

    end_time = time.time()
    print(
        f"Generated {num_generations} angles in {end_time - start_time:.4f} seconds with mean {torch.mean(angles).item():.2f}")


# Example usage
num_generations = 100000  # Number of times to generate angles
test_multiple_generations(num_generations)
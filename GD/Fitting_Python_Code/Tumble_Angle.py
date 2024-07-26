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


# class CustomTumbleAngleDistribution:
#     def __init__(self, a=0, b=np.pi):
#         self.a = a
#         self.b = b
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.max_pdf = self.compute_max_pdf()
#
#     def pdf(self, x):
#         """
#         Probability Density Function for the tumble angle.
#         :param x: Angle in Rad.
#         :return: P(x) = 0.5*(1+cos(x))*sin(x)
#         """
#         return 0.5 * (1 + torch.cos(x)) * torch.sin(x)
#
#     def compute_max_pdf(self):
#         """
#         Compute the maximum value of the PDF for use in rejection sampling.
#         """
#         x_vals = torch.linspace(self.a, self.b, 100000, device=self.device)
#         pdf_vals = self.pdf(x_vals)
#         return torch.max(pdf_vals).item()
#
#     def rvs(self, size=1, batch_size_factor=100):
#         """
#         Random variate generation for the custom distribution using rejection sampling.
#         :param size: Number of samples to generate.
#         :param batch_size_factor: Factor to determine the batch size for sampling.
#         :return: Tensor of samples in radians.
#         """
#         samples = torch.empty(size, device=self.device)
#         count = 0
#         batch_size = size * batch_size_factor  # Generate a larger batch to improve efficiency
#
#         while count < size:
#             # Generate uniform random numbers between a and b
#             x = torch.rand(batch_size, device=self.device) * (self.b - self.a) + self.a
#             # Generate uniform random numbers between 0 and the maximum of the PDF
#             y = torch.rand(batch_size, device=self.device) * self.max_pdf
#             # Accept the samples that are under the PDF curve
#             accept = y < self.pdf(x)
#             num_accept = torch.sum(accept).item()
#
#             if num_accept > 0:
#                 # Calculate the number of samples to add
#                 num_to_add = min(num_accept, size - count)
#                 samples[count:count + num_to_add] = x[accept][:num_to_add]
#                 count += num_to_add
#
#         return samples
#
#
# class AngleGenerator_cuda:
#     def __init__(self):
#         self.distribution = CustomTumbleAngleDistribution(a=0, b=np.pi)
#
#     def tumble_angle_function_cuda(self, size=1):
#         random_angle_rad = self.distribution.rvs(size)
#         return torch.rad2deg(random_angle_rad).to(self.distribution.device)
#
#
# # Function to test multiple angle generations
# def test_multiple_generations(num_generations):
#     angle_generator = AngleGenerator_cuda()
#     start_time = time.time()
#
#     angles = angle_generator.tumble_angle_function_cuda(num_generations)
#
#     end_time = time.time()
#     print(
#         f"Generated {num_generations} angles in {end_time - start_time:.4f} seconds with mean {torch.mean(angles).item():.2f}")
#
#
# # Example usage
# num_generations = 100000  # Number of times to generate angles
# test_multiple_generations(num_generations)

class CustomTumbleAngleDistribution:
    def __init__(self, a=0, b=np.pi, num_points=1000000):
        self.a = a
        self.b = b
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_points = num_points
        self.x_vals = torch.linspace(self.a, self.b, self.num_points, device=self.device)
        self.pdf_vals = self.pdf(self.x_vals)
        self.cdf_vals = self.compute_cdf()

    def pdf(self, x):
        """
        Probability Density Function for the tumble angle.
        :param x: Angle in Rad.
        :return: P(x) = 0.5 * (1 + cos(x)) * sin(x)
        """
        return 0.5 * (1 + torch.cos(x)) * torch.sin(x)

    def compute_cdf(self):
        """
        Compute the Cumulative Distribution Function (CDF).
        :return: The normalized cdf with bins = num_points.
        """
        cdf_vals = torch.cumsum(self.pdf_vals, dim=0)
        cdf_vals /= cdf_vals[-1].clone()  # Normalize the CDF
        return cdf_vals

    def interpolate_inverse_cdf(self, u):
        """
        A linear interpolation for the inverse CDF.
        :param u: Uniform random samples in [0, 1].
        :return: Interpolated samples corresponding to the CDF values.
        """
        # Find the indices in self.cdf_vals where the values of u would be inserted to maintain order.
        indices = torch.searchsorted(self.cdf_vals, u)
        # Clamp the indices to ensure they are within the valid range (1 to len(self.x_vals) - 1)
        indices = torch.clamp(indices, 1, len(self.x_vals) - 1)

        # Get the x values corresponding to the indices and their preceding values.
        x1 = self.x_vals[indices - 1]
        x2 = self.x_vals[indices]
        # Get the corresponding CDF values for the indices and their preceding values.
        y1 = self.cdf_vals[indices - 1]
        y2 = self.cdf_vals[indices]
        # Calculate the slope for linear interpolation.
        slope = (y2 - y1) / (x2 - x1)
        # Calculate the intercept for linear interpolation.
        intercept = y1 - slope * x1
        # Perform the linear interpolation to find the corresponding x values for the given u.
        return (u - intercept) / slope

    def rvs(self, size=1):
        """
        Random variate generation for the custom distribution using inverse transform sampling.
        :param size: Number of samples to generate.
        :return: Tensor of samples in radians.
        """
        uniform_samples = torch.rand(size, device=self.device)
        samples = self.interpolate_inverse_cdf(uniform_samples)
        return samples


class AngleGenerator_cuda(CustomTumbleAngleDistribution):
    def __init__(self):
        super().__init__(a=0, b=np.pi)

    def tumble_angle_function_cuda(self, size=1):
        random_angle_rad = self.rvs(size)
        return torch.rad2deg(random_angle_rad).to(self.device)

# # Function to test multiple angle generations
# def test_multiple_generations(num_generations):
#     angle_generator = AngleGenerator_cuda()
#     start_time = time.time()
#
#     angles = angle_generator.tumble_angle_function_cuda(size=num_generations)
#
#     end_time = time.time()
#     print(
#         f"Generated {num_generations} angles in {end_time - start_time:.4f}
#         seconds with mean {torch.mean(angles).item():.2f}"
#         )


# Function to pre-generate a large batch of angles
def pre_generate_angles(total_angles):
    angle_generator = AngleGenerator_cuda()
    return angle_generator.tumble_angle_function_cuda(size=total_angles)

# # CUDA Kernel Function
# @torch.jit.script
# def cuda_kernel(pre_generated_angles, results, num_loops, num_samples_per_loop, dt, Ptum, num_blocks,
#                 threads_per_block):
#     """
#     The kernel is launched with the specified number of blocks and threads per block using
#     torch.jit.script for JIT compilation.
#     https://pytorch.org/docs/stable/jit_language_reference.html#language-reference
#     """
#     thread_id = torch.cuda.threadIdx.x + torch.cuda.blockIdx.x * torch.cuda.blockDim.x
#     if thread_id < num_loops:
#         for t in range(1, 1000, int(dt)):
#             R_rt = torch.rand(1, device='cuda').item()
#             if R_rt < Ptum:
#                 Next_Angle = pre_generated_angles[thread_id * num_samples_per_loop + t]
#                 results[thread_id, t] = Next_Angle
#             else:
#                 pass


# # Function to run parallel loops using pre-generated angles
# def run_parallel_loops(pre_generated_angles, num_loops, num_samples_per_loop, dt, Ptum):
#     total_samples = num_loops * num_samples_per_loop
#     assert pre_generated_angles.shape[0] >= total_samples, "Not enough pre-generated angles."
#
#     # Predefine tensor to store results.
#     results = torch.zeros((num_loops, 1000), device='cuda')
#
#     # Generate random numbers for the loop decision in parallel
#     R_rt = torch.rand((num_loops, 1000), device='cuda')
#
#     # Parallelize the outer loop using PyTorch operations
#     for loop_idx in range(num_loops):
#         Ptum_local = Ptum  # Use a local copy of Ptum to dynamically change it
#         for t in range(0, 1000):
#             if R_rt[loop_idx, t] < Ptum_local:
#                 angle_idx = loop_idx * num_samples_per_loop + t
#                 if angle_idx < pre_generated_angles.size(0):
#                     results[loop_idx, t] = pre_generated_angles[angle_idx]
#                 else:
#                     results[loop_idx, t] = 0  # Handle out-of-bounds access gracefully
#             else:
#                 results[loop_idx, t] = 0  # Replace with appropriate logic
#             # Update Ptum_local dynamically if needed
#         print(f"End of loop {loop_idx}")
#
#     return results


# # Define the inner loop as a top-level function
# def inner_loop(loop_idx, R_rt, pre_generated_angles, results, num_samples_per_loop, Ptum):
#     Ptum_local = Ptum  # Use a local copy of Ptum to dynamically change it
#     for t in range(0, 1000):
#         if R_rt[loop_idx, t] < Ptum_local:
#             angle_idx = loop_idx * num_samples_per_loop + t
#             if angle_idx < pre_generated_angles.size(0):
#                 results[loop_idx, t] = pre_generated_angles[angle_idx]
#             else:
#                 results[loop_idx, t] = 0  # Handle out-of-bounds access gracefully
#         else:
#             pass
#     print(f"End of loop {loop_idx}")
#
#
# # Function to run parallel loops using pre-generated angles
# def run_parallel_loops(pre_generated_angles, num_loops, num_samples_per_loop, dt, Ptum):
#     total_samples = num_loops * num_samples_per_loop
#     assert pre_generated_angles.shape[0] >= total_samples, "Not enough pre-generated angles."
#
#     # Predefine tensor to store results.
#     results = torch.zeros((num_loops, 1000), device='cuda')
#
#     # Generate random numbers for the loop decision in parallel
#     R_rt = torch.rand((num_loops, 1000), device='cuda')
#
#     # Use shared memory for results to be accessed by multiple processes
#     results.share_memory_()
#
#     # Parallelize the outer loop using torch.multiprocessing.Pool
#     # Utilizes mp.Pool and pool.starmap to parallelize the outer loop.
#     with torch.multiprocessing.Pool() as pool:
#         pool.starmap(inner_loop,
#                      [(loop_idx, R_rt, pre_generated_angles, results, num_samples_per_loop, Ptum) for loop_idx in
#                       range(num_loops)])
#
#     return results


# Function to run parallel loops using pre-generated angles
def run_parallel_loops(pre_generated_angles, num_loops, num_samples_per_loop, dt, Ptum):
    total_samples = num_loops * num_samples_per_loop
    assert pre_generated_angles.shape[0] >= total_samples, "Not enough pre-generated angles."

    # Predefine tensor to store results.
    results = torch.zeros((num_loops, num_samples_per_loop), device='cuda')

    # Generate random numbers for the loop decision in parallel
    R_rt = torch.rand((num_loops, num_samples_per_loop), device='cuda')

    # Create a tensor for indices
    indices = torch.arange(1000, device='cuda').repeat(num_loops, 1)
    loop_indices = torch.arange(num_loops, device='cuda').view(-1, 1).repeat(1, 1000) * num_samples_per_loop

    angle_indices = loop_indices + indices

    # Handle out-of-bounds access gracefully
    valid_indices = angle_indices < total_samples

    # Set results based on valid indices and R_rt condition
    results[valid_indices & (R_rt < Ptum)] = pre_generated_angles[angle_indices[valid_indices & (R_rt < Ptum)]]

    return results

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # Required for CUDA tensors

    # Example usage
    num_loops = 10000  # Number of parallel loops
    num_samples_per_loop = 10000  # Number of samples per loop
    total_angles_needed = num_loops * num_samples_per_loop

    # Pre-generate angles
    start_time = time.time()
    pre_generated_angles = pre_generate_angles(total_angles_needed)
    end_time = time.time()
    print(f"Pre-generated {total_angles_needed} angles in {end_time - start_time:.4f} seconds")

    # Run parallel loops using pre-generated angles
    dt = 0.1
    Ptum = 0.3  # Example probability threshold
    start_time = time.time()
    results = run_parallel_loops(pre_generated_angles, num_loops, num_samples_per_loop, dt, Ptum)
    end_time = time.time()
    print(f"Ran {num_loops} loops in {end_time - start_time:.4f} seconds")

    # Calculate the mean
    mean_angle = torch.mean(results[results != 0])
    print(f"Mean angle: {mean_angle:.2f} degrees")

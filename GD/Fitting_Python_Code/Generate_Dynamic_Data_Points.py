# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
import torch
import logging


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

    def simulate_bacterial_movement_cuda(self):
        # Initialize accumulators for sum of velocities and count
        total_velocity_sum = torch.tensor(0.0, device='cuda')
        total_count = torch.tensor(0, device='cuda')

        # Initializing total time steps.
        time_steps = torch.arange(0, 1000, self.dt, device='cuda')
        num_steps = time_steps.size(0)

        # Initialize tensors for pos and ang for all iterations.
        position = torch.zeros((num_steps, self.max_iter), device='cuda')
        ang = torch.zeros((num_steps, self.max_iter), device='cuda')

        # Starting Position.
        position[0, :] = torch.full((self.max_iter,), self.pos, device='cuda')

        # Apply model constraints
        position = torch.clamp(position, 0, self.nl * self.DL)

        # Starting Angle: All angles start at 'self.Angle'.
        ang[0, :] = torch.full((self.max_iter,), self.Angle, device='cuda').float()

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

        # Mask to track active bacteria
        active_mask = torch.ones(self.max_iter, dtype=torch.bool, device='cuda')

        for t_idx, t in enumerate(time_steps):
            # Tensors inside the for loop are vectorized and in parallel.
            if t_idx % 100 == 0:
                logging.info(f"Step {t_idx}/{num_steps}: Current time = {t.item()}")

            if active_mask.any():
                # Filter positions and angles using active_mask
                active_positions = position[t_idx, active_mask]

                # Calculate boundary_mask to identify bacteria reaching the end of the deme
                boundary_mask = active_positions >= (self.pos_ini + self.DL)

                # Handle bacteria that have reached the boundary.
                if boundary_mask.any():
                    # Calculate the distance traveled for these bacteria.
                    distance_travelled = active_positions[boundary_mask] - self.pos_ini
                    total_time = t

                    # Calculate the average velocity for these bacteria.
                    Calculated_Ave_Vd = distance_travelled / total_time

                    # Accumulate sum of velocities and count
                    total_velocity_sum += Calculated_Ave_Vd.sum()
                    total_count += Calculated_Ave_Vd.numel()

                # Update the active mask: remove bacteria that have reached the boundary
                active_mask[active_mask.clone()] = ~boundary_mask

                if active_mask.any():
                    # The .long() method ensures the tensor is of integer type, which is necessary for indexing.
                    i = (position[t_idx, active_mask] // self.DL).long()

                    # Direction for moving up or down gradient.
                    # The boolean is true if it is moving down the gradient.
                    direction_condition = (90 <= ang[t_idx, active_mask]) & (ang[t_idx, active_mask] < 270)

                    # Calculate Ptum using torch.where
                    Ptum[t_idx, active_mask] = torch.where(
                        direction_condition,
                        self.dt * torch.exp(-self.d + self.alpha * Rtroc_tensor[i]),
                        self.dt * torch.exp(-self.d - self.alpha * Rtroc_tensor[i])
                    ).float()  # Convert to float to match Ptum's dtype.

                    # Tumbling condition
                    tumble_mask = R_rt[t_idx, active_mask] < Ptum[t_idx, active_mask]
                    # Update angles based on tumbling condition.
                    # Condition: tumble_mask
                    # If the condition is True, update the angle with Next_Angle.
                    # If the condition is False, the bacteria does not tumble and the angle is left unchanged.
                    ang[t_idx, active_mask] = torch.where(
                        tumble_mask,
                        (ang[t_idx, active_mask] + Next_Angle[t_idx, active_mask]) % 360,
                        ang[t_idx, active_mask]
                    )

                    # Running condition
                    run_mask = ~tumble_mask
                    # Update position based on running condition
                    Dot_Product = torch.cos(ang[t_idx, active_mask] * (torch.pi / 180))  # Convert ang to radians manually
                    # Condition: run_mask
                    # If the condition is True, update the position with Vo_max*Dot_Product*dt.
                    # If the condition is False, the bacteria does not move and the position is left unchanged.
                    position[t_idx, active_mask] = torch.where(
                        run_mask,
                        position[t_idx, active_mask] + self.dt * self.Vo_max * Dot_Product,
                        position[t_idx, active_mask]
                    )

                    if t_idx < num_steps - 1:
                        # Set position and ang for the next time step
                        position[t_idx + 1, active_mask] = position[t_idx, active_mask]
                        ang[t_idx + 1, active_mask] = ang[t_idx, active_mask]

            # Remove finished bacteria from further calculations
            if not active_mask.any():
                logging.info("The break condition is met.")
                break

        # Final calculation for the remaining iterations.
        # Calculate boundary_mask to identify bacteria reaching the end of the deme
        final_remaining_positions = position[-1, active_mask]
        if active_mask.any():
            Calculated_Ave_Vd = (final_remaining_positions - self.pos_ini) / (time_steps[-1])
            if Calculated_Ave_Vd.numel() > 0:
                # Accumulate sum of velocities and count of the remaining iterations.
                total_velocity_sum += Calculated_Ave_Vd.sum()
                total_count += Calculated_Ave_Vd.numel()

        logging.info(f"total_velocity_sum = {total_velocity_sum}\ntotal_count = {total_count}")

        if total_count > 0:
            mean_results = total_velocity_sum / total_count
        else:
            logging.info("No velocities were calculated.")
            mean_results = float('nan')  # Return NaN if no valid velocities were calculated
            return mean_results

        # Return the results
        return mean_results.item()

# Import the Tumble Angle Module
from Tumble_Angle import AngleGenerator_cuda
import torch
import logging
from Generate_Dynamic_Data_Points import Norm_Vd_Mean_Data_Generator
from abc import ABC, abstractmethod
import logging


# Abstract Base Class for Data Generators
class BaseDataGenerator(ABC):
    @abstractmethod
    def generate_data(self, max_iter):
        pass


# Specific Data Generator Implementation for Norm_Vd_Mean_Data_Generator
class NormMeanDataGenerator(BaseDataGenerator):
    """
    This class inherits the format of BaseDataGenerator and is used
    for the data generator 'Norm_Vd_Mean_Data_Generator'.
    """
    def __init__(self, *args, **kwargs):
        # Initialize with parameters specific to Norm_Vd_Mean_Data_Generator
        self.generator = Norm_Vd_Mean_Data_Generator(*args, **kwargs)

    def generate_data(self, max_iter):
        """
        :param: max_iter: The total number of iterations to run in parallel, (the number of data points generated).
        :return: The datapoints generator specific to this system's generator method.
        """
        return self.generator.simulate_bacterial_movement_cuda(max_iter)


class Dynamic_Data_Evolving_Mean_Estimator:
    """
    Dynamic_Data_Evolving_Mean_Estimator, pronounced 'deme'.
    Instead of finding the mean of clusters, this algorithm generates dynamically changing data for each iteration of
    training using a stochastic function with an independent variable alpha.  This in turn produces a stochastic mean
    which is then compared to an optimal theoretical target value.  Gradient descent guides the ML algorithm to adjust
    the alpha accordingly so the dependent variable of the stochastic function will have an evolving mean that becomes
    more aligned with the theoretical value.
    :param: data_generator: Has to follow the same format of the BaseDataGenerator to make sure that the object used
    has a generate_data() method.
    :param: max_iter: The number of initial training iterations (data points generated) used by the ML model.
    The generate_data() method must have max_iter as an argument.
    :param: learning_rate: The learning rate for the ML algorithm during gradient descent.
    :param: step_size: The number of iteration the training loop goes through before it adjusts the parameters max_iter
    and learning_rate.
    :param: num_epochs: The total number of G.D. training iterations used by the ML algorithm.
    :param: theoretical_val: The target the model is trying to predict using the normalized data mean produced by
    generate_data().
    :param: alpha: The independent variable the ML model is optimizing for a stochastic function whose mean
    """
    def __init__(self, data_generator: BaseDataGenerator, num_epochs, learning_rate, theoretical_val, alpha, max_iter,
                 step_size):

        self.data_generator = data_generator
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.theoretical_val = torch.tensor(theoretical_val, device=self.device)
        self.alpha = torch.tensor(alpha, requires_grad=True, device=self.device)

        # Remove the bias term from the linear layer to avoid interference with the intrinsic
        # standard error of the dynamic mean.  y = W * x, (no 'b').
        self.model = torch.nn.Linear(1, 1, bias=False).cuda()

        # Fix the weight to 1 and prevent it from being updated to limit resources.
        # The weights of the linear layer won't interfere with optimizing alpha,
        # but it will keep the data in the GPU to perform calculations with the loss function.
        with torch.no_grad():
            self.model.weight.fill_(1.0)  # Set weight to 1.
            self.model.weight.requires_grad = False  # Disable gradient updates.

        # Initialize the optimizer with the model parameters and learning rate.
        # The optimizer will handle the update of alpha based on the computed gradients.
        self.optimizer = torch.optim.SGD([self.alpha], lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()  # Mean Squared Error Loss function.
        # The learning rate scheduler will reduce the learning rate by 0.5 every 20 epochs.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train(self):
        for epoch in range(self.num_epochs):
            # A new epoch of data is generated for every instance of the training loop.
            data = self.data_generator.generate_data()  # This is a tensor on the GPU

            # The tensor is passed through the model to compute the output using the linear layer.
            output = self.model(data.unsqueeze(-1))  # Add dimension if needed for linear layer.

            # Computing the MSE of the dynamic data point mean compared to the theoretical_val.
            # L(α)=MSE(∑vj(α),vd)
            loss = self.loss_function(output, self.theoretical_val)

            # Backward pass: Compute gradients
            self.optimizer.zero_grad()  # Reset previous gradient to prevent incorrect update.

            # Compute the gradient of the loss function with respect to the parameters with requires_grad=True,
            # in this case the alpha value.
            # ∂L(α)/∂α = 2/n*∑(vα(j)−vd)*∂vα(j)/∂α
            loss.backward()

            # Update the model's parameters (alpha) using gradient descent.
            # α_k_+_1 = α_k - γ*∂L(α)/∂α
            self.optimizer.step()  # This internally updates alpha based on the gradients and learning rate.

            # Scheduler step: Adjust the learning rate according to the schedule.
            self.scheduler.step()

            # Logging every 20 epochs.
            if epoch % 20 == 0:
                # Double the number of data samples generated every 20 epochs.
                self.max_iter *= 2
                if self.max_iter > 20000:
                    self.max_iter = 20000
                logging.info(f"Epoch {epoch}, Loss: {loss.item()}, Alpha: {self.alpha.item()}")

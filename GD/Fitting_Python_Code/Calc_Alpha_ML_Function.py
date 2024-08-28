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
    def generate_data(self):
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

    def generate_data(self):
        """
        :return: The datapoints generator specific to this system's generator method.
        """
        return self.generator.simulate_bacterial_movement_cuda()


class Dynamic_Data_Evolving_Mean_Estimator:
    """
    Dynamic_Data_Evolving_Mean_Estimator, pronounced 'deme'.
    The data_generator used has to follow the same format of the BaseDataGenerator
    to make sure that the object used has a generate_data() method.
    """
    def __init__(self, data_generator: BaseDataGenerator, num_epochs, learning_rate, theoretical_val, alpha):
        self.data_generator = data_generator
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.theoretical_val = theoretical_val
        self.alpha = torch.tensor(alpha, requires_grad=True, device='cuda')  # Alpha is the feature to optimize.

        # Remove the bias term from the linear layer to avoid interference with the intrinsic
        # standard error of the dynamic mean.
        self.model = torch.nn.Linear(1, 1, bias=False).cuda()
        # Initialize the optimizer, loss function, and scheduler.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train(self):
        for epoch in range(self.num_epochs):
            # A new epoch of data is generated for every instance of the training loop.
            data = self.data_generator.generate_data()  # This is a tensor on the GPU

            # The tensor is passed through the model to compute the output using the linear layer.
            output = self.model(data)

            loss = self.loss_function(output, self.theoretical_val)

            # Backward pass: Compute gradients
            self.optimizer.zero_grad()  # Reset previous gradient to prevent incorrect update.
            # Compute the gradient of the loss object with respect to the weights but no bias in the model.
            loss.backward()

            # Update alpha using custom gradient descent.
            # The .grad attribute of each parameter with the gradient.
            with torch.no_grad():
                for param in output.parameters():
                    self.alpha -= self.learning_rate * param.grad

            # Apply learning rate schedule every 20 epochs.
            if epoch % 20 == 0:
                self.learning_rate /= (epoch // 20 + 1)
                self.optimizer.param_groups[0]['lr'] = self.learning_rate

            # Logging every 20 epochs.
            if epoch % 20 == 0:
                logging.info(f"Epoch {epoch}, Loss: {loss.item()}, Alpha: {self.alpha.item()}")

# LOAD MODULES

# Standard library
from typing import Callable, Optional
import math

# Proprietary
from src.methods.utils import (
    # Standard imports
    ContinuousCATENN,
)

# Third party
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(ContinuousCATENN):
    """
    Multilayer Perceptron (MLP) class that inherits from the ContinuousCATENN class.

    This class represents a Multilayer Perceptron, which is a type of neural network. It inherits 
    from the ContinuousCATENN class.
    """
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.01,
        regularization_l2: float = 0.0,
        batch_size: int = 64,
        num_steps: int = 1000,
        num_layers: int = 2,
        binary_outcome: bool = False,
        hidden_size: int = 32,
        verbose: bool = False,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            input_size (int): The size of the input to the network.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            regularization_l2 (float, optional): The L2 regularization strength. Defaults to 0.0.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            num_steps (int, optional): The number of training steps. Defaults to 1000.
            num_layers (int, optional): The number of layers in the network. Defaults to 2.
            binary_outcome (bool, optional): Whether the outcome is binary. Defaults to False.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 32.
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        # Ini the super module
        super(MLP, self).__init__(
            input_size=input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_l2=regularization_l2,
            binary_outcome=binary_outcome,
            hidden_size=hidden_size,
            verbose=verbose,
            activation=activation,
        )

        # Save architecture settings
        self.num_layers = num_layers

        # Structure
        # Shared layers
        self.layers = nn.Sequential(nn.Linear(self.input_size + 1, self.hidden_size))
        self.layers.append(self.activation)
        # Add additional layers
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation)
        # Add output layer
        self.layers.append(nn.Linear(self.hidden_size, 1))
        # Sigmoid activation if binary is True
        if self.binary_outcome == True:
            self.layers.append(nn.Sigmoid())

    def forward(
        self, 
        x: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the network using the input tensor `x` and dose tensor `d`.

        This method takes as input a tensor `x` representing the input data and a tensor `d`.
        """
        x = torch.cat((x, d), dim=1)

        # Feed through layers
        x = self.layers(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step using the given batch of data.
        """
        x, y, d = batch

        y_hat = self(x, d)

        loss_mse = F.mse_loss(y, y_hat)

        return loss_mse

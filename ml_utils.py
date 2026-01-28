"""Utility functions and classes for neural network training and weight trajectory tracking.

This module provides:
- SimpleNN: A simple feedforward neural network
- train_network: Function to train a network and track weight trajectories
- get_weight_vector: Extract all weights from a model as a flat vector
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    """Simple feedforward neural network with 2 hidden layers.
    
    Architecture:
    - Input layer: 2 features
    - Hidden layer 1: 8 neurons with ReLU activation
    - Output layer: 2 classes
    """
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 2)
            
        Returns:
            Output tensor of shape (batch_size, 2)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_weight_vector(model):
    """Extract all weights and biases from a model as a flat numpy array.
    
    Args:
        model: PyTorch model
        
    Returns:
        Flattened numpy array containing all model parameters
    """
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)


def train_network(model, dataloader, epochs=100, lr=0.1):
    """Train a network and return loss history and weight trajectory.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader containing training data
        epochs: Number of training epochs (default: 100)
        lr: Learning rate (default: 0.1)
        
    Returns:
        tuple: (loss_history, weight_trajectory)
            - loss_history: List of average loss values per epoch
            - weight_trajectory: List of weight vectors (as numpy arrays) at each epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    loss_history = []
    weight_trajectory = []
    
    # Store initial weights
    weight_trajectory.append(get_weight_vector(model))
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Store current weights
        weight_trajectory.append(get_weight_vector(model))
    
    return loss_history, weight_trajectory


def create_dataloader(X, y, batch_size=32, shuffle=False):
    """Create a PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Input features as numpy array
        y: Labels as numpy array
        batch_size: Batch size for DataLoader (default: 32)
        shuffle: Whether to shuffle data (default: False)
        
    Returns:
        DataLoader object
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

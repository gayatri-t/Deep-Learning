import torch
import torch.nn as nn
from torch.optim import SGD

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class LossCalculator:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.total_loss = 0.0
        self.total_samples = 0

    def calculate_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    def train_model(self, input_data, target_data, num_epochs=20, num_trainings=10):
        for training in range(num_trainings):
            self.optimizer = SGD(self.model.parameters(), lr=0.01)  # Reset optimizer for each training

            for epoch in range(num_epochs):
                self.model.train()
                self.optimizer.zero_grad()

                predictions = self.model(input_data)
                loss = self.calculate_loss(predictions, target_data)

                loss.backward()
                self.optimizer.step()

                # Update total loss and samples
                self.total_loss += loss.item()
                self.total_samples += 1

        average_loss = self.total_loss / self.total_samples
        return average_loss



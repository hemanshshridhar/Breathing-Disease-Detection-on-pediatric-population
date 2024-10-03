import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Convolutional layer to extract features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()  # Activation function
        
        # Fully connected layer to reduce dimensionality
        self.fc1 = nn.Linear(in_features=8 * 798, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)  # Final layer for 4-class output
        
        # Softmax for final probability output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: [batch_size, 1, 798]
        
        # Apply convolutional layer
        x = self.conv1(x)  # [batch_size, 8, 798]
        x = self.relu(x)   # Apply ReLU activation
        
        # Flatten the output for fully connected layer
        x = x.view(x.size(0), -1)  # [batch_size, 8 * 798]
        
        # Apply fully connected layers
        x = self.fc1(x)  # [batch_size, 128]
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # [batch_size, 4]
        
        # Apply softmax to get class probabilities
        output = self.softmax(x)  # [batch_size, 4]
        
        return output

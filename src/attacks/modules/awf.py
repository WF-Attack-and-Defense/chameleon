import torch
import torch.nn as nn


class AWFNet(nn.Module):
    def __init__(self, length: int, num_classes: int = 100, in_channels: int = 1):
        super(AWFNet, self).__init__()
        self.length = length
        self.num_classes = num_classes
        
        dropout = 0.1
        filters = 32
        kernel_size = 5
        stride_size = 1
        pool_size = 4
        
        # Input dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Block 1
        self.block1_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride_size,
            padding=0  # valid padding
        )
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        
        # Block 2
        self.block2_conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride_size,
            padding=0  # valid padding
        )
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        
        # Block 3
        self.block3_conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride_size,
            padding=0  # valid padding
        )
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        
        # Flatten and dense layer
        # Calculate the flattened size after all conv and pooling operations
        # For Conv1d with valid padding: output_size = (input_size - kernel_size + 1)
        # For MaxPool1d with valid padding: output_size = (input_size - kernel_size) // stride + 1
        self.flatten_size = self._calculate_flatten_size(length, kernel_size, pool_size, filters)
        self.dense = nn.Linear(self.flatten_size, num_classes)
    
    def _calculate_flatten_size(self, length, kernel_size, pool_size, filters):
        """
        Calculate the flattened size after all conv and pooling operations.
        """
        size = length
        
        # Block 1: conv then pool
        size = size - kernel_size + 1  # conv1 with valid padding
        size = (size - pool_size) // pool_size + 1  # pool1 with valid padding
        
        # Block 2: conv then pool
        size = size - kernel_size + 1  # conv2
        size = (size - pool_size) // pool_size + 1  # pool2
        
        # Block 3: conv then pool
        size = size - kernel_size + 1  # conv3
        size = (size - pool_size) // pool_size + 1  # pool3
        
        # Multiply by number of filters
        return max(1, size) * filters
    
    def forward(self, x):
        # Input shape: (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dimension if needed
        
        # Input dropout
        x = self.dropout(x)
        
        # Block 1
        x = self.block1_conv(x)
        x = nn.functional.relu(x)
        x = self.block1_pool(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = nn.functional.relu(x)
        x = self.block2_pool(x)
        
        # Block 3
        x = self.block3_conv(x)
        x = nn.functional.relu(x)
        x = self.block3_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layer (output)
        x = self.dense(x)
        return x

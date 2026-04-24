import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_padding(kernel_size, dilation):
    """Calculate padding for causal convolution"""
    return (kernel_size - 1) * dilation


class DilatedBasicBlock1D(nn.Module):
    """
    A one-dimensional basic residual block with dilations.
    Based on the original Keras implementation.
    """
    def __init__(self, in_channels, out_channels, stage, block, dilations=(1, 1), kernel_size=3):
        super(DilatedBasicBlock1D, self).__init__()
        
        # Determine stride: if block == 0 and stage > 0, stride = 2, else stride = 1
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2
        
        # First conv layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=causal_padding(kernel_size, dilations[0]),
            dilation=dilations[0],
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-5)
        self.relu1 = nn.ReLU()
        
        # Second conv layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=causal_padding(kernel_size, dilations[1]),
            dilation=dilations[1],
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-5)
        
        # Shortcut connection (only for first block of each stage, i.e., block == 0)
        if block == 0:
            self.shortcut_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
            self.shortcut_bn = nn.BatchNorm1d(out_channels, eps=1e-5)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
        
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        # First conv block
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        
        # Second conv block
        y = self.conv2(y)
        y = self.bn2(y)
        
        # Shortcut connection
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut)
        else:
            shortcut = x
        
        # Ensure y and shortcut have the same length dimension
        # Crop y to match shortcut if needed (causal padding can make y longer)
        if y.size(2) != shortcut.size(2):
            # Crop from the end to match shortcut size
            diff = y.size(2) - shortcut.size(2)
            if diff > 0:
                y = y[:, :, :-diff]
            elif diff < 0:
                # If y is shorter, pad it (shouldn't happen with causal padding, but handle it)
                pad_size = -diff
                y = F.pad(y, (0, pad_size), mode='constant', value=0)
        
        y = y + shortcut
        y = self.relu2(y)
        return y


class VarCNNNet(nn.Module):
    """
    VarCNN Network based on ResNet18 architecture with dilated convolutions.
    Matches the original Keras implementation.
    """
    def __init__(self, length: int, num_classes: int = 100, in_channels: int = 1):
        super(VarCNNNet, self).__init__()
        self.length = length
        self.num_classes = num_classes
        
        # Initial conv block (layer 1)
        # ZeroPadding1D(padding=3) + Conv1D(64, 7, strides=2)
        self.layer1_padding = nn.ConstantPad1d(padding=(3, 3), value=0)
        self.layer1_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            bias=False
        )
        self.layer1_bn = nn.BatchNorm1d(64, eps=1e-5)
        self.layer1_relu = nn.ReLU()
        # MaxPooling1D(3, strides=2, padding='same')
        self.layer1_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers: 4 stages with [2, 2, 2, 2] blocks each
        blocks = [2, 2, 2, 2]
        features = 64  # Starting with 64 features
        
        self.res_layers = nn.ModuleList()
        for stage_id, iterations in enumerate(blocks):
            stage_modules = nn.ModuleList()
            
            # First block of stage: dilations=(1, 2), may double features
            if stage_id > 0:
                # Double features at start of each stage (except first stage)
                out_features = features * 2
            else:
                out_features = features
            
            # First block: dilations=(1, 2)
            stage_modules.append(
                DilatedBasicBlock1D(
                    in_channels=features,
                    out_channels=out_features,
                    stage=stage_id,
                    block=0,
                    dilations=(1, 2),
                    kernel_size=3
                )
            )
            
            # Remaining blocks in stage: dilations=(4, 8)
            features = out_features
            for block_id in range(1, iterations):
                stage_modules.append(
                    DilatedBasicBlock1D(
                        in_channels=features,
                        out_channels=features,
                        stage=stage_id,
                        block=block_id,
                        dilations=(4, 8),
                        kernel_size=3
                    )
                )
            
            self.res_layers.append(stage_modules)
            # Features double for next stage
            features = out_features
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final output layer
        # Final features should be 512 (64 * 2^3)
        self.fc = nn.Linear(features, num_classes)
    
    def forward(self, x):
        # Input shape: (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dimension if needed
        
        # Layer 1: Initial conv block
        x = self.layer1_padding(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)
        x = self.layer1_pool(x)
        
        # Residual layers
        for stage_modules in self.res_layers:
            for block in stage_modules:
                x = block(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Output layer
        x = self.fc(x)
        return x

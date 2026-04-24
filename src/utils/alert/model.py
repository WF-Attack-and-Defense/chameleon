"""ALERT defense: DF-style discriminator and MLP generator (see ``defenses.alert`` for training)."""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Discriminator(nn.Module):
    """DF-style CNN used as the surrogate classifier in ALERT."""

    def __init__(self, classes_num):
        super(Discriminator, self).__init__()

        self.classes_num = classes_num
        classes = classes_num
        filter_num = ['None', 32, 64, 128, 256]
        kernel_size = ['None', 8, 8, 8, 8]
        pool_stride_size = ['None', 4, 4, 4, 4]
        pool_size = ['None', 8, 8, 8, 8]
        self.block1 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=1, out_channels=filter_num[1], kernel_size=kernel_size[1]),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1]),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(),
            nn.ConstantPad1d((2, 2), 0),
            nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1]),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2]),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2]),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 3), 0),
            nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2]),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3]),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3]),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3]),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4]),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4]),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(),
            nn.ConstantPad1d((2, 3), 0),
            nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4]),
            nn.Dropout(0.1),
        )

        # Input length (burst / trace dim) changes flatten size after the CNN stack; 2048 was for len≈2000.
        self.fc_block = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
        )

        self.fc_block1 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, classes),
        )

    def forward(self, input):
        batch_size = input.shape[0]
        block1 = self.block1(input)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        flatten = block4.view(batch_size, -1)
        fc = self.fc_block(flatten)
        fc1 = self.fc_block1(fc)
        return fc, fc1

import torch
from torch import nn


class double_convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, st=1, pad=1):
        super(double_convolution, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=st, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=st, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convolution(x)


class down_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, st=1, pad=1):
        super(down_step, self).__init__()

        self.step_result = double_convolution(in_channels, out_channels, kernel, st, pad)
        self.down = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        skip = self.step_result(x)
        return self.down(skip), skip


class upper_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, st=2, pad=1, last=False):
        super(upper_step, self).__init__()
        self.drop = nn.Dropout(p=0.5)
        self.step_result = double_convolution(in_channels, out_channels, kernel=3, st=1, pad=1)
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=kernel, stride=st, padding=pad,
                                     output_padding=1)

    def forward(self, x, down_data):
        x = self.up(x)
        x = self.drop(torch.cat([x, down_data], dim=1))
        return self.step_result(x)


class final_step(torch.nn.Module):
    def __init__(self, in_channels, num):
        super(final_step, self).__init__()
        self.result = nn.Sequential(
            nn.Conv2d(in_channels, num, kernel_size=1, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.result(x)


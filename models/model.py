from torchvision import models, transforms
import torch.nn as nn
import ssl
import torch
import torch.nn.functional as F

from models.support_functions import double_convolution, final_step, upper_step, down_step


class double_conv(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class down_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(down_step, self).__init__()

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv = double_conv(in_channels, out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(self.pool(x))


class up_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(up_step, self).__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, from_up_step, from_down_step):
        upsampled = self.up(from_up_step)
        x = torch.cat([from_down_step, upsampled], dim=1)
        return self.conv(x)


class out_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(out_conv, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(torch.nn.Module):
    def __init__(self, n_channels, n_classes, init_features, depth, image_size, fourier_params=None):
        super(UNet2D, self).__init__()

        self.features = init_features
        self.depth = depth

        self.fourier_params = fourier_params

        self.down_path = torch.nn.ModuleList()
        self.down_path.append(double_conv(n_channels, self.features, self.features))
        for i in range(1, self.depth):
            self.down_path.append(down_step(self.features, 2 * self.features))
            self.features *= 2

        self.up_path = torch.nn.ModuleList()
        for i in range(1, self.depth):
            self.up_path.append(up_step(self.features, self.features // 2))
            self.features //= 2
        self.out_conv = out_conv(self.features, n_classes)

    def forward_down(self, input):
        downs = [input]
        for down_step in self.down_path:
            downs.append(down_step(downs[-1]))

        return downs

    def forward_up(self, downs):
        current_up = downs[-1]
        for i, up_step in enumerate(self.up_path):
            current_up = up_step(current_up, downs[-2 - i])

        return current_up

    def forward(self, x):
        if self.fourier_params is not None:
            x = self.fl(x)

        downs = self.forward_down(x)
        up = self.forward_up(downs)

        return self.out_conv(up)


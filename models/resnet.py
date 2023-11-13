import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_in=True, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__()
        stack = []
        stack.append(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        )
        if use_in == True:
            stack.append(nn.InstanceNorm2d(mid_channels))
        stack.append(nn.LeakyReLU(0.02, inplace=True))

        stack.append(
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=(3, 3), stride=1, padding="same"
            )
        )
        if use_in == True:
            stack.append(nn.InstanceNorm2d(out_channels))

        self.main_path = nn.Sequential(*stack)

    def forward(self, x):
        y = self.main_path(x)
        assert y.shape == x.shape
        return x + y


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, L=3, M=3, N=3) -> None:
        # Following the paper : Cross-modal Image Synthesis within Dual-Energy X-ray Security Imagery
        super().__init__()

        # f
        f_stack = []
        f_stack.append(nn.Conv2d(in_channels, out_channels=64, kernel_size=(7, 7)))
        f_stack.append(nn.InstanceNorm2d(64))
        f_stack.append(nn.LeakyReLU(0.02))
        f_stack.append(nn.Conv2d(64, 128, kernel_size=(3, 3)))
        f_stack.append(nn.InstanceNorm2d(128))
        f_stack.append(nn.LeakyReLU(0.02))
        f_stack.append(nn.Conv2d(128, 256, kernel_size=(3, 3)))
        f_stack.append(nn.InstanceNorm2d(256))
        f_stack.append(nn.LeakyReLU(0.02))
        for _ in range(L):
            f_stack.append(ResidualBlock(256, 256))

        self.f = nn.Sequential(*f_stack)

        # g
        g_stack = []
        for _ in range(M):
            g_stack.append(ResidualBlock(256, 256))
        self.g = nn.Sequential(*g_stack)

        # h
        h_stack = []
        for _ in range(N):
            h_stack.append(ResidualBlock(256, 256))
        h_stack.append(nn.ConvTranspose2d(256, 128, kernel_size=(3, 3)))
        h_stack.append(nn.InstanceNorm2d(128))
        h_stack.append(nn.LeakyReLU(0.02))
        h_stack.append(nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)))
        h_stack.append(nn.InstanceNorm2d(64))
        h_stack.append(nn.LeakyReLU(0.02))
        h_stack.append(nn.ConvTranspose2d(64, out_channels, kernel_size=(7, 7)))

        self.h = nn.Sequential(*h_stack)

    def forward(self, x):
        x = self.f(x)
        x = self.g(x)
        x = self.h(x)

        return (F.tanh(x) + 1) * 0.5  # image range [-1, 1] -> [0, 1]

# if __name__ == "__main__":
#     im = torch.randn((1, 2, 64, 64))
#     model = ResNet(2, 1)
#     out = model(im)
#     print(out.shape)
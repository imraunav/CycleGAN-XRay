import torch
from torch import nn
import torch.nn.functional as F

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer="batch"):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__()
        if (
            norm_layer == "batch"
        ):  # no need to use bias as BatchNorm2d has affine parameters
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=(3, 3),
                padding="same",
                bias=False,
            )),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=(3, 3),
                padding="same",
                bias=False,
            )),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d((2, 2)), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.utils.spectral_norm(nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=(2, 2), stride=2
        ))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input -> (C, H, W)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_sn(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.e1 = DoubleConv(n_channels, 64)
        self.e2 = DownSample(64, 128)
        self.e3 = DownSample(128, 256)
        self.e4 = DownSample(256, 512)
        self.e5 = DownSample(512, 1024)

        self.d1 = UpSample(1024, 512)
        self.d2 = UpSample(512, 256)
        self.d3 = UpSample(256, 128)
        self.d4 = UpSample(128, 64)
        self.d5 = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)

        # Decoder
        x = self.d1(x5, x4)
        x = self.d2(x, x3)
        x = self.d3(x, x2)
        x = self.d4(x, x1)
        logits = self.d5(x)

        return logits

# if __name__ == "__main__":
#     im = torch.randn((1, 1, 128, 128))
#     model = PixelDiscriminator(1)
#     out = model(im)
#     print(out.shape)

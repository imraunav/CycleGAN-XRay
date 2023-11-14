import torch
from torch import nn
import torch.nn.functional as F

from models.unet import DoubleConv, UpSample, DownSample, OutConv


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


class UNet_sn(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.e1 = nn.utils.spectral_norm(DoubleConv(n_channels, 64))
        self.e2 = nn.utils.spectral_norm(DownSample(64, 128))
        self.e3 = nn.utils.spectral_norm(DownSample(128, 256))
        self.e4 = nn.utils.spectral_norm(DownSample(256, 512))
        self.e5 = nn.utils.spectral_norm(DownSample(512, 1024))

        self.d1 = nn.utils.spectral_norm(UpSample(1024, 512))
        self.d2 = nn.utils.spectral_norm(UpSample(512, 256))
        self.d3 = nn.utils.spectral_norm(UpSample(256, 128))
        self.d4 = nn.utils.spectral_norm(UpSample(128, 64))
        self.d5 = nn.utils.spectral_norm(OutConv(64, n_classes))

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

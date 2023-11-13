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

if __name__ == "__main__":
    im = torch.randn((1, 1, 128, 128))
    model = PixelDiscriminator(1)
    out = model(im)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=1, repeat_num=6, mask_type='no'):
        super(Generator, self).__init__()
        self.mask_type = mask_type
        c_dim = c_dim
        if c_dim == 1:
            c_dim = 0

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                      stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(
            conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                          kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                          2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7,
                      stride=1, padding=3, bias=False),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(curr_dim, 1, kernel_size=7,
                      stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, c=None):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        h = self.main(x)

        if c is not None:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, h.size(2), h.size(3))
            h = torch.cat([h, c], dim=1)
            return self.conv1(h)

        if self.mask_type == 'res':
            return self.conv1(h) + x
        if self.mask_type == 'mask':
            mask = self.conv2(h)
            return mask * self.conv1(h) + (1 - mask) * x
        if self.mask_type == 'bin_mask':
            mask = self.conv2(h) < 0.5
            return mask * self.conv1(h) + ~mask * x
        return self.conv1(h)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=256, conv_dim=64, c_dim=1, repeat_num=6):
        super(Discriminator, self).__init__()
        self.c_dim = c_dim

        layers = []
        layers.append(
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                          kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        if self.c_dim == 1:
            return out_src
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


def test0():
    x = torch.randn((5, 3, 256, 256))
    model = Generator()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test0()

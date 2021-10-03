import torch
from torch import nn
from torch.nn.modules.activation import ReLU


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, dilation=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(True),
            # Padding scales with dilation.
            nn.Conv2d(dim_out, dim_out, 3, 1, dilation, dilation=dilation),
            nn.InstanceNorm2d(dim_out))

    def forward(self, x):
        return x + self.main(x)


class DMFB(nn.Module):
    """Dense Multi-scale Fusion Block"""

    def __init__(self, dim_in):
        super().__init__()

        def padded_conv(dim_in, dim_out, kernel_size, stride=1, dilation=1):
            padding = int((kernel_size - 1) / 2) * dilation
            return nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation)

        self.c1 = padded_conv(dim_in, dim_in // 4, 3, 1)
        self.d1 = padded_conv(dim_in // 4, dim_in // 4, 3, 1, 1)  # rate = 1
        self.d2 = padded_conv(dim_in // 4, dim_in // 4, 3, 1, 2)  # rate = 2
        self.d3 = padded_conv(dim_in // 4, dim_in // 4, 3, 1, 4)  # rate = 4
        self.d4 = padded_conv(dim_in // 4, dim_in // 4, 3, 1, 8)  # rate = 8
        self.act = nn.ReLU(True)
        self.norm = nn.InstanceNorm2d(dim_in)
        self.c2 = padded_conv(dim_in, dim_in, 3, 1)  # fusion

    def forward(self, x):
        output1 = self.act(self.norm(self.c1(x)))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], 1)
        output2 = self.c2(self.act(self.norm(combine)))
        output = x + self.norm(output2)
        return output


class Menno(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, 5, 1, 2))
        layers.append(nn.ReLU())

        # Down sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, 5, 2, 2))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(nn.ReLU(True))
            curr_dim *= 2

        # Bottleneck layers
        for _ in range(repeat_num):
            layers.append(DMFB(curr_dim))

        # Up-sampling layers.
        for _ in range(2):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(curr_dim, curr_dim // 2, 3, 1, 1))
            layers.append(nn.InstanceNorm2d(curr_dim // 2))
            layers.append(nn.ReLU(True))
            curr_dim //= 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(curr_dim, 3, 5, 1, 2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 5, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.main(x)

        mask = self.conv2(h)
        return mask * self.conv1(h) + (1 - mask) * x


class DMFN(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, mask_type='no', block_type='resb', dilation=1):
        super().__init__()
        self.mask_type = mask_type

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, 5, 1, 2))
        layers.append(nn.ReLU())

        # Down sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(nn.ReLU(True))
            curr_dim *= 2

        # Bottleneck layers
        for _ in range(repeat_num):
            if block_type == 'resb':
                layers.append(ResidualBlock(curr_dim, curr_dim, dilation))
            if block_type == 'dmfb':
                layers.append(DMFB(curr_dim))

        # Up-sampling layers.
        for _ in range(2):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(curr_dim, curr_dim // 2, 3, 1, 1))
            layers.append(nn.InstanceNorm2d(curr_dim // 2))
            layers.append(nn.ReLU(True))
            curr_dim //= 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(curr_dim, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.main(x)

        if self.mask_type == 'res':
            return self.conv1(h) + x
        if self.mask_type == 'mask':
            mask = self.conv2(h)
            return mask * self.conv1(h) + (1 - mask) * x
        if self.mask_type == 'bin_mask':
            mask = self.conv2(h) < 0.5
            return mask * self.conv1(h) + ~mask * x
        return self.conv1(h)


class StarGen(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6, mask_type='no'):
        super().__init__()
        self.mask_type = mask_type

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, 7, 1, 3))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(nn.ReLU(True))
            curr_dim *= 2

        # Bottleneck layers.
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        # Up-sampling layers.
        for _ in range(2):
            layers.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim // 2))
            layers.append(nn.ReLU(True))
            curr_dim //= 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(curr_dim, 3, 7, 1, 3),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 7, 1, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.main(x)

        if self.mask_type == 'res':
            return self.conv1(h) + x
        if self.mask_type == 'mask':
            mask = self.conv2(h)
            return mask * self.conv1(h) + (1 - mask) * x
        if self.mask_type == 'bin_mask':
            mask = self.conv2(h) < 0.5
            return mask * self.conv1(h) + ~mask * x
        return self.conv1(h)


def test0():
    x = torch.randn((5, 3, 256, 256))
    model = DMFN()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test0()

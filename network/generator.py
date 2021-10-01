from torch import nn


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, 1),
            nn.InstanceNorm2d(dim_out))

    def forward(self, x):
        return x + self.main(x)


class StarGen(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=6, mask_type='no'):
        super().__init__()
        self.mask_type = mask_type

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, 7, 1, 3))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim * 2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim // 2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

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

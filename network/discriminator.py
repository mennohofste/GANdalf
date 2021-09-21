import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDisc(nn.Module):
    def __init__(self, nr_class=2):
        super().__init__()
        self.nr_class = nr_class

        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
            )

        layers = []
        layers.append(nn.Conv2d(3, 64, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(block(64, 128))
        layers.append(block(128, 256))
        layers.append(block(256, 512, 1))
        self.conv = nn.Sequential(*layers)

        self.src = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Flatten(),
        )
        self.cls = nn.Conv2d(512, nr_class, 31)

    def forward(self, x):
        x = self.conv(x)
        if self.nr_class == 1:
            return self.src(x)
        return self.src(x), self.cls(x).squeeze()


# Discriminator of PEPSI++
class RED(nn.Module):
    def __init__(self, nr_class=2, disc_m='red'):
        super().__init__()
        self.nr_class = nr_class
        self.disc_m = disc_m

        def block(in_c, out_c):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 5, 2, 2)),
                nn.LeakyReLU(0.2),
            )

        layers = []
        layers.append(block(3, 64))
        layers.append(block(64, 128))
        layers.append(block(128, 256))
        layers.append(block(256, 256))
        layers.append(block(256, 256))
        layers.append(block(256, 512))
        self.conv = nn.Sequential(*layers)

        if disc_m == 'patchgan':
            self.src = nn.Sequential(
                nn.Conv2d(512, 1, 1),
                nn.Flatten(),
            )
        if disc_m == 'snpatchgan':
            self.src = nn.Flatten()
        if disc_m == 'red':
            self.src = [nn.Linear(512, 1) for _ in range(4 ** 2)]
        self.cls = nn.Conv2d(512, nr_class, 4)

    def forward(self, x):
        x = self.conv(x)

        if self.disc_m == 'red':
            pixels = x.flatten(2).split(1, 2)
            temp = []
            for i, pixel in enumerate(pixels):
                temp.append(self.src[i](pixel.squeeze()))
            src = torch.cat(temp, 1)
        else:
            src = self.src(x)

        if self.nr_class == 1:
            return src
        return src, self.cls(x).squeeze()


class StarDisc(nn.Module):
    def __init__(self, nr_class=2):
        super().__init__()
        self.nr_class = nr_class

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.LeakyReLU()
            )

        layers = []
        layers.append(block(3, 64))
        layers.append(block(64, 128))
        layers.append(block(128, 256))
        layers.append(block(256, 256))
        layers.append(block(256, 256))
        layers.append(block(256, 512))
        self.conv = nn.Sequential(*layers)

        self.src = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(512, nr_class, 4),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.conv(x)
        if self.nr_class == 1:
            return self.src(x)
        return self.src(x), self.cls(x).squeeze()


def test0():
    x = torch.randn((5, 3, 256, 256))
    model = PatchDisc()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)


def test1():
    x = torch.randn((5, 3, 256, 256))
    model = RED()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)


def test2():
    x = torch.randn((5, 3, 256, 256))
    model = StarDisc()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)


if __name__ == "__main__":
    test0()
    test1()
    test2()

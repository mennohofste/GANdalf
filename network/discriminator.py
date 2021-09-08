import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SNPatchDisc(nn.Module):
    def __init__(self, nr_class=2):
        super().__init__()
        self.nr_class = nr_class

        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 5, stride, 2)),
                nn.LeakyReLU(0.2),
            )

        layers = []
        layers.append(block(3, 64, 1))
        layers.append(block(64, 128))
        layers.append(block(128, 256))
        layers.append(block(256, 256))
        layers.append(block(256, 256))
        layers.append(block(256, 256))
        self.conv = nn.Sequential(*layers)

        self.cls = nn.Conv2d(256, nr_class, 8)

    def forward(self, x):
        x = self.conv(x)
        if self.nr_class == 1:
            return x.flatten(1)
        return x.flatten(1), self.cls(x).squeeze()


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
        layers.append(block(256, 512))
        layers.append(block(512, 1024))
        layers.append(block(1024, 2048))
        self.conv = nn.Sequential(*layers)

        self.src = nn.Conv2d(2048, 1, 3, 1, 1)
        self.cls = nn.Conv2d(2048, nr_class, 4)

    def forward(self, x):
        x = self.conv(x)
        if self.nr_class == 1:
            return self.src(x)
        return self.src(x), self.cls(x).squeeze()


def test():
    x = torch.randn((5, 3, 256, 256))
    model = SNPatchDisc()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)


def test1():
    x = torch.randn((5, 3, 256, 256))
    model = StarDisc()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)


if __name__ == "__main__":
    test()
    test1()
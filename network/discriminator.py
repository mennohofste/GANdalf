import torch
import torch.nn as nn


class StarDisc(nn.Module):
    def __init__(self, img_size=256, nr_class=2):
        super().__init__()

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
        self.cls = nn.Conv2d(2048, nr_class, img_size // 64)

    def forward(self, x):
        features = self.conv(x)
        return self.src(features), self.cls(features).squeeze()


def test1():
    x = torch.randn((5, 3, 256, 256))
    model = StarDisc()
    preds = model(x)
    print(preds[0].shape, preds[1].shape)


if __name__ == "__main__":
    test1()
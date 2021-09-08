import torch
import torch.nn as nn


class StarGen(nn.Module):
    def __init__(self, nr_class=2, nr_resblocks=6, upconv='deconv'):
        super().__init__()

        def block(in_c=256, out_c=256, kernel=3, stride=1, padding=1, conv_m='conv'):
            if conv_m == 'conv':
                conv = nn.Conv2d(in_c, out_c, kernel, stride, padding)
            elif conv_m == 'deconv':
                conv = nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding)
            elif conv_m == 'upconv':
                conv = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_c, out_c, kernel, stride, padding),
                )
            else:
                raise Exception('Convolutional method', conv_m, 'not recognised!')

            return nn.Sequential(
                conv,
                nn.InstanceNorm2d(out_c),
                nn.ReLU(),
            )
        
        layers = []
        layers.append(block(3 + nr_class, 64, 7, 1, 3))
        layers.append(block(64, 128, 4, 2, 1))
        layers.append(block(128, 256, 4, 2, 1))
        self.encoder = nn.Sequential(*layers)

        self.bottleneck = []
        for _ in range(nr_resblocks):
            self.bottleneck.append(block())
            
        layers = []
        layers.append(block(256, 128, 4, 2, 1, upconv))
        layers.append(block(128, 64, 4, 2, 1, upconv))
        layers.append(nn.Conv2d(64, 3, 7, 1, 3))
        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.bottleneck:
            x = layer(x) + x
        return self.decoder(x)


def test():
    x = torch.randn((7, 3 + 2, 256, 256))
    model = StarGen()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
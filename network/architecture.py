import torch

from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_msssim import ssim

from network.generator import StarGen
from network.discriminator import StarDisc


class CycleGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr_gen = 1e-4
        self.lr_disc = 4e-4
        self.betas = (0.5, 0.9)

        nr_classes = 1
        self.gen_x = StarGen(nr_classes)
        self.gen_y = StarGen(nr_classes)
        self.disc_x = StarDisc(nr_classes)
        self.disc_y = StarDisc(nr_classes)

    def forward(self, x):
        return self.gen_y(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        zero = torch.tensor(0)

        # Update generator
        if optimizer_idx == 0:
            x_hat = self.gen_x(y)
            y_hat = self.gen_y(x)

            x_adv_loss = -self.disc_x(x_hat).mean()
            y_adv_loss = -self.disc_y(y_hat).mean()

            x_rec_loss = F.l1_loss(x_hat, x)
            y_rec_loss = F.l1_loss(y_hat, y)

            loss = x_rec_loss + y_rec_loss + x_adv_loss + y_adv_loss
            self.log("train/gen_loss", loss)
            return loss

            # Update discriminator
        if optimizer_idx == 1:
            x_hat = self.gen_x(y)
            y_hat = self.gen_y(x)

            x_adv_loss = (1 - self.disc_x(x)).maximum(zero).mean() + \
                (1 + self.disc_x(x_hat)).maximum(zero).mean()
            y_adv_loss = (1 - self.disc_y(y)).maximum(zero).mean() + \
                (1 + self.disc_y(y_hat)).maximum(zero).mean()

            loss = x_adv_loss + y_adv_loss
            self.log("train/disc_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        def psnr(x, y):
            x = (x + 1) / 2
            y = (y + 1) / 2
            return -10 * torch.log10(F.mse_loss(x, y))

        x, y = batch
        y_hat = self.gen_y(x)
        l1_loss = F.l1_loss(y, y_hat)
        l2_loss = F.mse_loss(y, y_hat)
        psnr_loss = psnr(y, y_hat)
        ssim_loss = ssim(y, y_hat)

        self.log("val/l1", l1_loss)
        self.log("val/l2", l2_loss)
        self.log("val/psnr", psnr_loss)
        self.log("val/ssim", ssim_loss)
        return l1_loss

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            list(self.gen_x.parameters()) + list(self.gen_y.parameters()),
            lr=self.lr_gen,
            betas=self.betas,
        )
        opt_disc = torch.optim.Adam(
            list(self.disc_x.parameters()) + list(self.disc_y.parameters()),
            lr=self.lr_disc,
            betas=self.betas,
        )

        return [opt_gen, opt_disc]

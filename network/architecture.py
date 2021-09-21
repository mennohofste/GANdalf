import torch

from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from pytorch_msssim import ssim
from lpips import LPIPS

from network.generator import StarGen
from network.discriminator import StarDisc


def psnr(x, y):
    return -10 * torch.log10(F.mse_loss(x, y))


def adv_loss(src_r, src_f):
    return (src_r + 1e-8).log().mean() + (1 - src_f + 1e-8).log().mean()


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

        self.lpips = LPIPS()
        self.automatic_optimization = False

    def forward(self, x):
        return self.gen_y(x)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        x, y = batch

        x_hat = self.gen_x(y)
        y_hat = self.gen_y(x)

        # Discriminator
        x_src_r = self.disc_x(x)
        y_src_r = self.disc_y(y)

        x_src_f = self.disc_x(x_hat.detach())
        y_src_f = self.disc_y(y_hat.detach())

        x_adv_loss = adv_loss(x_src_r, x_src_f)
        y_adv_loss = adv_loss(y_src_r, y_src_f)

        x_d_loss = -x_adv_loss
        y_d_loss = -y_adv_loss
        d_loss = x_d_loss + y_d_loss

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # Generator
        # Recompute probabilities over source, because
        # we need backward graph through generator
        x_src_f = self.disc_x(x_hat)
        y_src_f = self.disc_y(y_hat)

        x_adv_loss = adv_loss(x_src_r.detach(), x_src_f)
        y_adv_loss = adv_loss(y_src_r.detach(), y_src_f)

        x_rec_loss = F.l1_loss(self.gen_x(y_hat), x)
        y_rec_loss = F.l1_loss(self.gen_y(x_hat), y)

        lambda_rec = 10
        x_g_loss = x_adv_loss + lambda_rec * x_rec_loss
        y_g_loss = y_adv_loss + lambda_rec * y_rec_loss
        g_loss = x_g_loss + y_g_loss

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        self.log("train/gen_loss", g_loss)
        self.log("train/y_rec_loss", y_rec_loss)
        self.log("train/x_src_r", x_src_r.mean())
        self.log("train/x_src_f", x_src_f.mean())
        self.log("train/y_src_r", y_src_r.mean())
        self.log("train/y_src_f", y_src_f.mean())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.gen_y(x)

        x = (x + 1) / 2
        y = (y + 1) / 2
        y_hat = (y_hat + 1) / 2

        l1_loss = F.l1_loss(y, y_hat)
        l2_loss = F.mse_loss(y, y_hat)
        psnr_loss = psnr(y, y_hat)
        ssim_loss = ssim(y, y_hat, 1)
        lpips_loss = self.lpips(y, y_hat)

        self.log("val/l1", l1_loss)
        self.log("val/l2", l2_loss)
        self.log("val/psnr", psnr_loss)
        self.log("val/ssim", ssim_loss)
        self.log("val/lpips", lpips_loss)
        return l1_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.gen_y(x)
        x_hat = self.gen_x(y)

        x = (x + 1) / 2
        y = (y + 1) / 2
        y_hat = (y_hat + 1) / 2

        l1_loss = F.l1_loss(y, y_hat)
        l2_loss = F.mse_loss(y, y_hat)
        psnr_loss = psnr(y, y_hat)
        ssim_loss = ssim(y, y_hat, 1)
        lpips_loss = self.lpips(y, y_hat)

        print(l1_loss)
        print(l2_loss)
        print(psnr_loss)
        print(ssim_loss)
        print(lpips_loss)
        save_image(y_hat[0], 'test0_y_hat.jpg')
        save_image(y[0], 'test0_y.jpg')
        save_image(x[0], 'test0_x.jpg')
        save_image(x_hat[0], 'test0_x_hat.jpg')
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


class StarGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr_gen = 1e-4
        self.lr_disc = 4e-4
        self.betas = (0.5, 0.9)

        nr_classes = 2
        self.gen = StarGen(nr_classes)
        self.disc = StarDisc(nr_classes)

        self.lpips = LPIPS()

    def forward(self, x):
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        ones = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([self.gen(x), zeros, ones], 1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        ones = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])

        def to_x(y):
            return torch.cat([y, ones, zeros], 1)

        def to_y(x):
            return torch.cat([x, zeros, ones], 1)

        # Update generator
        if optimizer_idx == 0:
            y_hat = self.gen(to_y(x))
            src_r, cls_r = self.disc(x)
            src_f, cls_f = self.disc(y_hat)
            x_adv_loss = (src_r + 1e-8).log().mean() + \
                (1 - src_f + 1e-8).log().mean()
            x_cls_loss = -cls_f.log().mean()
            x_rec_loss = F.l1_loss(self.gen(to_x(y_hat)), x)

            x_hat = self.gen(to_x(y))
            src_r, cls_r = self.disc(y)
            src_f, cls_f = self.disc(x_hat)
            y_adv_loss = (src_r + 1e-8).log().mean() + \
                (1 - src_f + 1e-8).log().mean()
            y_cls_loss = -cls_f.log().mean()
            y_rec_loss = F.l1_loss(self.gen(to_y(x_hat)), y)

            lambda_rec = 10
            x_loss = x_adv_loss + x_cls_loss + lambda_rec * x_rec_loss
            y_loss = y_adv_loss + y_cls_loss + lambda_rec * y_rec_loss
            loss = x_loss + y_loss
            self.log("train/gen_loss", loss)
            self.log("train/y_rec_loss", y_rec_loss)
            return loss

        # Update discriminator
        if optimizer_idx == 1:
            y_hat = self.gen(to_y(x))
            src_r, cls_r = self.disc(x)
            src_f, cls_f = self.disc(y_hat)
            x_adv_loss = (src_r + 1e-8).log().mean() + \
                (1 - src_f + 1e-8).log().mean()
            x_cls_loss = -cls_r.log().mean()

            x_hat = self.gen(to_x(y))
            src_r, cls_r = self.disc(y)
            src_f, cls_f = self.disc(x_hat)
            y_adv_loss = (src_r + 1e-8).log().mean() + \
                (1 - src_f + 1e-8).log().mean()
            y_cls_loss = -cls_r.log().mean()

            x_loss = -x_adv_loss + x_cls_loss
            y_loss = -y_adv_loss + y_cls_loss
            loss = x_loss + y_loss
            self.log("train/disc_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        ones = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])

        def to_y(x):
            return torch.cat([x, zeros, ones], 1)

        y_hat = self.gen(to_y(x))
        x = (x + 1) / 2
        y = (y + 1) / 2
        y_hat = (y_hat + 1) / 2

        l1_loss = F.l1_loss(y, y_hat)
        l2_loss = F.mse_loss(y, y_hat)
        psnr_loss = psnr(y, y_hat)
        ssim_loss = ssim(y, y_hat)
        lpips_loss = self.lpips(y, y_hat)

        self.log("val/l1", l1_loss)
        self.log("val/l2", l2_loss)
        self.log("val/psnr", psnr_loss)
        self.log("val/ssim", ssim_loss)
        self.log("val/lpips", lpips_loss)
        return l1_loss

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr_gen,
            betas=self.betas,
        )
        opt_disc = torch.optim.Adam(
            self.disc.parameters(),
            lr=self.lr_disc,
            betas=self.betas,
        )
        return [opt_gen, opt_disc]

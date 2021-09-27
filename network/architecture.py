import torch

from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from pytorch_msssim import ssim
from lpips import LPIPS

from network.stargan import Generator
from network.stargan import Discriminator


def psnr(x, y):
    return -10 * torch.log10(F.mse_loss(x, y))


class Loss:
    def __init__(self, network, loss_name):
        self.network = network
        self.loss = loss_name
        self.lambda_adv = 0.1

    def __call__(self, src_r, src_f):
        if self.network == 'g':
            return self.g_adv_loss(src_r, src_f)
        if self.network == 'd':
            return self.d_adv_loss(src_r, src_f)

    def g_adv_loss(self, src_r, src_f):
        if self.loss == 'minimax':
            src_r = torch.sigmoid(src_r)
            src_f = torch.sigmoid(src_f)
            return self.lambda_adv * ((src_r + 1e-8).log().mean() + (1 - src_f + 1e-8).log().mean())
        if self.loss == 'minimax_m':
            src_r = torch.sigmoid(src_r)
            src_f = torch.sigmoid(src_f)
            return -self.lambda_adv * (src_f + 1e-8).log().mean()
        if self.loss == 'hinge':
            return -self.lambda_adv * src_f.mean()
        if self.loss == 'wasserstein':
            return -self.lambda_adv * src_f.mean()
        if self.loss == 'rahinge':
            diff_r_f = src_r - src_f.mean()
            diff_f_r = src_f - src_r.mean()
            return self.lambda_adv * (F.relu(1 + diff_r_f).mean() + F.relu(1 - diff_f_r).mean())
        if self.loss == 'ls':
            return self.lambda_adv * src_f.sigmoid().sub(1).pow(2).mean()
        if self.loss == 'rals':
            diff_r_f = src_r.sigmoid() - src_f.sigmoid().mean()
            diff_f_r = src_f.sigmoid() - src_r.sigmoid().mean()
            return self.lambda_adv * (diff_r_f.add(1).pow(2).mean() + diff_f_r.sub(1).pow(2).mean())

    def d_adv_loss(self, src_r, src_f):
        if self.loss == 'minimax' or self.loss == 'minimax_m':
            src_r = torch.sigmoid(src_r)
            src_f = torch.sigmoid(src_f)
            return -self.lambda_adv * ((src_r + 1e-8).log().mean() + (1 - src_f + 1e-8).log().mean())
        if self.loss == 'hinge':
            return self.lambda_adv * (F.relu(1 - src_r).mean() + F.relu(1 + src_f).mean())
        if self.loss == 'wasserstein':
            return self.lambda_adv * (src_f.mean() - src_r.mean())
        if self.loss == 'rahinge':
            diff_r_f = src_r - src_f.mean()
            diff_f_r = src_f - src_r.mean()
            return self.lambda_adv * (F.relu(1 - diff_r_f).mean() + F.relu(1 + diff_f_r).mean())
        if self.loss == 'ls':
            return self.lambda_adv * (src_r.sigmoid().sub(1).pow(2).mean() + src_f.sigmoid().pow(2).mean())
        if self.loss == 'rals':
            diff_r_f = src_r.sigmoid() - src_f.sigmoid().mean()
            diff_f_r = src_f.sigmoid() - src_r.sigmoid().mean()
            return self.lambda_adv * (diff_r_f.sub(1).pow(2).mean() + diff_f_r.add(1).pow(2).mean())

    # def gradient_penalty(self, src_r, src_f):
    #     """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    #     alpha = torch.ones(src_r.size(0), 1, 1, 1)
    #     x_hat = (alpha * src_r.data + (1 - alpha) * src_f.data)
    #     dydx = torch.autograd.grad(outputs=y,
    #                                inputs=x,
    #                                grad_outputs=weight,
    #                                retain_graph=True,
    #                                create_graph=True,
    #                                only_inputs=True)[0]

    #     dydx = dydx.view(dydx.size(0), -1)
    #     dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    #     return torch.mean((dydx_l2norm-1)**2)


class CycleGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr_gen = 1e-4
        self.lr_disc = 4e-4
        self.betas = (0.5, 0.9)

        self.gen_x = Generator()
        self.gen_y = Generator()
        self.disc_x = Discriminator()
        self.disc_y = Discriminator()

        self.lpips = LPIPS()
        self.automatic_optimization = False

        if not args:
            return
        self.g_adv_loss = Loss('g', args.loss)
        self.d_adv_loss = Loss('d', args.loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CycleGAN")
        parser.add_argument("--loss", type=str, default='hinge')
        return parent_parser

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

        x_adv_loss = self.d_adv_loss(x_src_r, x_src_f)
        y_adv_loss = self.d_adv_loss(y_src_r, y_src_f)

        x_d_loss = x_adv_loss
        y_d_loss = y_adv_loss
        d_loss = x_d_loss + y_d_loss

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log("train/x_adv_loss_disc", x_adv_loss)
        self.log("train/y_adv_loss_disc", y_adv_loss)
        # Generator
        # Recompute probabilities over source, because
        # we need backward graph through generator
        x_src_f = self.disc_x(x_hat)
        y_src_f = self.disc_y(y_hat)

        x_adv_loss = self.g_adv_loss(x_src_r.detach(), x_src_f)
        y_adv_loss = self.g_adv_loss(y_src_r.detach(), y_src_f)

        x_rec_loss = F.l1_loss(self.gen_x(y_hat), x)
        y_rec_loss = F.l1_loss(self.gen_y(x_hat), y)

        x_g_loss = x_adv_loss + x_rec_loss
        y_g_loss = y_adv_loss + y_rec_loss
        g_loss = x_g_loss + y_g_loss

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        self.log("train/disc_loss", d_loss)
        self.log("train/gen_loss", g_loss)
        self.log("train/x_rec_loss", x_rec_loss)
        self.log("train/y_rec_loss", y_rec_loss)
        self.log("train/x_adv_loss", x_adv_loss)
        self.log("train/y_adv_loss", y_adv_loss)

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
        x_hat = (x_hat + 1) / 2
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
        print(lpips_loss.mean())
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

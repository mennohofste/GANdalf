from numpy import dtype
import torch

import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from pytorch_msssim import ssim
from lpips import LPIPS

from loss import Dloss, Gloss
from network.stargan import Generator
from network.stargan import Discriminator


def psnr(x, y):
    return -10 * torch.log10(F.mse_loss(x, y))


class CycleGAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
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
        self.g_adv_loss = Gloss(args.loss, args.lambda_adv)
        self.d_adv_loss = Dloss(args.loss, args.lambda_adv)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CycleGAN")
        parser.add_argument("--loss", type=str, default='wasserstein')
        parser.add_argument("--lambda_adv", type=float, default=0.1)
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

        x_adv_loss = self.d_adv_loss(x_src_r, x_src_f, self.disc_x, x, x_hat)
        y_adv_loss = self.d_adv_loss(y_src_r, y_src_f, self.disc_y, y, y_hat)

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

    def _eval(self, batch):
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
        return {
            'l1': l1_loss,
            'l2': l2_loss,
            'psnr': psnr_loss,
            'ssim': ssim_loss,
            'lpips': lpips_loss,
        }

    def validation_step(self, batch, batch_idx):
        losses = self._eval(batch)
        self.log("val/l1", losses['l1'])
        self.log("val/l2", losses['l2'])
        self.log("val/psnr", losses['psnr'])
        self.log("val/ssim", losses['ssim'])
        self.log("val/lpips", losses['lpips'])
        return losses

    def test_step(self, batch, batch_idx):
        losses = self._eval(batch)

        if batch_idx == 0:
            x, y = batch
            y_hat = self.gen_y(x)

            x = (x + 1) / 2
            y = (y + 1) / 2
            y_hat = (y_hat + 1) / 2

            for i in range(x.size(0)):
                save_image(y_hat[i], f'images/{i}_m.jpg')
                save_image(y[i], f'images/{i}_target.jpg')
                save_image(x[i], f'images/{i}_input.jpg')
        return losses

    def test_epoch_end(self, outputs):
        avg_l1 = torch.stack([x['l1'] for x in outputs]).mean()
        avg_l2 = torch.stack([x['l2'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        avg_lpips = torch.stack([x['lpips'] for x in outputs]).mean()
        result = {
            'l1': avg_l1,
            'l2': avg_l2,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
        }
        print(result)
        return result

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
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.lr_gen = 1e-4
        self.lr_disc = 4e-4
        self.betas = (0.5, 0.9)

        self.gen = Generator(c_dim=2)
        self.disc = Discriminator(c_dim=2)

        self.lpips = LPIPS()
        self.automatic_optimization = False

        if not args:
            return
        self.g_adv_loss = Gloss(args.loss, args.lambda_adv)
        self.d_adv_loss = Dloss(args.loss, args.lambda_adv)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StarGAN")
        parser.add_argument("--loss", type=str, default='wasserstein')
        parser.add_argument("--lambda_adv", type=float, default=0.1)
        return parent_parser

    def forward(self, x):
        _, y_label = self.get_labels(x.size(0))
        return self.gen(x, y_label)

    def get_labels(self, batch_size):
        x_label = torch.zeros(
            batch_size, dtype=torch.int64, device=self.device)
        x_label = F.one_hot(x_label, num_classes=2)
        y_label = torch.ones(batch_size, dtype=torch.int64, device=self.device)
        y_label = F.one_hot(y_label, num_classes=2)
        return x_label, y_label

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        x, y = batch
        x_label, y_label = self.get_labels(x.size(0))

        x_hat = self.gen(y, x_label)
        y_hat = self.gen(x, y_label)

        # Discriminator
        x_src_r, x_cls_r = self.disc(x)
        y_src_r, y_cls_r = self.disc(y)

        x_src_f, _ = self.disc(x_hat.detach())
        y_src_f, _ = self.disc(y_hat.detach())

        x_adv_loss = self.d_adv_loss(x_src_r, x_src_f, self.disc, x, x_hat)
        y_adv_loss = self.d_adv_loss(y_src_r, y_src_f, self.disc, y, y_hat)

        x_cls_loss = F.binary_cross_entropy_with_logits(
            x_cls_r, x_label.float())
        y_cls_loss = F.binary_cross_entropy_with_logits(
            y_cls_r, y_label.float())

        x_d_loss = x_adv_loss + x_cls_loss
        y_d_loss = y_adv_loss + y_cls_loss
        d_loss = x_d_loss + y_d_loss

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log("train/x_loss_disc", x_d_loss)
        self.log("train/y_loss_disc", y_d_loss)
        self.log("train/x_adv_loss_disc", x_adv_loss)
        self.log("train/x_adv_loss_disc", x_adv_loss)
        self.log("train/y_adv_loss_disc", y_adv_loss)
        self.log("train/x_cls_loss_disc", x_cls_loss)
        self.log("train/y_cls_loss_disc", y_cls_loss)
        # Generator
        # Recompute probabilities over source, because
        # we need backward graph through generator
        x_src_f, x_cls_f = self.disc(x_hat)
        y_src_f, y_cls_f = self.disc(y_hat)

        x_adv_loss = self.g_adv_loss(x_src_r.detach(), x_src_f)
        y_adv_loss = self.g_adv_loss(y_src_r.detach(), y_src_f)

        x_rec_loss = F.l1_loss(self.gen(y_hat, x_label), x)
        y_rec_loss = F.l1_loss(self.gen(x_hat, y_label), y)

        x_cls_loss = F.binary_cross_entropy_with_logits(
            x_cls_f, x_label.float())
        y_cls_loss = F.binary_cross_entropy_with_logits(
            y_cls_f, y_label.float())

        x_g_loss = x_adv_loss + x_rec_loss + x_cls_loss
        y_g_loss = y_adv_loss + y_rec_loss + y_cls_loss
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
        self.log("train/x_cls_loss", x_cls_loss)
        self.log("train/y_cls_loss", y_cls_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_label = self.get_labels(x.size(0))
        y_hat = self.gen(x, y_label)

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
        x_label, y_label = self.get_labels(x.size(0))
        y_hat = self.gen(x, y_label)
        x_hat = self.gen(y, x_label)

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


class StarGAN_2disc(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.lr_gen = 1e-4
        self.lr_disc = 4e-4
        self.betas = (0.5, 0.9)

        self.gen = Generator(c_dim=2)
        self.disc_x = Discriminator()
        self.disc_y = Discriminator()

        self.lpips = LPIPS()
        self.automatic_optimization = False

        if not args:
            return
        self.g_adv_loss = Gloss(args.loss, args.lambda_adv)
        self.d_adv_loss = Dloss(args.loss, args.lambda_adv)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StarGAN")
        parser.add_argument("--loss", type=str, default='wasserstein')
        parser.add_argument("--lambda_adv", type=float, default=0.1)
        return parent_parser

    def forward(self, x):
        _, y_label = self.get_labels(x.size(0))
        return self.gen(x, y_label)

    def get_labels(self, batch_size):
        x_label = torch.zeros(
            batch_size, dtype=torch.int64, device=self.device)
        x_label = F.one_hot(x_label, num_classes=2)
        y_label = torch.ones(batch_size, dtype=torch.int64, device=self.device)
        y_label = F.one_hot(y_label, num_classes=2)
        return x_label, y_label

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        x, y = batch
        x_label, y_label = self.get_labels(x.size(0))

        x_hat = self.gen(y, x_label)
        y_hat = self.gen(x, y_label)

        # Discriminator
        x_src_r = self.disc_x(x)
        y_src_r = self.disc_y(y)

        x_src_f = self.disc_x(x_hat.detach())
        y_src_f = self.disc_y(y_hat.detach())

        x_adv_loss = self.d_adv_loss(x_src_r, x_src_f, self.disc_x, x, x_hat)
        y_adv_loss = self.d_adv_loss(y_src_r, y_src_f, self.disc_y, y, y_hat)

        x_d_loss = x_adv_loss
        y_d_loss = y_adv_loss
        d_loss = x_d_loss + y_d_loss

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log("train/x_loss_disc", x_d_loss)
        self.log("train/y_loss_disc", y_d_loss)
        self.log("train/x_adv_loss_disc", x_adv_loss)
        self.log("train/x_adv_loss_disc", x_adv_loss)
        self.log("train/y_adv_loss_disc", y_adv_loss)
        # Generator
        # Recompute probabilities over source, because
        # we need backward graph through generator
        x_src_f = self.disc_x(x_hat)
        y_src_f = self.disc_y(y_hat)

        x_adv_loss = self.g_adv_loss(x_src_r.detach(), x_src_f)
        y_adv_loss = self.g_adv_loss(y_src_r.detach(), y_src_f)

        x_rec_loss = F.l1_loss(self.gen(y_hat, x_label), x)
        y_rec_loss = F.l1_loss(self.gen(x_hat, y_label), y)

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

    def _eval(self, batch):
        x, y = batch
        _, y_label = self.get_labels(x.size(0))
        y_hat = self.gen(x, y_label)

        x = (x + 1) / 2
        y = (y + 1) / 2
        y_hat = (y_hat + 1) / 2

        l1_loss = F.l1_loss(y, y_hat)
        l2_loss = F.mse_loss(y, y_hat)
        psnr_loss = psnr(y, y_hat)
        ssim_loss = ssim(y, y_hat, 1)
        lpips_loss = self.lpips(y, y_hat)

        return {
            'l1': l1_loss,
            'l2': l2_loss,
            'psnr': psnr_loss,
            'ssim': ssim_loss,
            'lpips': lpips_loss,
        }

    def validation_step(self, batch, batch_idx):
        losses = self._eval(batch)
        self.log("val/l1", losses['l1'])
        self.log("val/l2", losses['l2'])
        self.log("val/psnr", losses['psnr'])
        self.log("val/ssim", losses['ssim'])
        self.log("val/lpips", losses['lpips'])
        return losses

    def test_step(self, batch, batch_idx):
        losses = self._eval(batch)

        if batch_idx == 0:
            x, y = batch
            x_label, y_label = self.get_labels(x.size(0))
            y_hat = self.gen(x, y_label)
            x_hat = self.gen(y, x_label)
            y_hat_hat = self.gen(x_hat, y_label)

            x = (x + 1) / 2
            y = (y + 1) / 2
            x_hat = (x_hat + 1) / 2
            y_hat = (y_hat + 1) / 2
            y_hat_hat = (y_hat_hat + 1) / 2

            for i in range(x.size(0)):
                save_image(x_hat[i], f'images/{i}_gy.jpg')
                save_image(y_hat[i], f'images/{i}_gx.jpg')
                save_image(y_hat_hat[i], f'images/{i}_ggy.jpg')
                save_image(y[i], f'images/{i}_target.jpg')
                save_image(x[i], f'images/{i}_input.jpg')
        return losses

    def test_epoch_end(self, outputs):
        avg_l1 = torch.stack([x['l1'] for x in outputs]).mean()
        avg_l2 = torch.stack([x['l2'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        avg_lpips = torch.stack([x['lpips'] for x in outputs]).mean()
        result = {
            'l1': avg_l1,
            'l2': avg_l2,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
        }
        print(result)
        return result

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr_gen,
            betas=self.betas,
        )
        opt_disc = torch.optim.Adam(
            list(self.disc_x.parameters()) + list(self.disc_y.parameters()),
            lr=self.lr_disc,
            betas=self.betas,
        )
        return [opt_gen, opt_disc]

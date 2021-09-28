from network.stargan import Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dloss:
    def __init__(self, loss_name, lambda_adv=0.1):
        self.loss = loss_name
        self.lambda_adv = lambda_adv

    def __call__(self, src_r, src_f, disc=None, real=None, fake=None):
        if self.loss == 'minimax':
            src_r = torch.sigmoid(src_r)
            src_f = torch.sigmoid(src_f)
            return -self.lambda_adv * ((src_r + 1e-8).log().mean() + (1 - src_f + 1e-8).log().mean())
        if self.loss == 'hinge':
            return self.lambda_adv * (F.relu(1 - src_r).mean() + F.relu(1 + src_f).mean())
        if self.loss == 'wasserstein':
            assert disc is not None and real is not None and fake is not None
            return self.lambda_adv * (src_f.mean() - src_r.mean() + 10 * self.gradient_penalty(disc, real, fake))
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

    def gradient_penalty(self, disc, real, fake):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        epsilon = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        x_hat = (epsilon * real + (1 - epsilon)
                 * fake.detach()).requires_grad_()
        src_out = disc(x_hat)

        dydx = torch.autograd.grad(outputs=src_out,
                                   inputs=x_hat,
                                   grad_outputs=torch.ones_like(src_out),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)


class Gloss:
    def __init__(self, loss_name, lambda_adv=0.1):
        self.loss = loss_name
        self.lambda_adv = lambda_adv

    def __call__(self, src_r, src_f):
        if self.loss == 'minimax':
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

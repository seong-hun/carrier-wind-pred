import torch
import torch.nn as nn
from torch.nn import functional as F

import args
from .network import set_device


class LossEG(nn.Module):
    def __init__(self, feed_forward=True):
        super().__init__()

        self.match_loss = not feed_forward

        set_device(self)

    def loss_cnt(self, x, x_hat):
        # return F.mse_loss(
        #     x.reshape(-1), x_hat.reshape(-1)) * args.LOSS_CNT_WEIGHT
        return F.l1_loss(
            x.reshape(-1), x_hat.reshape(-1)) * args.LOSS_CNT_WEIGHT

    def loss_adv(self, r_x_hat):
        return -r_x_hat.mean()

    def loss_mch(self, e_hat, W_i):
        return F.l1_loss(
            W_i.reshape(-1), e_hat.reshape(-1)) * args.LOSS_MCH_WEIGHT

    def forward(self, x, x_hat, r_x_hat, e_hat, W_i):
        x = x.to(self.device)
        x_hat = x_hat.to(self.device)
        r_x_hat = r_x_hat.to(self.device)
        e_hat = e_hat.to(self.device)

        cnt = self.loss_cnt(x, x_hat)
        adv = self.loss_adv(r_x_hat)
        mch = self.loss_mch(e_hat, W_i)

        return (cnt + adv + mch).reshape(1)


class LossD(nn.Module):
    def __init__(self):
        super().__init__()

        set_device(self)

    def forward(self, r_x, r_x_hat):
        r_x = r_x.to(self.device)
        r_x_hat = r_x_hat.to(self.device)

        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(-1)
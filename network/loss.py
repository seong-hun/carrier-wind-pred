import torch
import torch.nn as nn
from torch.nn import functional as F

import args


class LossEG(nn.Module):
    def __init__(self, feed_forward=True):
        super().__init__()

        self.match_loss = not feed_forward

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
        cnt = self.loss_cnt(x, x_hat)
        adv = self.loss_adv(r_x_hat)
        mch = self.loss_mch(e_hat, W_i)

        return (cnt + adv + mch).reshape(1)


class LossD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r_x, r_x_hat):
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(-1)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg19

import args
from .network import set_device
from network.vgg import vgg_face, VGG_Activations


class LossEG(nn.Module):
    def __init__(self, feed_forward=True):
        super().__init__()

        self.VGG_FACE_AC = VGG_Activations(
            vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        self.VGG19_AC = VGG_Activations(
            vgg19(pretrained=True), [1, 6, 11, 20, 29])

        self.match_loss = not feed_forward

        set_device(self)

    def loss_cnt(self, x, x_hat):
        return F.l1_loss(x.reshape(-1), x_hat.reshape(-1))

    # def loss_cnt(self, x, x_hat):
    #     IMG_NET_MEAN = torch.Tensor([
    #         0, 0, 0, 0.3, 0.3]).reshape(
    #             [1, args.CHANNEL, 1, 1]).to(self.device)
    #     IMG_NET_STD = torch.Tensor([
    #         0.3, 0.3, 0.3, 0.2, 0.3]).reshape(
    #             [1, args.CHANNEL, 1, 1]).to(self.device)

    #     x = (x - IMG_NET_MEAN) / IMG_NET_STD
    #     x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD

    #     # VGG19 Loss
    #     vgg19_x_hat = self.VGG19_AC(x_hat[:, :3, ...])
    #     vgg19_x = self.VGG19_AC(x[:, :3, ...])

    #     vgg19_loss = 0
    #     for i in range(0, len(vgg19_x)):
    #         vgg19_loss += F.l1_loss(vgg19_x_hat[i], vgg19_x[i])

    #     # VGG Face Loss
    #     vgg_face_x_hat = self.VGG_FACE_AC(x_hat[:, :3, ...])
    #     vgg_face_x = self.VGG_FACE_AC(x[:, :3, ...])

    #     vgg_face_loss = 0
    #     for i in range(0, len(vgg_face_x)):
    #         vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

    #     vgg_loss = (
    #         vgg19_loss * args.LOSS_VGG19_WEIGHT
    #         + vgg_face_loss * args.LOSS_VGG_FACE_WEIGHT)

    #     # Mask loss
    #     mask_loss = F.l1_loss(
    #         x[:, 3, ...].reshape(-1),
    #         x_hat[:, 3, ...].reshape(-1)) * args.LOSS_MASK_WEIGHT

    #     return vgg_loss + mask_loss

    def loss_adv(self, r_x_hat):
        return -r_x_hat.mean()

    def loss_mch(self, e_hat, W_i):
        return F.l1_loss(
            W_i.reshape(-1), e_hat.reshape(-1)) * args.LOSS_MCH_WEIGHT

    def forward(self, x, y, x_hat, r_x_hat, e_hat, W_i):
        x = x.to(self.device)
        y = y.to(self.device)
        x_hat = x_hat.to(self.device)
        r_x_hat = r_x_hat.to(self.device)
        e_hat = e_hat.to(self.device)
        W_i = W_i.to(self.device)

        cnt = self.loss_cnt(x, x_hat)
        adv = self.loss_adv(r_x_hat)
        mch = self.loss_mch(e_hat, W_i) if self.match_loss else 0

        return (cnt + adv + mch).reshape(1)


class LossD(nn.Module):
    def __init__(self):
        super().__init__()
        set_device(self)

    def forward(self, r_x, r_x_hat):
        r_x = r_x.to(self.device)
        r_x_hat = r_x_hat.to(self.device)
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(-1)

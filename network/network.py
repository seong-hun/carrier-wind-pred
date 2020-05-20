from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .components import (
    ResidualBlock, AdaptiveResidualBlock, ResidualBlockDown,
    AdaptiveResidualBlockUp, SelfAttention
)
import args

can_cuda = torch.cuda.is_available()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_device(module):
    use_cuda = can_cuda and args.GPU[type(module).__name__]
    module.device = "cuda" if use_cuda else "cpu"
    module.to(module.device)


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ResidualBlockDown(args.CHANNEL * 2, 32)
        self.conv2 = ResidualBlockDown(32, 64)
        self.conv3 = ResidualBlockDown(64, 128)
        self.att = SelfAttention(128)
        self.conv4 = ResidualBlockDown(128, 256)
        self.conv5 = ResidualBlock(256)
        self.conv6 = ResidualBlock(256)

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.apply(weights_init)

        set_device(self)

    def forward(self, x, y):
        # x, y: [BxK, 5, 32, 32]
        assert x.dim() == 4 and x.shape[1] == args.CHANNEL
        assert x.shape == y.shape

        x = x.to(self.device)
        y = y.to(self.device)

        out = torch.cat((x, y), dim=1)  # [BxK, 10, 64, 64]

        # Encode
        out = (self.conv1(out))  # [BxK, 32, 32, 32]
        out = (self.conv2(out))  # [BxK, 64, 16, 16]
        out = (self.conv3(out))  # [BxK, 128, 8, 8]
        out = self.att(out)
        out = (self.conv4(out))  # [BxK, 256, 4, 4]
        out = (self.conv5(out))  # [BxK, 256, 4, 4]
        out = (self.conv6(out))  # [BxK, 256, 4, 4]

        # Vectorize [BxK, E_VECTOR_LENGTH]
        out = self.pooling(out)  # [BxK, 256, 1, 1]
        out = F.relu(out.view(-1, args.E_VECTOR_LENGTH))  # [BxK, 256]

        return out


class Generator(nn.Module):
    ADAIN_LAYERS = OrderedDict([
        ('res1', (256, 256)),
        ('res2', (256, 256)),
        ('res3', (256, 256)),
        ('res4', (256, 256)),
        ('res5', (256, 256)),
        ('deconv6', (256, 256)),
        ('deconv5', (256, 256)),
        ('deconv4', (256, 128)),
        ('deconv3', (128, 64)),
        ('deconv2', (64, 32)),
        ('deconv1', (32, args.CHANNEL))
    ])

    def __init__(self):
        super().__init__()

        # Projection layer
        self.PSI_PORTIONS, self.psi_length = self.define_psi_slices()
        self.projection = nn.Parameter(torch.rand(
            self.psi_length, args.E_VECTOR_LENGTH).normal_(0.0, 0.02))

        # encoding layers
        self.conv1 = ResidualBlockDown(args.CHANNEL, 32)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = ResidualBlockDown(32, 64)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ResidualBlockDown(64, 128)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        self.att1 = SelfAttention(128)

        self.conv4 = ResidualBlockDown(128, 256)
        self.in4_e = nn.InstanceNorm2d(256, affine=True)

        self.conv5 = ResidualBlock(256)
        self.in5_e = nn.InstanceNorm2d(256, affine=True)

        self.conv6 = ResidualBlock(256)
        self.in6_e = nn.InstanceNorm2d(256, affine=True)

        # residual layers
        self.res1 = AdaptiveResidualBlock(256)
        self.res2 = AdaptiveResidualBlock(256)
        self.res3 = AdaptiveResidualBlock(256)
        self.res4 = AdaptiveResidualBlock(256)
        self.res5 = AdaptiveResidualBlock(256)

        # decoding layers
        self.deconv6 = AdaptiveResidualBlock(256)
        self.in6_d = nn.InstanceNorm2d(256, affine=True)

        self.deconv5 = AdaptiveResidualBlock(256)
        self.in5_d = nn.InstanceNorm2d(256, affine=True)

        self.deconv4 = AdaptiveResidualBlockUp(256, 128, upsample=2)
        self.in4_d = nn.InstanceNorm2d(128, affine=True)

        self.deconv3 = AdaptiveResidualBlockUp(128, 64, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.att2 = SelfAttention(64)

        self.deconv2 = AdaptiveResidualBlockUp(64, 32, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = AdaptiveResidualBlockUp(32, args.CHANNEL, upsample=2)
        self.in1_d = nn.InstanceNorm2d(args.CHANNEL, affine=True)

        self.apply(weights_init)

        set_device(self)

    def forward(self, y, e):
        e = e.to(self.device)
        y = y.to(self.device)

        out = y  # [B, 5, 64, 64]

        # Calculate psi_hat parameters
        P = self.projection.unsqueeze(0)
        P = P.expand(e.shape[0], P.shape[1], P.shape[2])
        psi_hat = torch.bmm(P, e.unsqueeze(2)).squeeze(2)

        # Encode
        out = self.in1_e(self.conv1(out))  # [B, 32, 32, 32]
        out = self.in2_e(self.conv2(out))  # [B, 64, 16, 16]
        out = self.in3_e(self.conv3(out))  # [B, 128, 8, 8]
        out = self.att1(out)
        out = self.in4_e(self.conv4(out))  # [B, 256, 4, 4]
        out = self.in5_e(self.conv5(out))  # [B, 256, 4, 4]
        out = self.in6_e(self.conv6(out))  # [B, 256, 4, 4]

        # Residual layers
        out = self.res1(out, *self.slice_psi(psi_hat, 'res1'))
        out = self.res2(out, *self.slice_psi(psi_hat, 'res2'))
        out = self.res3(out, *self.slice_psi(psi_hat, 'res3'))
        out = self.res4(out, *self.slice_psi(psi_hat, 'res4'))
        out = self.res5(out, *self.slice_psi(psi_hat, 'res5'))

        # Decode
        out = self.in6_d(self.deconv6(
            out, *self.slice_psi(psi_hat, 'deconv6')))  # [B, 256, 4, 4]
        out = self.in5_d(self.deconv5(
            out, *self.slice_psi(psi_hat, 'deconv5')))  # [B, 256, 4, 4]
        out = self.in4_d(self.deconv4(
            out, *self.slice_psi(psi_hat, 'deconv4')))  # [B, 128, 8, 8]
        out = self.in3_d(self.deconv3(
            out, *self.slice_psi(psi_hat, 'deconv3')))  # [B, 64, 16, 16]
        out = self.att2(out)
        out = self.in2_d(self.deconv2(
            out, *self.slice_psi(psi_hat, 'deconv2')))  # [B, 32, 32, 32]
        out = self.in1_d(self.deconv1(
            out, *self.slice_psi(psi_hat, 'deconv1')))  # [B, 4, 64, 64]

        out[:, :3, ...] = torch.tanh(out[:, :3, ...]) * 10
        out[:, 3, ...] = torch.sigmoid(out[:, 3, ...])
        out[:, 4, ...] = torch.sigmoid(out[:, 4, ...])

        return out

    def slice_psi(self, psi, portion):
        idx0, idx1 = self.PSI_PORTIONS[portion]
        len1, len2 = self.ADAIN_LAYERS[portion]
        aux = psi[:, idx0:idx1].unsqueeze(-1)
        mean1, std1 = aux[:, 0:len1], aux[:, len1:2 * len1]
        mean2, std2 = aux[:, 2 * len1:2 * len1 + len2], aux[:, 2 * len1 + len2:]
        return mean1, std1.abs(), mean2, std2.abs()

    def define_psi_slices(self):
        out = {}
        d = self.ADAIN_LAYERS
        start_idx, end_idx = 0, 0
        for layer in d:
            end_idx = start_idx + d[layer][0] * 2 + d[layer][1] * 2
            out[layer] = (start_idx, end_idx)
            start_idx = end_idx

        return out, end_idx


class Discriminator(nn.Module):
    def __init__(self, training_videos):
        super().__init__()

        self.conv1 = ResidualBlockDown((args.CHANNEL - 1) * 2, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlock(256)
        self.conv5 = ResidualBlock(256)
        self.conv6 = ResidualBlock(256)
        self.res_block = ResidualBlock(256)

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.W = nn.Parameter(torch.rand(256, training_videos).normal_(0.0, 0.02))
        self.w_0 = nn.Parameter(torch.rand(256, 1).normal_(0.0, 0.02))
        self.b = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.apply(weights_init)

        set_device(self)

    def forward(self, x, y, i):
        assert x.dim() == 4 and x.shape[1] == args.CHANNEL
        assert x.shape == y.shape

        x = x[:, :-1, :, :]
        y = y[:, :-1, :, :]
        x = x.to(self.device)
        y = y.to(self.device)

        # Concatenate x & y
        out = torch.cat((x, y), dim=1)  # [B, 10, 64, 64]

        # Encode
        out_0 = (self.conv1(out))  # [B, 64, 32, 32]
        out_1 = (self.conv2(out_0))  # [B, 128, 16, 16]
        out_2 = (self.conv3(out_1))  # [B, 256, 8, 8]
        out_3 = self.att(out_2)
        out_4 = (self.conv4(out_3))  # [B, 256, 4, 4]
        out_5 = (self.conv5(out_4))  # [B, 256, 4, 4]
        out_6 = (self.conv6(out_5))  # [B, 256, 4, 4]
        out_7 = (self.res_block(out_6))

        # Vectorize
        out = F.relu(self.pooling(out_7)).view(-1, 256, 1)  # [B, 256, 1]

        # Calculate Realism Score
        _out = out.transpose(1, 2)  # [B, 1, 256]
        _W_i = (self.W[:, i].unsqueeze(-1)).transpose(0, 1)  # [B, 256, 1]
        out = torch.bmm(_out, _W_i + self.w_0) + self.b

        out = out.reshape(x.shape[0])
        # out = torch.tanh(out)

        return out, [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg

import args


class VGG(nn.Module):
    def __init__(self, features, num_classes=50, init_weights=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    in_channels = 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
}


class VGG_Activations(nn.Module):
    """
    This class allows us to execute only a part of a given VGG network
    and obtain the activations for the specified
    feature blocks. Note that we are taking the result of
    the activation function after each one of the given indeces,
    and that we consider 1 to be the first index.
    """
    def __init__(self, vgg_network, feature_idx):
        super().__init__()
        features = list(vgg_network.features)
        self.features = nn.ModuleList(features).eval()
        self.idx_list = feature_idx

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.idx_list:
                results.append(x)

        return results


def vgg_face(pretrained=False, in_channels=3, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(
        make_layers(cfgs['D'], in_channels=in_channels, batch_norm=False),
        num_classes=50,
        **kwargs)
    if pretrained:
        model.load_state_dict(vgg_face_state_dict())
    return model


def vgg_face_state_dict():
    default = torch.load(args.VGG_FACE)
    state_dict = OrderedDict({
        'features.0.weight': default['conv1_1.weight'],
        'features.0.bias': default['conv1_1.bias'],
        'features.2.weight': default['conv1_2.weight'],
        'features.2.bias': default['conv1_2.bias'],
        'features.5.weight': default['conv2_1.weight'],
        'features.5.bias': default['conv2_1.bias'],
        'features.7.weight': default['conv2_2.weight'],
        'features.7.bias': default['conv2_2.bias'],
        'features.10.weight': default['conv3_1.weight'],
        'features.10.bias': default['conv3_1.bias'],
        'features.12.weight': default['conv3_2.weight'],
        'features.12.bias': default['conv3_2.bias'],
        'features.14.weight': default['conv3_3.weight'],
        'features.14.bias': default['conv3_3.bias'],
        'features.17.weight': default['conv4_1.weight'],
        'features.17.bias': default['conv4_1.bias'],
        'features.19.weight': default['conv4_2.weight'],
        'features.19.bias': default['conv4_2.bias'],
        'features.21.weight': default['conv4_3.weight'],
        'features.21.bias': default['conv4_3.bias'],
        'features.24.weight': default['conv5_1.weight'],
        'features.24.bias': default['conv5_1.bias'],
        'features.26.weight': default['conv5_2.weight'],
        'features.26.bias': default['conv5_2.bias'],
        'features.28.weight': default['conv5_3.weight'],
        'features.28.bias': default['conv5_3.bias'],
        'classifier.0.weight': default['fc6.weight'],
        'classifier.0.bias': default['fc6.bias'],
        'classifier.3.weight': default['fc7.weight'],
        'classifier.3.bias': default['fc7.bias'],
        'classifier.6.weight': default['fc8.weight'],
        'classifier.6.bias': default['fc8.bias']
    })
    return state_dict

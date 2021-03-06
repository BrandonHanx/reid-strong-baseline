"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.deit import deit
from .backbones.resnet import BasicBlock, Bottleneck, ResNet
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.senet import (SEBottleneck, SENet, SEResNetBottleneck,
                              SEResNeXtBottleneck)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(
        self,
        num_classes,
        last_stride,
        model_path,
        neck,
        neck_feat,
        model_name,
        pretrain_choice,
    ):
        super(Baseline, self).__init__()
        if model_name == "resnet18":
            self.in_planes = 512
            self.base = ResNet(
                last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2]
            )
        elif model_name == "resnet34":
            self.in_planes = 512
            self.base = ResNet(
                last_stride=last_stride, block=BasicBlock, layers=[3, 4, 6, 3]
            )
        elif model_name == "resnet50":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3]
            )
        elif model_name == "resnet101":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 4, 23, 3]
            )
        elif model_name == "resnet152":
            self.base = ResNet(
                last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3]
            )

        elif model_name == "se_resnet50":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 6, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet101":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 23, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet152":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 8, 36, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext50":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 6, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext101":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 23, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "senet154":
            self.base = SENet(
                block=SEBottleneck,
                layers=[3, 8, 36, 3],
                groups=64,
                reduction=16,
                dropout_p=0.2,
                last_stride=last_stride,
            )
        elif model_name == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride)

        if model_name == "deit_small":
            self.base = deit("vit_deit_small_patch16_224")
            self.gap = None
            self.in_planes = self.base.embed_dim
        elif model_name == "deit_base":
            self.base = deit("vit_deit_base_patch16_224")
            self.gap = None
            self.in_planes = self.base.embed_dim
        elif model_name == "vit_base":
            self.base = deit("vit_base_patch16_224_in21k")
            self.gap = None
            self.in_planes = self.base.embed_dim
        elif model_name == "vit_base_jpm":
            self.base = deit("vit_jpm_base_patch16_224_in21k")
            self.gap = None
            self.in_planes = self.base.vit.embed_dim
        elif model_name == "deit_small_jpm":
            self.base = deit("deit_jpm_small_patch16_224")
            self.gap = None
            self.in_planes = self.base.vit.embed_dim
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            # self.gap = nn.AdaptiveMaxPool2d(1)

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......")

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == "no":
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        if self.gap:
            global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
            global_feat = global_feat.view(
                global_feat.shape[0], -1
            )  # flatten to (bs, 2048)
        else:
            global_feat = self.base(x)

        jpm = False
        if isinstance(global_feat, list):
            jpm = True

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            if jpm:
                feat = [self.bottleneck(x) for x in global_feat]
            else:
                feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            if jpm:
                cls_score = [self.classifier(x) for x in feat]
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss

        if self.neck_feat == "after":
            # print("Test with feature after BN")
            if jpm:
                return feat[0]
            return feat

        # print("Test with feature before BN")
        if jpm:
            return global_feat[0]
        return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if "classifier" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

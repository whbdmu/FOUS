from collections import OrderedDict
import torch.nn.functional as F
import torchvision
import torch
import math
from torch import nn
from spcl.models.dsbn import DSBN2d, DSBN1d

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024
        self.att = ADAM(1024)


    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):

        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]


    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])
    
    def bottleneck_forward(self, bottleneck, x, is_source):
        identity = x

        out = bottleneck.conv1(x)
        out = bottleneck.bn1(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv2(out)
        out = bottleneck.bn2(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv3(out)
        out = bottleneck.bn3(out, is_source)
        if bottleneck.downsample is not None:
            for module in bottleneck.downsample:
                if not isinstance(module, DSBN2d):
                    identity = module(x)
                else:
                    identity = module(identity, is_source)
        out += identity
        out = bottleneck.relu(out)
        return out

    def forward(self, x, is_source=True):

        # x = self.att(x)
        #对于reid head的dsbn特殊处理
        #需要取出没有child的module组成list一次执行，可以避免递归中重新实现所有带is_source的forward
        #Bottleneck的forward步骤有缺失
        module_seq=[]
        is_reid_head = False
        for _, (child_name, child) in enumerate(self.named_modules()):
            if isinstance(child, DSBN2d) or isinstance(child, DSBN1d):
                is_reid_head = True
            if isinstance(child, torchvision.models.resnet.Bottleneck):
                module_seq.append(child)
        if is_reid_head:
            feat = x.clone()
            for module in module_seq:
                # x = CBAM(self.out_channels[0])
                feat = self.bottleneck_forward(module, feat, is_source)
        else:
            feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])

class ChannelAttention1(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ADAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(ADAM, self).__init__()
        self.channelattention = ChannelAttention1(channel, ratio=ratio)
        self.spatialattention = SpatialAttention1(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

import torch
import torch.nn as nn
import numpy as np
import config

conf=[
    (32, 3, 1),#outchannel, kernel size, stride
    (64, 3, 2),#downsample
    ["R", 1],
    (128, 3, 2),#downsample
    ["R", 2],
    (256, 3, 2),#downsample
    ["R", 8],
    (512, 3, 2),#downsample
    ["R", 8],
    (1024, 3, 2),#downsample
    ["R", 4],
    ["B", 3],
    (512, 1, 1),
    "P",#include (256, 3, 1) + (255, 1, 1) + predictionlayer
    (256, 1, 1),
    "U",#upsample + route
    (256, 1, 1),
    (512, 3, 1),
    ["B", 1],
    (256, 1, 1),
    "P",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    ["B", 1],
    (128, 1, 1),
    "P"
]

anchors = config.ANCHORS


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, relu = True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = (kernel_size-1)//2, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.relu = relu

    def forward(self, x):
        if self.relu:
            x = self.leaky(self.bn(self.conv(x)))
        else:
            x = self.bn(self.conv(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, repeat_nums = 1, IsRes = True):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.repeat_nums = repeat_nums
        for repeat in range(repeat_nums):
            self.layers += [nn.Sequential(
                BasicConv(channels, channels//2, 1),
                BasicConv(channels//2, channels, 3)
                )
            ]
        self.IsRes = IsRes

    def forward(self, x):
        for layer in self.layers:
            if self.IsRes:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class PredictionNet(nn.Module):
    def __init__(self, channnels, num_classes):
        super(PredictionNet, self).__init__()
        self.prep = nn.Sequential(
            BasicConv(channnels, channnels*2),
            BasicConv(channnels*2, (num_classes+5)*3, relu=False)
        )
        self.num_classes = num_classes
    def forward(self, x, index):
        x = self.prep(x)
        grids = x.shape[2]
        bs = x.shape[0]
        x = x.view(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]) # bs, 3, 85, 13, 13
        x = x.permute(0, 3, 4, 1, 2).contiguous() # bs, 13, 13, 3,85
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * 3, self.num_classes + 5)  # bs, 13 * 13 * 3, 85
        if self.training:
            return x #出来的是没有经过任何处理的 bs, 13*13*3, 5+num_class
        else:
            size = np.arange(grids)
            gridx, gridy = np.meshgrid(size, size)  # 其中size为arange类型的0-12,gridx,gridy都是np下面的数据类型，13*13
            gridx = torch.FloatTensor(gridx).view(-1, 1)
            gridy = torch.FloatTensor(gridy).view(-1, 1)  # 13*13,1
            offset_xy = torch.cat((gridy, gridx), -1).repeat(1, 3).view(-1, 2)  # 13*13,2->13*13,3*2->13*13*3,2
            anchor = anchors[2 - index]
            offset_xy = offset_xy.unsqueeze(0).repeat(bs, 1, 1).to(config.DEVICE) # 为了让offset_xy和我们前面变换得到的预测结果相匹配  1,13*13*3,2
            anchor = torch.FloatTensor(anchor).unsqueeze(0).repeat(bs, grids * grids, 1).to(config.DEVICE)  # 3,2->1,3,2->bs, 13*13*3,2
            # offset_xy = offset_xy.unsqueeze(0).repeat(bs, 1, 1)
            # anchor = torch.FloatTensor(anchor).unsqueeze(0).repeat(bs, grids * grids, 1)

            x[..., 0: 2] = (torch.sigmoid(x[..., 0: 2]) + offset_xy) * (config.IMAGE_SIZE//grids)
            x[..., 2: 4] = torch.exp(x[..., 2: 4]) * anchor
            x[..., 4: ] = torch.sigmoid(x[..., 4: ]) #confidence+classes
            return x


class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.num_classes = config.NUM_CLASSES
        self.layers = self.create_model()

    def forward(self, x):
        output = []
        route = []
        index = 0
        for layer in self.layers:
            if isinstance(layer, PredictionNet):
                output.append(layer(x, index))
                index += 1
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock):
                if layer.repeat_nums == 8:
                    route.append(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat((x, route.pop()), 1)
        out = torch.cat((output[0], output[1], output[2]), 1)
        return out

    def create_model(self):
        channels = 3
        layers = nn.ModuleList()
        for block in conf:
            if isinstance(block, tuple):
                layers.append(BasicConv(channels, block[0], kernel_size= block[1], stride= block[2]))
                channels = block[0]
            elif isinstance(block, list):
                if block[0] == "R":
                    layers.append(ResidualBlock(channels, block[1]))
                elif block[0] == "B":
                    layers.append(ResidualBlock(channels, block[1], IsRes= False))
            elif isinstance(block, str):
                if block == "U":
                    layers.append(nn.Upsample(scale_factor= 2))
                    channels = channels * 3
                elif block == "P":
                    layers.append(PredictionNet(channels, self.num_classes))
        return layers


if __name__ == "__main__":
    YOLO = YOLOv3().to(config.DEVICE)
    YOLO.train()
    image = torch.randn(3, 3, 320, 320).to(config.DEVICE)
    output = YOLO(image)
    print(output[0, 0: 30, 0:5])
    YOLO.eval()
    output = YOLO(image)
    print(output[0, 0: 30, 0:5])

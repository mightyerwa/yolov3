import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loss import Loss
import config
from model import YOLOv3
from mydataset import ZDataset
from zqnutils import(non_max_suppression,
    get_loaders)
from val import get_mAP
from tqdm import tqdm
import warnings
from config import hyp
warnings.filterwarnings("ignore")

import math
if __name__ == "__main__":
    a = torch.randn((3,5))*5
    a = a.int()
    print(a)
# #    model = YOLOv3().to(config.DEVICE)
#     print(config.DEVICE)
#     loss_fn = Loss() #loss为nn.module子类
#     train_loader, test_loader = get_loaders("../VOCdevkit/VOC2012/ImageSets/Main/zqntrain.txt", "../VOCdevkit/VOC2012/ImageSets/Main/zqntrain.txt")
# #    scaler = torch.cuda.amp.GradScaler()
#
#     model = torch.load("last_model.pt")
#     model.train()
#     for x,y in train_loader:
#         x = x.to(config.DEVICE)
#         y = y.to(config.DEVICE)
#         obj = y[0][..., 4] == 1
#         model.train()
#         out1 = model(x)
#         print(out1[0][obj])
#         model.eval()
#         out = model(x)
#
#         print(y[0][obj])
#         print(out[0][obj])
#     # model.eval()
#     # for x,y in test_loader:
#     #     print(model(x))
#     # map = get_mAP("../VOCdevkit/VOC2012/detfile", "../VOCdevkit/VOC2012/Annotations/{}.xml",
#     #               "../VOCdevkit/VOC2012/ImageSets/Main/zqntrain.txt", "../VOCdevkit/VOC2012/cache", test_loader, model)
#     # model.train()
#     print("mAP: ", map)
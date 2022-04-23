import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import config
import cv2
from model import YOLOv3
from zqnutils import (
    xmltoyolo,
    iou_width_height
)


class ZDataset(Dataset):
    def __init__(
            self,
            txtfile,
            img_dir,
            label_dir,
            mode = "training",
            transform = None
    ):
        """

        :param txtfile: (str)包含所有图像名称的txt文件（不包含.jpg和.xml）
        :param img_dir:(str)
        :param label_dir:(str)
        :param mode:training or eval
        :param transform: defalut none
        """
        self.annotations = open(txtfile).read().split("\n")[:-1] #读取数据集,last line is whitespace
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = config.ANCHORS
        self.grids = config.S
        self.image_size = config.IMAGE_SIZE
        self.numc = config.NUM_CLASSES
        self.transform = transform
        self.ignore_iou_thresh = 0.5
        self.mode = mode

    def __getitem__(self, index):
        #首先得出image，先对image进行读取，随后将BGR转为RGB，方便matplotlib，后续会加上
        #其次对于image的维度进行变化，从height,width,channels->channels,width,height，方便后续将图像直接传入model中
        image_path = os.path.join(self.img_dir, self.annotations[index] + ".jpg")
        image = cv2.resize(cv2.imread(image_path), (self.image_size, self.image_size))[:, :, ::-1].copy() # 传入image，同时将BGR格式转为RGB格式， (Height, Width, Channels)
        image = torch.from_numpy(image).permute(2, 1, 0).contiguous().to(torch.float32)
        label_path = os.path.join(self.label_dir, self.annotations[index] + ".xml")
        if self.mode == "training":
            #读取image对应的annotation文档
            #training出来的target要与model出来的一一对应
            #e.g bs, 13*13*3+26*26*3+52*52*3, 5+numc
            labels = xmltoyolo(label_path)
            labels = torch.tensor(labels)#labels是一个包含图像中所有目标的一个张量 len(labels)个目标， 每个目标有五个参数 类别 中心点x坐标 中心点y坐标 宽w 高h
            anchors = torch.tensor(self.anchors[2] + self.anchors[1] + self.anchors[0]) / self.image_size #也就是将anchors合并成一个(9, 2)的张量， 其中gridsize = 13的张量在开头
                                                                                        #主要也是为了和模型出来的时候的顺序相吻合，后面的target定义起来比较方便
            num_anchors = (self.grids[0]**2 + self.grids[1]**2 + self.grids[2]**2) * 3 #所有anchors的数量，如果是416*416的图像，num_anchors = 10647
            #第二次更新后将targets从10647,6->10647,25
            #通过阅读论文后，将损失函数由crossentropy变为binarycrossentropy
            targets = torch.zeros((num_anchors, 25)) #target和目标检测出来的结果可以一一对应，便于后续的loss的计算
                                                    #target最后一维的六个参数分别为  x y w h 是否负责检测目标1:负责\0:不负责\-1:忽略 class_label

            for label in labels:
                iouwh = iou_width_height(label[3:5], anchors) #anchors[2,1,0]排序正常,[9,2]
                rank_index = iouwh.argsort(descending = True, dim = 0)
                has_anchor = False  # 判断是否有一个anchor负责该目标的检测
                for ind in rank_index:
                    ind = int(ind)
                    scale_index = ind // 3 # 在13*13 26*26 52*52中的哪一个 0, 1, 2
                    anchor_index = ind % 3 # 在 同一个维度下使用哪一个anchor 0, 1, 2
                    size = self.grids
                    cell_x = int(label[1] * size[scale_index]) #看在哪一个gridcell里面，e.g 0.5 grids =13 6.5->6
                    cell_y = int(label[2] * size[scale_index])
                    if scale_index == 0:
                        target_index = (cell_x * size[0] + cell_y) * 3 + anchor_index
                    if scale_index == 1:
                        target_index = 3 * size[0] ** 2 + (cell_x * size[1] + cell_y) * 3 + anchor_index
                    if scale_index == 2:
                        target_index = 3 * size[0] ** 2 + 3 * size[1] ** 2 + (cell_x * size[2] + cell_y) * 3 + anchor_index
                    if has_anchor == False and targets[target_index, 4] != 1:
                        # targets 25个数：sigmoidx sigmoidy bw bh confidence 20classes
                        targets[target_index, 4] = 1
                        targets[target_index, 0] = label[1] * size[scale_index] - cell_x
                        targets[target_index, 1] = label[2] * size[scale_index] - cell_y
                        targets[target_index, 2: 4] = label[3: 5] / anchors[anchor_index]  # 把target的 w h 变为像素格式表示的形式，与模型出来的参数相吻合
        #有没有必要乘以imagesize还需要考量
                        targets[target_index, 5 + int(label[0])] = 1  # class_label
                        has_anchor = True
                    elif has_anchor == True and iouwh[ind] > self.ignore_iou_thresh and targets[target_index, 0] == 0:
                        targets[target_index, 4] = -1

                    elif has_anchor == True and iouwh[ind] < self.ignore_iou_thresh:
                        break

            return image, targets
        else:
            return image, index


    def __len__(self):
        return(len(self.annotations))


if __name__ == "__main__":
    dataset = ZDataset(
        "../VOCdevkit/VOC2012/ImageSets/Main/zqntrain.txt",
        mode = "training"
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    index = 0
    for x,y in loader:
        img = x[0].permute(2,1,0)
        print(img.shape)
        img = img.numpy().astype(np.uint8)
        cv2.imshow("img",img)
        obj = y[0][..., 4] ==1
        print(y[0][obj])
        cv2.waitKey(0)
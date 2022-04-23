import torch
from mydataset import ZDataset
from model import YOLOv3
from voc_eval import non_max_suppression, voc_eval
import os
import config
from tqdm import tqdm
import numpy as np

PASCAL_CLASSES = [
    "aeroplane",#0
    "bicycle",  #1
    "bird",     #2
    "boat",     #3
    "bottle",   #4
    "bus",      #5
    "car",      #6
    "cat",          #7
    "chair",        #8
    "cow",          #9
    "diningtable",  #10
    "dog",          #11
    "horse",        #12
    "motorbike",    #13
    "person",       #14
    "pottedplant",  #15
    "sheep",        #16
    "sofa",         #17
    "train",        #18
    "tvmonitor"     #19
]

def get_detfile(detpath, txtfile, test_loader, model):
    #先创建txt文件，保存eval的结果
    imagenames = open(txtfile).read().split("\n")[ : -1]
    if not os.path.isdir(detpath):
        os.mkdir(detpath)
    f = {}
    detpath = detpath + "/det_{}.txt"
    for i in range(20):
        path = detpath.format(PASCAL_CLASSES[i])
        f[i] = open(path,"w")
    for image, index in tqdm(test_loader, leave = True, total = len(test_loader)):
        image = image.to(config.DEVICE)
        output = model(image)
        out = non_max_suppression(output,conf_thres=0.25,iou_thres=0.45)
        for i in range(len(out)):
            for j in range(out[i].shape[0]):
                f[int(out[i][j][5].item())].write(imagenames[index[i]] + " " + "%.2f" % float(out[i][j][4]) + " " + str(int(out[i][j][0])) + " " + str(int(out[i][j][1])) + " " + str(int(out[i][j][2])) + " " + str(int(out[i][j][3])) + "\n")
    for i in range(20):
        f[i].close()


def get_mAP(detpath, annopath, txtfile, cachedir, test_loader, model,img_size):
    get_detfile(detpath, txtfile, test_loader, model)
    detpath = detpath + "/det_{}.txt"
    rec_all = []
    prec_all = []
    ap_all = []
    aap = 0
    l = len(PASCAL_CLASSES)
    for i in range(l):
        rec, prec, ap = voc_eval(detpath, annopath, txtfile, PASCAL_CLASSES[i], cachedir, img_size=img_size)
        rec_all.append(rec)
        prec_all.append(prec)
        ap_all.append(ap)
        aap += ap
    mAP = aap/l
    return mAP


#if __name__ == "__main__":
    # model = YOLOv3()
    # model.to(config.DEVICE)
    # model.eval()
    # train_loader, test_loader = get_loaders("VOC2012/test_train.txt", "VOC2012/test_train.txt")
    # map = get_mAP("VOC2012/detfile/det_{}.txt", "VOC2012/Annotations/{}.xml", "VOC2012/test_train.txt", "VOC2012/cache", test_loader, model)
    # print(map)

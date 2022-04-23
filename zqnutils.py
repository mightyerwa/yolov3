import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from torch.utils.data import DataLoader
import logging
import torchvision
import time
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



def midtocorner(pred):
    assert pred.shape[-1] == 4
    corner = pred.clone()
    corner[..., 0] = pred[..., 0] - pred[..., 2]/2
    corner[..., 1] = pred[..., 1] - pred[..., 3]/2
    corner[..., 2] = pred[..., 0] + pred[..., 2]/2
    corner[..., 3] = pred[..., 1] + pred[..., 3]/2
    corner = torch.clamp(corner, 0, config.IMAGE_SIZE)
    return corner

def xmltoyolo(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    image_size = tree.find("size")
    width = float(image_size.find("width").text)
    height = float(image_size.find("height").text)
    for obj in tree.findall('object'):
        name = obj.find('name').text  # name节点存的是class,图像类型名称
        bbox = obj.find('bndbox')  # 获取尺寸位置信息
        corner_bbox = [float(bbox.find('xmin').text),
                              float(bbox.find('ymin').text),
                              float(bbox.find('xmax').text),
                              float(bbox.find('ymax').text)]
        index = [PASCAL_CLASSES.index(name)]
        midpoint_bbox = [(corner_bbox[0] + corner_bbox[2])/2/width,
                         (corner_bbox[1] + corner_bbox[3])/2/height,
                         (corner_bbox[2]-corner_bbox[0])/width,
                         (corner_bbox[3]-corner_bbox[1])/height]
        objects.append(index + midpoint_bbox)
    return objects

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    image_size = tree.find("size")
    width = float(image_size.find("width").text)
    height = float(image_size.find("height").text)
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text  # name节点存的是class,图像类型名称
        obj_struct['pose'] = obj.find('pose').text  # 默认为Unspecified
        obj_struct['truncated'] = int(obj.find('truncated').text)  # 默认为0
        obj_struct['difficult'] = int(obj.find('difficult').text)  # 默认为0
        bbox = obj.find('bndbox')  # 获取尺寸位置信息
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text) / width * 416),
                              int(float(bbox.find('ymin').text) / height * 416),
                              int(float(bbox.find('xmax').text) / width * 416),
                              int(float(bbox.find('ymax').text) / height * 416)]
        objects.append(obj_struct)

    return objects

def draw_pr(rec, prec, classname):
    plt.figure()
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("PR curve")
    plt.plot(rec.tolist(), prec.tolist())
    plt.savefig("{}_pr.png".format(classname))

def voc_ap(rec, pre):
    """
    说白了就是给两个包含rec和pre的一一对应的ndarray，计算积分->最终的map
    :param rec(numpy.ndarray): recall
    :param prec(numpy.ndarray): precision
    :return: ap
    """
    #将recall和precision补全，主要用于积分计算，保证recall的域为[0,1]
    mrec = np.concatenate(([0.],rec,[1.]))
    mpre = np.concatenate(([0.],pre,[0.]))

    #不知道具体含义，后续需要补充，听说是：消除fp增加条件下导致的pre减小的无效值
    for i in range(mpre.size -1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

    #消除总检测样本增加导致的计算的recall为增加的量,依据函数的意思是相邻两个recall如果相同，则消除相同的recall
    # i 为index
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh = 0.5):
    """
    最高级的进行PASCALVOC评估的函数？？？？
    :param detpath: (classname) should produce the detection results file
    :param annopath: (imagename) should be xml annotations file ././{}.format(imagenames)
    :param imagesetfile: txt file containing the list of images, one image per line
    :param classname: category name(duh)
    :param cachedir:Directory for caching the annotations
    :param vthresh: overlap threshold(default = 0.5
    :return:
    """
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")

    with open(imagesetfile,'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]#消除\n，但是为什么不直接用read().split(\n)???

    if not os.path.isfile(cachefile):
        # 载入标签文件，recs这个字典中，存储了验证集所有的GT信息
        recs = {}
        for i, imagename in enumerate(imagenames):
            # 解析标签xml文件，annopath为/{}.xml文件，加format表示为{}中赋值
            # imagename来源于从imagesetfile中提取，循环采集所有的的信息
            recs[imagename] = parse_rec(annopath.format(imagename))
            # 解析标签文件图像进度条
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # 将读取标签内容存入缓存文件annots.pkl，这是个数据流二进制文件
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)  # 使用了pickle.dump，存入后保存成二进制文件
    else:
        # 有标签缓存文件，直接读取,recs中存的是GT标签信息
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
        #recs imagename的字典，字典内容：object,bbox,difficulty???
        #################################################################################################
        ##### 第二步：从字典recs中提取当前类型的GT标签信息，存入字典class_recs中，key为图片名imagename #####
        #################################################################################################
        # 针对某个class名称的result文件提取对应的每个图片文件中GT的信息，存入R
        # bbox中保存该类型GT所有的box信息，difficult、det、npos等都从R中提取
        # 提取完毕针对每个图片生成一个字典，存入class_recs
        # 这里相当于根据class对图片中新不同类型目标进行归纳，每个类型计算一个AP
    class_recs = {}
    npos = 0

    # 上篇文章中说了当result文件名前面含有comp4_det_test_时的2种方法，这里还有个更简单的，即将classname后加上[15:]
    # 表示读取第15位开始到结束的内容，这是第3种方法
    for imagename in imagenames:
        # R中为所有图片中，类型匹配上的GT信息
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # bbox中存储了该文件中该类型的所有box信息
        bbox = np.array([x['bbox'] for x in R])
        # bbox为所有classname所代表的的类别的bbox左上右下坐标
        # difficult转化成bool型变量，其中xml文件中difficult的含义，表示目标检测的难度，如果为1的话，表示难检测出来。模型检测不出来，也不会把它当做漏检测
        #diffcult的顺序和bbox的顺序一样，一一对应的关系
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # 该图片中，没有匹配到当前类型det=[],匹配到1个，det=[False]，匹配到多个det=[False, ...]
        # det将是和difficult不同的地方，当不是difficult的时候，det也是false，这是一个区别
        det = [False] * len(R)
        # 利用difficult进行计数，这里所有的值都是difficult，如果不是difficult就累加，~是取反。其实就是统计xml文件中<difficult>0</difficult>的个数，表示gt的个数，便于求recall。
        npos = npos + sum(~difficult)
        # class_recs是一个字典，第一层key为文件名，一个文件名对应的子字典中，存储了key对应的图片文件中所有的该类型的box、difficult、det信息
        # 这些box、difficult、det信息可以包含多个GT的内容
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    #################################################################################################
    ##### 第三步：从当前class的result文件中读取结果，并将结果按照confidence从大到小排序 ################
    #####        排序后的结果存在BB和image_ids中                                      ################
    #################################################################################################

    # 读取当前class的result文件内容，这里要去result文件以class命名
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # 删除result文件中的''，对于非voc数据集，有的就没有这些内容
    splitlines = [x.strip().split(' ') for x in lines]
    # 将每个结果条目中第一个数据，就是图像id,这个image_ids是文件名
    image_ids = [x[0] for x in splitlines]
    # 提取每个结果的置信度，存入confidence
    confidence = np.array([float(x[1]) for x in splitlines])
    # 提取每个结果的结果，存入BB，[[x,y,w,h or zuoshang,youxia],[]]
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # 对confidence从大到小排序，获取id
    sorted_ind = np.argsort(-confidence)
    # 获得排序值，这个值后来没有再用过
    sorted_scores = np.sort(-confidence)
    # 按confidence排序对BB进行排序
    BB = BB[sorted_ind]
    # 对相应的图像的id进行排序，其实每个图像对应一个id，即对应一个目标，当一个图中识别两个相同的GT,是可以重复的
    # 这样image_ids中，不同位置就会有重复的内容
    image_ids = [image_ids[x] for x in sorted_ind]

    #################################################################################################
    ##### 第四步：对比GT参数和result，计算出IOU，在fp和tp相应位置标记1 #################################
    #################################################################################################

    # go down dets and mark TPs and FPs
    nd = len(image_ids)  # 图像id的长度
    tp = np.zeros(nd)  # 设置TP初始值
    fp = np.zeros(nd)  # 设置FP初始值

    # 对一个result文件中所有目标进行遍历，每个图片都进行循环，有可能下次还会遇到这个图片，如果
    for d in range(nd):
        # 提取排序好的GT参数值，里面可以有多个目标，当image_ids[d1]和image_ids[d2]相同时，两个R内容相同，且都可能存了多个目标信息
        R = class_recs[image_ids[d]]
        # 将BB中confidence第d大的BB内容提取到bb中，这是result中的信息，只可能包含一个目标
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        # BBGT就是当前confidence从大到小排序条件下，第d个GT中bbox中的信息
        BBGT = R['bbox'].astype(float)

        # 当BBGT中有信息，就是没有虚警目标，计算IOU
        # 当一个图片里有多个相同目标，选择其中最大IOU，GT和检测结果不重合的IOU=0
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            # 大于0就输出正常值，小于等于0就输出0
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni  # 计算交并比，就是IOU
            ovmax = np.max(overlaps)  # 选出最大交并比，当有
            jmax = np.argmax(overlaps)  # 求出两个最大交并比的值的序号

        # 当高于阈值，对应图像fp = 1
        # ovmax > ovthresh的情况肯定不存在虚警，ovmax原始值为-inf，则没有目标肯定不可能进入if下面的任务
        if ovmax > ovthresh:
            # 如果不存在difficult，初始状态，difficult和det都是False
            # 找到jamx后，第一任务是确定一个tp，第二任务就是将R['det'][jmax]置为1，表示为已检测，下次再遇到就认为是fp
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1  # 标记为已检测
                else:
                    fp[d] = 1.  # 一个目标被检测两次
        else:
            fp[d] = 1.

    #################################################################################################
    ##### 第五步：计算ap,rec，prec ###################################################################
    #################################################################################################
    ##difficult用于标记真值个数，prec是precision，rec是recall
    # compute precision recall
    fp = np.cumsum(fp)  # 采用cumsum计算结果是一种积分形式的累加序列，假设fp=[0,1,1],那么np.cumsum(fp)为[0,1,2]。
    tp = np.cumsum(tp)
    # print("float(npos):", float(npos))
    rec = tp / float(npos)  # npos表示gt的个数
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # maximum这一大串表示防止分母为0
    ap = voc_ap(rec, prec)

    return rec, prec, ap

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def get_loaders(train_path, test_path, batch_size):
    from mydataset import ZDataset
    train_dataset = ZDataset(
        train_path,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
    )
    test_dataset = ZDataset(
        test_path,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        mode = "testing",
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

#     train_eval_dataset = YOLODataset(
#         train_csv_path,
# #        transform=config.test_transforms,
#         grids=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
#         img_dir=config.IMG_DIR,
#         label_dir=config.LABEL_DIR,
#         anchors=config.ANCHORS,
#     )
#     train_eval_loader = DataLoader(
#         dataset=train_eval_dataset,
#         batch_size=config.BATCH_SIZE,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY,
#         shuffle=False,
#         drop_last=False,
#     )

    return train_loader, test_loader



if __name__ == "__main__":
    a,b,c =voc_eval("../VOCdevkit/VOC2012/detfile/det_{}.txt",
                 "../VOCdevkit/VOC2012/Annotations/{}.xml",
                 "../VOCdevkit/VOC2012/ImageSets/Main/zqntrain.txt",
                 "person",
                 "../VOCdevkit/VOC2012/cachefile")
    print(c)





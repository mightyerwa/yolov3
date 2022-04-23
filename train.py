import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.callbacks import Callbacks
from torch.optim import SGD, Adam, AdamW, lr_scheduler
import numpy as np
from loss import Loss
from model import YOLOv3
from mydataset import ZDataset
from zqnutils import(
    get_loaders)
from val import get_mAP
from tqdm import tqdm
import warnings
from thop import profile
import math
import time
#from pytorch_model_summary import summary
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
from config import hyp
from utils.callbacks import Callbacks

def main(opt):
    device = opt.device
    epochs = opt.epochs
    batch_size = opt.batch_size
    img_size = opt.img_size
    model = YOLOv3().to(opt.device)
    if opt.load_model == True:
        model = torch.load("last_model.pt", map_location='cpu') # load checkpoint to CPU to avoid CUDA memory leak
        model.to(opt.device)

    print(opt.device)

    model.train()

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    if opt.optimizer == 'Adam':
        optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)

    del g

    lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (hyp["lrf"] - 1) + 1 # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    start_epoch, best_fitness = 0, 0.0
    train_loader, test_loader = get_loaders("../VOCdevkit/VOC2012/ImageSets/Main/trainval.txt",
                                            "../VOCdevkit/VOC2012/ImageSets/Main/val.txt",batch_size)
    nbs = 64
    nb = len(train_loader)  # number of batches
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    mAP = np.zeros(20)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    
    # lr0 = config.LEARNING_RATE[0]#0.001
    # lr1 = list(map(lambda x:x/4000, range(40)))#0.001-0.01
    # lr2 = config.LEARNING_RATE[1]#0.01
    # lr3 = config.LEARNING_RATE[2]#0.0001


 #   optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=config.WEIGHT_DECAY)
    loss_fn = Loss()  # loss
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, epochs):

        model.train()
        losses = []  # mean losses
        pbar = enumerate(train_loader)
        
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()

        pbar.set_description("epoches:{}".format(epoch))
        for i, (imgs, targets) in pbar:
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            imgs = imgs.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                output = model(imgs)
                loss = loss_fn(output, targets)

            scaler.scale(loss).backward()

            losses.append(loss.item())  # item()什么用？ 将torch类型转化为浮点数


            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni
            # scaler.step(optimizer)  # optimizer.step
            # scaler.update()
            # optimizer.zero_grad()
            # scheduler.step()
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            mean_loss = sum(losses) / len(losses)
            pbar.set_postfix(loss = mean_loss)
            pbar.update(1)
#            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()
        if epoch == epochs - 1:
            flops, params = profile(model, inputs=(input,))
            print("flops", flops)
            print("params", params)

        if epoch > 0 and epoch % 3 ==0:
            if epoch % 20 == 0:
                torch.save(model,"last_model.pt")
            model.eval()
            meanAP = get_mAP("../VOCdevkit/VOC2012/detfile", "../VOCdevkit/VOC2012/Annotations/{}.xml", "../VOCdevkit/VOC2012/ImageSets/Main/val.txt", "../VOCdevkit/VOC2012/cache", test_loader, model,img_size)
            model.train()
            print("mAP: ",meanAP)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--img-size', type=int, default=416, help='img_size of the image to resize')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', default='cuda:1', help='cuda device, i.e. cuda:0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--load_model', action='store_true')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)







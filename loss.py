import torch
import torch.nn as nn

import config
import random

class YOLOMSELoss(nn.Module):
    def __init__(self, reduction = "mean"):
        super(YOLOMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, wh):
        pred = pred.view(-1, 4)
        num = len(pred)
        target = target.view(-1, 4)
        wh = wh.view(-1, 2)
        weights = 2 - wh[:, 0] * wh[:, 1]
        weights = weights.unsqueeze(-1).repeat(1, 4)
        sum = torch.sum(weights*(pred - target)**2)
        if self.reduction == "mean":
            return sum
        else:
            mean = sum/num
            return mean



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction = "mean")
#        self.entropy = nn.CrossEntropyLoss()
        self.bcecls = nn.BCEWithLogitsLoss(reduction= 'mean')
        self.bceobj = nn.BCEWithLogitsLoss(reduction= 'mean')
        self.bce = nn.BCEWithLogitsLoss(reduction= 'mean')
        self.sigmoid = nn.Sigmoid()
        self.mesloss = nn.MSELoss(reduction = "mean")

        self.lamada_noobj = 64.3
        self.lamada_obj = 1
        self.lamada_coordinate = 1
        self.lamada_class = 37.4

    def forward(self, prediction, target):
        """
        计算损失
        :param prediction:tensor [batch_size, 5 + num_classes] 预测的模型框架，在这个函数内使用了像素表示，默认图像大小416*416， x y w h condidence classes
        training的时候prediction没有进行任何后续的处理
        :param target: tensor [batch_size , 5 + num_classes] 目标框架，同样采用像素表示，默认图像大小416*416， x y w h obj class_label
        :param anchors:
        :return: 总损失
        """
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0
        

        anchors = torch.cat((torch.tensor(config.ANCHORS[2]).repeat(config.S[0]**2, 1),torch.tensor(config.ANCHORS[1]).repeat(config.S[1]**2, 1),torch.tensor(config.ANCHORS[0]).repeat(config.S[2]**2, 1)), dim=0).to(config.DEVICE)

        #noobject loss
        noobj_loss = self.lamada_noobj * self.bce(prediction[..., 4][noobj], target[..., 4][noobj])

        # object loss 在yolov3中confidence的gt值始终为1/0
        obj_loss = self.lamada_obj * self.bceobj(prediction[..., 4][obj], target[..., 4][obj])

        # class loss
        class_loss = self.lamada_class * self.bcecls(prediction[..., 5:][obj], target[..., 5:][obj])

        # coordinate loss
        prediction[..., 0: 2][obj] = self.sigmoid(prediction[..., 0: 2][obj]) #sigma(x, y)
        wh = target[..., 2: 4] * anchors /config.IMAGE_SIZE
        target[..., 2: 4][obj] = torch.log(target[..., 2: 4][obj] + 1e-16)
        coordinate_loss = self.lamada_coordinate * \
                          self.mse(prediction[..., 0: 4][obj], target[..., 0: 4][obj] )#, wh[obj])
        msel = nn.MSELoss()(prediction[..., 0: 4][obj], target[..., 0: 4][obj])
        # print("___________")
        # print("noobjloss: ", noobj_loss)
        # print("objloss: ", obj_loss)
        # print("classloss: ", class_loss)
        # print("coordinate_loss: ", coordinate_loss)
        # print("coordinatemse: ", msel)
        # print("___________")

        return (noobj_loss + obj_loss + class_loss + coordinate_loss, noobj_loss, obj_loss, class_loss, coordinate_loss)

if __name__ == "__main__":
     target = torch.randn(3, 60, 4)
     pred = torch.randn(3,60,4)
     weight = torch.ones(3,60,2) * 0.5
     a = YOLOMSELoss()(pred, target, weight)
     b = nn.MSELoss()(pred, target) * 1.75
     print(a,b)

import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
from nets.vgg16 import vgg16
from nets.alexnet import alexnet
from nets.densenet import densenet121
from nets.densenet import densenet161
from nets.densenet import densenet169
from nets.densenet import densenet201
from nets.inception import inception_v3
from nets.shufflenetv2 import shufflenet_v2_x0_5
from nets.shufflenetv2 import shufflenet_v2_x1_0
from nets.shufflenetv2 import shufflenet_v2_x1_5
from nets.shufflenetv2 import shufflenet_v2_x2_0
from nets.googlenet import googlenet
from nets.resnet import resnet34
from nets.resnet import resnet101
from nets.resnet import resnet152
from utils.utils import weights_init
from utils.dataloader import DataGenerator, detection_collate

get_model_from_name = {
    "mobilenet" : mobilenet_v2,
    "resnet50"  : resnet50,
    "vgg16"     : vgg16,
    "alexnet"   :alexnet,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "inception_v3": inception_v3,
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
    "googlenet": googlenet,
    "resnet34": resnet34,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

freeze_layers = {
    "mobilenet" :81,
    "resnet50"  :173,
    "vgg16"     :19,
}


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,gen2,gen_val2,gen3,gen_val3,Epoch,cuda):
    total_loss = 0
    total_accuracy = 0
    confusion = ConfusionMatrix(num_classes=4, labels=labels)
    val_toal_loss = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for i, data in enumerate(zip(gen, gen2, gen3)):
            images, targets = data[0][0], data[0][1]
            text, targets2 = data[1][0], data[1][1]
            velocity, targets3 = data[2][0], data[2][1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                targets2 = torch.from_numpy(targets2).type(torch.FloatTensor).long()
                targets3 = torch.from_numpy(targets3).type(torch.FloatTensor).long()
                text = torch.from_numpy(text).type(torch.FloatTensor)
                velocity = torch.from_numpy(velocity).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()
                    targets2 = targets2.cuda()
                    targets3 = targets3.cuda()
                    velocity = velocity.cuda()
                    text = text.cuda()

            optimizer.zero_grad()

            outputs1 = net(images)
            outputs2 = net(velocity)
            #outputs3 = net(text)
            outputs = outputs1 + outputs2 #+ outputs3
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()
                outputs = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
                confusion.update(outputs.to("cpu").numpy(), targets.to("cpu").numpy())
    
    
            pbar.set_postfix(**{'total_loss': total_loss / (i + 1),
                                'accuracy'  : total_accuracy / (i + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            global list1
            list1.append(total_accuracy / (i + 1))
            global list2
            list2.append(total_loss / (i + 1))
            pbar.update(1)
            with open("test1.txt", "w") as f:
                for i in list1:
                    f.write(str(i).strip(','))
                    f.write('\n')
            with open("test2.txt", "w") as f:
                for i in list2:
                    f.write(str(i).strip(','))
                    f.write('\n')

        confusion.summary()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for i, data in enumerate(zip(gen, gen2, gen3)):
            images, targets = data[0][0], data[0][1]
            text, targets2 = data[1][0], data[1][1]
            velocity, targets3 = data[2][0], data[2][1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                targets2 = torch.from_numpy(targets2).type(torch.FloatTensor).long()
                targets3 = torch.from_numpy(targets3).type(torch.FloatTensor).long()
                text = torch.from_numpy(text).type(torch.FloatTensor)
                velocity = torch.from_numpy(velocity).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()
                    targets2 = targets2.cuda()
                    targets3 = targets3.cuda()
                    velocity = velocity.cuda()
                    text = text.cuda()

                optimizer.zero_grad()

                outputs1 = net(images)
                outputs2 = net(velocity)
                outputs3 = net(text)
                outputs = outputs1 + outputs2 + outputs3
                val_loss = nn.CrossEntropyLoss()(outputs, targets)
                #val_loss = nn.CrossEntropyLoss()(outputs, targets)
                
                val_toal_loss += val_loss.item()
                
            pbar.set_postfix(**{'total_loss': val_toal_loss / (i + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#----------------------------------------#
#   主函数
#----------------------------------------#
if __name__ == "__main__":


    log_dir = "./logs/"
    #---------------------#

    list1 = []
    list2 = []

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]

    #   所用模型种类
    #---------------------#
    backbone = "densenet121"
    #---------------------#
    #   输入的图片大小
    #---------------------#
    input_shape = [224,224,3]
    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = True

    #-------------------------------#
    #   是否使用网络的imagenet
    #   预训练权重
    #-------------------------------#
    pretrained = False

    classes_path = './model_data/cls_classes.txt' 
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    assert backbone in ["mobilenet", "resnet50", "vgg16", "alexnet", "densenet121", "densenet161", "densenet169",
                        "densenet201", "inception_v3", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5",
                        "shufflenet_v2_x2_0", "googlenet", "resnet34", "resnet101", "resnet152"]

    model = get_model_from_name[backbone](num_classes=num_classes,pretrained=pretrained)
    if not pretrained:
        weights_init(model)

    #------------------------------------------#
    #   注释部分可用于断点续练
    #   将训练好的模型重新载入
    #------------------------------------------#
    # # 加快模型训练的效率
    # model_path = "model_data/Omniglot_vgg.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    with open(r"./cls_train.txt","r") as f:
        lines = f.readlines()

    with open(r"./cls_train2.txt", "r") as f:
        lines2 = f.readlines()

    with open(r"./cls_train3.txt", "r") as f:
        lines3 = f.readlines()

    np.random.seed(10101)
    #np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val
    num_val2 = int(len(lines2) * 0.1)
    num_train2 = len(lines2) - num_val2

    num_val3 = int(len(lines3) * 0.1)
    num_train3 = len(lines3) - num_val3

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
        
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr              = 1e-2
        Batch_size      = 16
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        
        optimizer       = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_dataset   = DataGenerator(input_shape,lines[:num_train])
        val_dataset     = DataGenerator(input_shape,lines[num_train:], False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

        train_dataset2 = DataGenerator(input_shape, lines2[:num_train])
        val_dataset2 = DataGenerator(input_shape, lines2[num_train:], False)
        gen2 = DataLoader(train_dataset2, batch_size=Batch_size, num_workers=2, pin_memory=True,
                          drop_last=True, collate_fn=detection_collate)
        gen_val2 = DataLoader(val_dataset2, batch_size=Batch_size, num_workers=2, pin_memory=True,
                              drop_last=True, collate_fn=detection_collate)

        train_dataset3 = DataGenerator(input_shape, lines3[:num_train])
        val_dataset3 = DataGenerator(input_shape, lines3[num_train:], False)
        gen3 = DataLoader(train_dataset3, batch_size=Batch_size, num_workers=2, pin_memory=True,
                          drop_last=True, collate_fn=detection_collate)
        gen_val3 = DataLoader(val_dataset3, batch_size=Batch_size, num_workers=2, pin_memory=True,
                              drop_last=True, collate_fn=detection_collate)

        epoch_size      = train_dataset.get_len()//Batch_size
        epoch_size_val  = val_dataset.get_len()//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        #model.freeze_backbone()

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, gen2,gen_val2,gen3,gen_val3,Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr              = 1e-2
        Batch_size      = 16
        Freeze_Epoch    = 50
        Epoch           = 100

        optimizer       = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_dataset   = DataGenerator(input_shape,lines[:num_train])
        val_dataset     = DataGenerator(input_shape,lines[num_train:], False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        train_dataset2 = DataGenerator(input_shape, lines2[:num_train])
        val_dataset2 = DataGenerator(input_shape, lines2[num_train:], False)
        gen2 = DataLoader(train_dataset2, batch_size=Batch_size, num_workers=2, pin_memory=True,
                          drop_last=True, collate_fn=detection_collate)
        gen_val2 = DataLoader(val_dataset2, batch_size=Batch_size, num_workers=2, pin_memory=True,
                              drop_last=True, collate_fn=detection_collate)

        train_dataset3 = DataGenerator(input_shape, lines3[:num_train])
        val_dataset3 = DataGenerator(input_shape, lines3[num_train:], False)
        gen3 = DataLoader(train_dataset3, batch_size=Batch_size, num_workers=2, pin_memory=True,
                          drop_last=True, collate_fn=detection_collate)
        gen_val3 = DataLoader(val_dataset3, batch_size=Batch_size, num_workers=2, pin_memory=True,
                              drop_last=True, collate_fn=detection_collate)

        epoch_size      = train_dataset.get_len()//Batch_size
        epoch_size_val  = val_dataset.get_len()//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        model.Unfreeze_backbone()

        for epoch in range(Freeze_Epoch,Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, gen2,gen_val2,gen3,gen_val3,Freeze_Epoch, Cuda)
            lr_scheduler.step()

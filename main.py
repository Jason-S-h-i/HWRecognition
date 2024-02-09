import dataset.MNIST.dataset_analysis as ds_analyze
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

#重写数据集类
class train_Dataset(Dataset):
    def __init__(self):
        self.train_image = ds_analyze.load_train_images()
        self.train_label = torch.tensor(ds_analyze.load_train_labels(), dtype=torch.long)

    def __getitem__(self, index):
        img = torch.Tensor(self.train_image[index])
        target = self.train_label[index]
        return img, target

    def __len__(self):
        return len(self.train_label)

class test_Dataset(Dataset):
    def __init__(self):
        self.test_image = ds_analyze.load_train_images()
        self.test_label = torch.tensor(ds_analyze.load_train_labels(), dtype=torch.long)
        self.test_len = len(self.test_label)

    def __getitem__(self, index):
        img = torch.Tensor(self.test_image[index])
        target = self.test_label[index]
        return img, target

    def __len__(self):
        return len(self.test_label)

#建立模型
class LeNet(nn.Module):
    def __init__(self, transform=None):
        super(LeNet, self).__init__()
        #输入图像为28*28，需要padding
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5*5*16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)


    #神经网络输入到输出的函数
    def forward(self,input):
        input = F.sigmoid(self.conv1(input))
        input = self.maxpool1(input)
        input = F.sigmoid(self.conv2(input))
        input = self.maxpool2(input)
        input = self.flatten(input)
        input = F.sigmoid(self.linear1(input))
        input = F.sigmoid(self.linear2(input))
        output = self.linear3(input)
        return output



#if __name__ == '__main__':
    # f_debugger = 0
    # train_image = ds_analyze.load_train_images(debugger=f_debugger)
    # ds_analyze.show_pic(train_image[0])
    # train_label = ds_analyze.load_train_labels(debugger=f_debugger)
    # test_image = ds_analyze.load_test_images(debugger=f_debugger)
    # test_label = ds_analyze.load_test_labels(debugger=f_debugger)
#------------------MAIN------------------------------------MAIN------------------------------------MAIN------------------
#超参数
Epoch = 5
Batch_Size = 16
Learning_Rate = 0.08
SGD_mometum = 0.9
device = torch.device("cpu")

#添加TensorBoard
#Command tensorboard --logdir=logs_train --port=6007
writer = SummaryWriter("./logs_train")

#构建Dataset数据集实例
train_Dataset = train_Dataset()
test_Dataset = test_Dataset()

#构建Dataloader
train_Dataloader = DataLoader(train_Dataset, batch_size=Batch_Size, shuffle=True, num_workers=0)
test_Dataloader = DataLoader(test_Dataset, batch_size=Batch_Size, shuffle=True, num_workers=0)

#网络实例化
lenet = LeNet()
#损失函数
loss_function = nn.CrossEntropyLoss()
#优化器
optimizer = optim.SGD(lenet.parameters(), lr=Learning_Rate, momentum=SGD_mometum)

#参数作记录使用
total_train_step = 0
total_test_step = 0
num_train = train_Dataset.train_label.__len__()
num_test = test_Dataset.test_label.__len__()


for epoch in range(Epoch):
    train_loss = 0.0
    print("-------------第 {} 轮训练-------------".format(epoch+1))

    start_time = time.time()
    #训练
    lenet.train()
    for data in train_Dataloader:
        #将数据读取出来输入网络，并计算损失函数
        imgs, targets = data
        output = lenet(imgs)
        train_loss = loss_function(output, targets)
        #优化器设置+反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            print("训练次数：{}".format(total_train_step))
    end_time = time.time()
    print("on {} train_time:{}".format(device, end_time-start_time))

    #测试
    lenet.eval()
    total_testloss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_Dataloader:
            imgs, targets = data
            output = lenet(imgs)
            test_loss = loss_function(output, targets)
            #数据计算
            total_testloss = total_testloss + test_loss.item()
            total_accuracy = total_accuracy + torch.sum(torch.argmax(output, dim=1)== targets)
    print("total_test_loss：{}".format(total_testloss))
    print("train_loss:{}".format(train_loss.item()))
    print("test_loss：{}".format(test_loss))
    print("test_accu: {}".format(total_accuracy/num_test))

    total_test_step = total_test_step + 1
    writer.add_scalars('model_eval',
                      { 'train_loss':train_loss,
                        'test_loss':test_loss,
                        'test_accu':total_accuracy/num_test},
                      total_test_step)


    #保存模型
    torch.save(lenet, "./model/lenet_{}.pth".format(epoch+1))
    print("The model of epoch{} have saved.".format(epoch+1))

writer.close()
#------------------MAIN------------------------------------MAIN------------------------------------MAIN------------------

# Pytorch的使用

# Dataset
重写getitem和len方法

getitem获取数据集所有的图像和图像所对应的标签

应该得到一个图像矩阵（数量×通道数×宽×高）、一个标签矩阵（数量）

len返回数据集的数量

## 创建方式
对于图像分类任务，图像+分类

对于目标检测任务，图像+bbox、分类

对于超分辨率任务，低分辨率图像+超分辨率图像

对于文本分类任务，文本+分类




# TensorBoard

`from torch.utils.tensorboard import SummaryWriter`

ctrl+左键，在pycharm中可以点入文件进行查看

```python
#建立对象，传入文件夹名
writer = SummaryWriter("logs")

# 常用方法

#加入图像
writer.add_image()
# tag (str): 图标标题
# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): 图像数据
# global_step (int): x轴值

#加入图表
writer.add_scalar()
# tag (str): 图标标题
# scalar_value (float or string/blobname): y轴值
# global_step (int): x轴值

writer.close()
```

```commandline
tensorboard --logdir=logs(文件名) --port=6007
```

在命令行中 指定路径、端口号，可以打开刚刚画的图

# Transforms

导入包

`from torchvision import transforms`

transforms该如何使用

```python
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
#使用transforms中的ToTensor类，将img图片转化为张量
```

python的__call__相当于c++中的重载函数，也就是在创建对象的同时，执行一次此函数

compose类：将多个transform类组合在一起

ToTensor类:将PIL图像或narray转换为张量并相应地缩放至0-1

PILToTensor类：将PIL图像转换为相同类型的张量，不会缩放值

Normalize类：用均值和标准差归一化张量图像

Resize类：将输入图像的大小调整为给定的大小

RandomCrop类：在随机位置裁剪给定的图像。

# dataloader

```python
class
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, 
batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, 
timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, 
prefetch_factor=None, persistent_workers=False, pin_memory_device='')
```
dataset实例化后传入

batch_size在一轮训练中每次训练的个数

shuffle 打乱，每个epoch数据打乱传入

num_workers 传入数据时使用的进程数

drop_last 当数据集每次以batchsize传入时，若除不尽，最后一个批次是否需要传入


dataloader传入神经网络
```python
dataloader = Dataloader(dataset, batch_size=1)

for date in dataloader:
    imgs, targets = data
    outputs = net_name(imgs)
```

# 使用tensoboard展示网络

```python
writer = SummaryWriter("../logs_seq")
writer.add_graph(net_name, input)
writer.close()
```


# 损失函数、反向传播、优化器
## 损失函数

nn.L1Loss

平均绝对误差(Mean Absolute Error)

$\frac{\sum_{i=1}^{n}\lvert y_i-x_i \rvert$

nn.MSELoss

均方差

nn.CrossEntropyLoss

交叉熵误差

通常是衡量分类问题误差

## 误差函数、反向传播的使用

```python
loss = nn.CrossEntropyLoss()

result_loss = loss(outputs, targets)
result_loss.backward()
```
## 优化器

torch.optim

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)

for input, target in dataset:
    optimizer.zero_grad() #必须写，很重要
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```


# 网络训练架构

```python
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        ...
        running_loss = running_loss + result_loss
    print(running_loss)
```


# 网络的模型与加载

## 保存

```python
torch.save()

torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)

torch.save(net_name, "net_name_.pt")

```

此种保存方式，保存了模型的结构和参数

将对象保存到磁盘文件。

Parameters

obj (object):保存对象

f (Union[str, PathLike, BinaryIO, IO[bytes]]):类文件对象(必须实现写入和刷新)或字符串或操作系统。包含文件名的类路径对象

pickle_module (Any):用于pickle元数据和对象的模块

pickle_protocol (int):可以指定覆盖缺省协议


另一种保存方式

```python
import torch

torch.save()

torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)

torch.save(net_name.state_dict(), "net_name_.pt")
# 保存为python字典形式的模型参数
# 以此种方式加载模型
# 创建相同的网络模型
net = net_name()
# 加载
net.load_state_dict(torch.load("net_name_.pt"))
```


## 加载

```python
torch.load()

torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=False, mmap=None, **pickle_load_args)

net = torch.load("net_name_.pt")

```

# 训练的套路

1. 准备数据集，训练数据集和测试数据集

使用datasets重写方法，读取自己的数据集

2. 利用 Dataloader 来加载数据集，包括训练数据集和测试数据集
3. 搭建神经网络

```python
import torch.nn as nn
import torch.nn.functional as F

class Net_Name(nn.Module):
    def __init__(self):
        super(Net_Name, self).__init()
        ###此部分可以依照需求自定义
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5*5*16, 120)
        ###此部分可以依照需求自定义

    #神经网络输入到输出的函数
    def forward(self,input):
        ###此部分可以依照需求自定义
        input = F.sigmoid(self.conv1(input))
        input = self.maxpool1(input)
        input = F.sigmoid(self.linear2(input))
        input = self.linear3(input)
        ###此部分可以依照需求自定义
        
        #记得return
        return input
```

4. 创建网络，即实例化网络
5. 选择损失函数
6. 选择优化器。有参数要传入：网络的参数，学习率，优化器的参数。
7. 设置训练网络的一些参数：记录训练的次数、记录测试的次数、训练的轮次
8. 两个for循环的嵌套构成训练流程。外层为epoch的循环，内层为data的循环

内层for循环具体步骤

从dataloader的data中获取每个data的imgs和targets

输入神经网络得到输出

计算输出与targets的损失值

对优化器的梯度清零。这一步是由于pytorch会默认对梯度进行累加，需要手动对其清零

反向传播

加入优化器

训练次数加1

9. 测试代码的编写。在每轮训练完成后可以对网络进行测试，目的是利用测试集模拟真实情况下，网络模型的识别能力
```python
#测试步骤开始
total_test_loss = 0
#不计算梯度
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        outputs = net_name(imgs)
        Loss = Loss_fn(outputs,targets)
        total_test_loss = total_test_loss + Loss.item()
        print("整体测试集上的Loss: ".format(total_test_loss))
```
10. 可以加入TensorBoard可视化训练过程
11. 保存每一轮训练的结果

# 评价网络的指标
## 训练时损失函数的数值

## 训练正确率

## 测试正确率

# GPU训练
## 第一种
 网络模型、损失函数、数据
调用GPU

```python
# 在所有前面加上
torch.cuda.is_available()

net_name = net_name.cuda()
loss_fn = loss_fn.cuda()

imgs = imgs.cuda()
targets = targets.cuda()
```


## 第二种

```python
something.to(device)

device = torch.device("cpu")
device = torch.device("cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_name = net_name.to(device)
loss_fn = loss_fn.to(device)

imgs = imgs.to(device)
targets = targets.to(device)
```

# 完整的模型应用
1. 读取本地图片，转化为tensor类型
2. 读取保存的模型
3. 将图片输入模型
4. 打印输出

2、3步中间加入
```python
model.eval()
with torch.no_grad():
```

分类问题的输出。output.argmax(1)

若在测试中，保存的模型与现在使用的模型，计算时cpu和gpu不同的话，则须在torch.load上传入参数map_location = torch.device('cpu')

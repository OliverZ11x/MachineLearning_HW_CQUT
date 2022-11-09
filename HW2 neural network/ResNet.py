#randomcrop


import torch
from bs4 import Tag
from matplotlib.font_manager import stretch_dict
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

"""
卷积运算 使用mnist数据集，和10-4，11类似的，只是这里：1.输出训练轮的acc 2.模型上使用torch.nn.Sequential
"""
# Super parameter ------------------------------------------------------------------------------------
batch_size = 128
learning_rate = 0.01
momentum = 0.5
EPOCH = 3

# Prepare dataset ------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)) # 其中0.1307是mean均值和0.3081是std标准差
#                                ,transforms.RandomCrop((24,24))# 数据增强，数据集随机处理
                               ])
train_dataset = datasets.MNIST(root='./data/mnist',
                               train=True, transform=transform, download=True)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/mnist',
                              train=False, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# fig = plt.figure()
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.tight_layout()
#     plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
#     plt.title("Labels: {}".format(train_dataset.train_labels[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


# Inception model ---------------------------------------------------------------------------------------


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

# Residual model ---------------------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

# 训练集乱序，测试集有序
# Design model using class ------------------------------------------------------------------------------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
#papers with code
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        in_size = x.size(0)  # 取x张量中的第0个维度
        x = F.relu(self.conv1(x))
        x = self.rblock1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        # print(x.shape)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

# My model using class-----------------------------------------------------------------------


class MY_Net(nn.Module):
    def __init__(self):
        super(MY_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.BN = nn.BatchNorm2d(64)

        self.rblock1 = ResidualBlock(32)
        self.rblock2 = ResidualBlock(64)

        self.MP = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Sequential(nn.Dropout(0.1),
                                nn.Flatten(),
                                nn.Linear(12544,1024),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024,10),
                                )

    def forward(self, x):
        in_size = x.size(0)  # 取x张量中的第0个维度
        x = F.relu(self.conv1(x))
        torch.sum(x).backward()
        x = self.rblock1(x)
        x = F.relu(self.conv2(x))
        x = self.rblock2(x)
        x = F.relu(self.BN(x))
        x= self.MP(x)
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = MY_Net()

print(model)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device) #Convert parameters and buffers of all modules to CUDA Tensor

#summary(model, (1, 28, 28))
loss_train = []

# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.1) # 学习率调整策略，学习率下降

# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        
        
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        
        # loss.backward(retain_graph=True) # 设置为True后，再进行backward，计算图就不会消失
        loss.backward()# 反向传播backward之后计算图就会全部消失只剩下梯度，不能再次backward
        
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()  # 取item，防止生成计算图，浪费资源
        loss_train.append(loss.item())
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            # 返回两个值，第一个是每一行的最大值是多少，第二个是每一行的最大值下标是多少
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.2f %% ' %
          (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()

    plt.plot(loss_train)
    plt.ylabel('loss')
    plt.show()



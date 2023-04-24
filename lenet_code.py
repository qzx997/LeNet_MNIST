import torch
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
torch.backends.cudnn.enabled = True


# 定义网络结构，只是定义，没有运行顺序
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 构造网络有两种方式一个是seqential还有一个是module,前者在后者中也可以使用，这里使用的是sequential方式，将网络结构按顺序添加即可
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            # 第一个卷积层，输入通道为1，输出通道为6，卷积核大小为5，步长为1，填充为2保证输入输出尺寸相同
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            # 激活函数,两个网络层之间加入，引入非线性

            nn.ReLU(),  # input_size=(6*28*28)
            # 池化层，大小为2步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        # 全连接层，输入是16*5*5特征图，神经元数目120
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        # 全连接层神经元数目输入为上一层的120，输出为84
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # 最后一层全连接层神经元数目10，与上一个全连接层同理
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x，也就是把前面定义的网络结构赋予了一个运行顺序
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    # stpe1:初始化
    def __init__(self, txt, transform, target_transform=None):
        fh = open(txt, 'r')  # 打开标签文件
        imgs = []  # 创建列表，装东西
        for line in fh:  # 遍历标签文件每行
            line = line.rstrip()  # 删除字符串末尾的空格
            words = line.split()  # 通过空格分割字符串，变成列表
            imgs.append((words[0], int(words[1])))  # 把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 检索函数
        fn, label = self.imgs[index]  # 读取文件名、标签
        fn = './image/' + fn
        img = Image.open(fn).convert('L')  # 通过PIL.Image读取图片
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 64
EPOCH = 20
LR = 0.001  # 学习率
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((28, 28))
    ])
test_train = MyDataset("./train_image/train_label.txt", transform)

# --------------------------------------------------#
# 训练数据集
# --------------------------------------------------#
trainloader = torch.utils.data.DataLoader(
    test_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# --------------------------------------------------#
# 测试图片
# --------------------------------------------------#
img = Image.open("34.png")
img = img.convert('L')

plt.imshow(img)
plt.show()

img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)

net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # 梯度下降法求损失函数最小值


flag = 0
if __name__ == "__main__":
    # 遍历训练
    if flag == 1:
        for epoch in range(EPOCH):
            sum_loss = 0.0
            # 读取数据集
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # print(i)

                # 梯度清零
                optimizer.zero_grad()

                # forward + backward正向传播以及反向传播更新网络参数
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 每训练100个batch打印一次平均loss，基本上是一直减小的，一个epoch有9个因为是6w张，一次batch64个
                sum_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %d] loss: %.03f'
                          % (epoch + 1, i + 1, sum_loss / 100))
                    sum_loss = 0.0

        torch.save(net.state_dict(), '%s/epoch_%03d.pth' % ("./model", epoch + 1))


    # 单张图片测试验证
    else:
        net.load_state_dict(torch.load('./model/epoch_020.pth'))
        with torch.no_grad():
            correct = 0
            total = 0
            outputs = net(img)
            outputs = outputs.to(device)

            _, predicted = torch.max(outputs.data, 1)
            print("该图像的标签为：", predicted)



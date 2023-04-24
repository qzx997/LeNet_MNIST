import torch
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import struct
from PIL import Image
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
# 定义网络结构，只是定义，没有运行顺序
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #构造网络有两种方式一个是seqential还有一个是module,前者在后者中也可以使用，这里使用的是sequential方式，将网络结构按顺序添加即可
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            #第一个卷积层，输入通道为1，输出通道为6，卷积核大小为5，步长为1，填充为2保证输入输出尺寸相同
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            #激活函数,两个网络层之间加入，引入非线性

            nn.ReLU(),      #input_size=(6*28*28)
            #池化层，大小为2步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        #全连接层，输入是16*5*5特征图，神经元数目120
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        #全连接层神经元数目输入为上一层的120，输出为84
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        #最后一层全连接层神经元数目10，与上一个全连接层同理
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
#使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #模型保存路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  #模型加载路径
opt = parser.parse_args()




# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
 
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
 
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    # print((num_images, num_rows, num_cols))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()
 
    return images




def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
 
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
 
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

# 超参数设置
EPOCH = 8   #遍历数据集次数
BATCH_SIZE = 1      #批处理尺寸(batch_size)一次训练的样本数，相当于一次将64张图送入
LR = 0.001        #学习率



#-------------------------------------------------------#
#   训练集
#-------------------------------------------------------#
train_image_file = "./data/train-images.idx3-ubyte"
train_label_file = "./data/train-labels.idx1-ubyte"
train_images = decode_idx3_ubyte(train_image_file)
train_labels = decode_idx1_ubyte(train_label_file)

train_size = len(train_labels)
for o in range(train_size):
    for i in range(train_images[o].shape[0]):#转化为二值矩阵
        for j in range(train_images[o].shape[1]):
            if train_images[o][i,j] >= 1:
                train_images[o][i,j] = 1.0
                
train_images = torch.tensor(train_images, dtype=torch.double)

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
trainset = GetLoader(train_images, train_labels)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

#-------------------------------------------------------#
#   测试集
#-------------------------------------------------------#

test_image_file = "./data/t10k-images.idx3-ubyte"
test_label_file = "./data/t10k-labels.idx1-ubyte"
test_images = decode_idx3_ubyte(test_image_file)
test_labels = decode_idx1_ubyte(test_label_file)

test_size = len(test_labels)
for o in range(test_size):
    for i in range(test_images[o].shape[0]):#转化为二值矩阵
        for j in range(test_images[o].shape[1]):
            if test_images[o][i,j] >= 1:
                test_images[o][i,j] = 1


test_images = torch.tensor(test_images, dtype=torch.double)

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
testset = GetLoader(train_images, train_labels)

testloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )



#下载四个数据集

# # 定义训练数据集
# trainset = tv.datasets.MNIST(
#     root='./data/',
#     train=True,
#     download=True,
#     transform=transform)

# # 定义训练批处理数据
# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     )

# # 定义测试数据集
# testset = tv.datasets.MNIST(
#     root='./data/',
#     train=False,
#     download=True,
#     transform=transform)

# # 定义测试批处理数据
# testloader = torch.utils.data.DataLoader(
#     testset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     )

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9) #梯度下降法求损失函数最小值

# 训练
if __name__ == "__main__":
      #八次遍历训练
    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 读取下载的数据集
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

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
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
    #torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))



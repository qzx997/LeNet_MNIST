import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision as tv
from PIL import Image
from cnn_letnet import LeNet

BATCH_SIZE = 64
transform = transforms.ToTensor()
# 定义测试数据集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

net = LeNet()
# 加载预训练模型
net.load_state_dict(torch.load('./model/net_020.pth'))

with torch.no_grad():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(images)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if predicted.equal(labels):
            pass
        else:
            print("该图像的标签为：", predicted)
            print("实际标签为：", labels)
    print('识别准确率为：%d%%' % (100 * correct / total))

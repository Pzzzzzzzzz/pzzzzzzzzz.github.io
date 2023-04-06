---
title: "测试各种激活函数在FashionMNIST上"
date: 2023-04-02T13:46:02+08:00
draft: false
---

上一题
深度学习:第一个实验
利用MNIST数据集上进行激活函数的训练实验实验要求:
利用手写体数据的数据集，对不同的激活函数进行分析和比较 (SELU，ELU，Leaky-ReLU,ReLU等)
网络可以用最简单的前馈全连接网络(自行设)
要写出分析结果，并讨论可能的原因

**所以本次个人使用FashionMNIST来感受不同数据集之间的差距**

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义超参数
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# 加载FashionMNIST数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())

# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义全连接神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, activation):
        super(NeuralNet, self).__init__()
        
        # 定义输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(784, 512)
        
        # 根据传入的激活函数类型，定义对应的激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        
        # 定义隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

# 训练模型函数
def train_model(model, activation_func, criterion, optimizer):
    loss_history = [] # 损失函数值历史记录
    acc_history = []  # 准确率历史记录
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 在训练每个epoch之后在测试集上测试精度
        acc = test_model(model)
        
        # 记录本轮损失函数值和准确率
        loss_history.append(loss.item())
        acc_history.append(acc)
        
        # 打印日志信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%')
    
    # 返回训练好的模型和损失函数历史记录以及准确率历史记录
    return model, loss_history, acc_history

# 测试模型函数
def test_model(model):
    correct = 0
    total = 0
    for images, labels in test_loader:
        # 前向传播
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    
    # 返回准确率
    return accuracy

# 设置随机种子
torch.manual_seed(0)

# 初始化模型和优化器
activations = ['relu', 'leaky_relu', 'elu', 'selu']
results = []
for activation in activations:
    model = NeuralNet(activation=activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    trained_model, loss_history, acc_history = train_model(model, activation, criterion, optimizer)
    
    # 测试模型
    test_accuracy = test_model(trained_model)
    
    # 记录结果
    results.append({'name': activation, 'model': trained_model, 'loss_history': loss_history, 'acc_history': acc_history, 'accuracy': test_accuracy})

# 可视化结果
plt.figure(figsize=(10,5))
for res in results:
    plt.plot(res['loss_history'], label=res['name'])
plt.legend()
plt.title('Loss')
plt.show()

plt.figure(figsize=(10,5))
for res in results:
    plt.plot(res['acc_history'], label=res['name'])
plt.legend()
plt.title('Accuracy')
plt.show()

```

![](https://pzzzzzzzzz.github.io/images/20230403233355.png)
![](https://pzzzzzzzzz.github.io/images/20230403233405.png)


损失函数
从损失函数的图像可以看出，四种激活函数的表现大同小异，基本保持一致。其中SELU稍微好一点，损失函数的下降速度略微快一些。使用LeakyReLU的表现最差，导致损失函数在后面的训练过程中比其他三种激活函数更难降低。

准确率
从准确率的图像来看，四种激活函数的表现也差不多，其中SELU表现最好，而LeakyReLU表现最差。在训练开始时，ELU和SELU的表现略微好于ReLU和LeakyReLU，但训练后期四种激活函数的表现逐渐趋于一致。注意到使用LeakyReLU这种激活函数的模型表现比其他三种差一些，但得到的准确率也不会低太多。另外，需要注意到每次运行这段代码的结果可能会有所不同，部分原因是由于神经网络的训练过程存在一定的随机性，另外还可能受到硬件等因素的影响。

综上，对于这个特定的数据集，四种不同的激活函数的表现差别不大，目前就这两个数据集而言SELU和ELU在训练过程中往往会比ReLU和LeakyReLU表现略好一些。在选择激活函数时，需要结合具体情况进行选择，并且需要注意到不同的激活函数在不同的数据集上效果可能会有所不同。
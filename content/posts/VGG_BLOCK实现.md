---
title: "vgg_block块"
date: 2023-04-08T13:46:02+08:00
draft: false
---

vgg block的实现思想

```python

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True, dropout=True):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
        if dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if hasattr(self, 'bn1'): # 判断是否有 Batch Normalization
            x = self.bn1(x)
        x = F.relu(self.conv2(x))
        if hasattr(self, 'bn2'): # 判断是否有 Batch Normalization
            x = self.bn2(x)
        x = F.relu(self.conv3(x))
        if hasattr(self, 'bn3'): # 判断是否有 Batch Normalization
            x = self.bn3(x)
        x = self.pool(x)
        if hasattr(self, 'dropout'): # 判断是否使用 Dropout
            x = self.dropout(x)
        return x
```
上面的代码实现了一个 VGG Block，它包含3个卷积层和一个最大池化层，用于提取输入图像的特征信息。此外，通过添加 batch_norm 和 dropout 参数，我们可以决定是否在模型中使用 Batch Normalization 和 Dropout。

其中，kernel_size 和 padding 分别指定了卷积核大小和填充大小，in_channels 和 out_channels 分别指定了输入数据和输出数据的通道数（通道数类似于图像的深度），我们这里将 VGG Block 中每个卷积层的卷积核大小设置为 3*3。在实现中，我们使用了PyTorch 中的 nn.Conv2d 和 nn.MaxPool2d 来定义对应的卷积层和池化层。
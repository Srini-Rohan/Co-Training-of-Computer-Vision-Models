import torch.nn as  nn


class Baselayer(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Baselayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, z):
        x = z[0]
        add_labels = z[1]
        index = z[2]
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x+=add_labels[index]*identity
        x=self.relu(x)

        return [x,add_labels,index+1]

class ResNet(nn.Module):
    def __init__(self, Baselayer, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.layer1 = self.residual_connection(Baselayer, layer_list[0], planes=64)
        self.layer2 = self.residual_connection(Baselayer, layer_list[1], planes=128, stride=2)
        self.layer3 = self.residual_connection(Baselayer, layer_list[2], planes=256, stride=2)
        self.layer4 = self.residual_connection(Baselayer, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*Baselayer.expansion, num_classes)

    def forward(self, x):
        x[0] = self.relu(self.batch_norm1(self.conv1(x[0])))
        x[0] = self.max_pool(x[0])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x[0] = self.avgpool(x[0])
        x[0] = x[0].reshape(x[0].shape[0], -1)
        x[0] = self.fc(x[0])

        return x[0]

    def residual_connection(self, Baselayer, blocks, planes, stride=1):
        idownsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*Baselayer.expansion:
            idownsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*Baselayer.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*Baselayer.expansion)
            )

        layers.append(Baselayer(self.in_channels, planes, downsample=idownsample, stride=stride))
        self.in_channels = planes*Baselayer.expansion

        for i in range(blocks-1):
            layers.append(Baselayer(self.in_channels, planes))

        return nn.Sequential(*layers)



def ResNet50(num_classes, channels=3):
    return ResNet(Baselayer, [3,4,6,3], num_classes, channels)

def ResNet101(num_classes, channels=3):
    return ResNet(Baselayer, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Baselayer, [3,8,36,3], num_classes, channels)
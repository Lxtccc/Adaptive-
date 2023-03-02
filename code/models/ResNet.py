import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable

__all__ = ['ResNet20', 'ResNet56', 'ResNet110', 'ResNet164']

##########   Original_Module   ##########
class BasicBolock(nn.Module):
    def __init__(self, in_places, places, stride=1, group=1, downsampling=False):
        super(BasicBolock,self).__init__()
        self.downsampling = downsampling
        self.alphas_para = 1
        self.groups = group

        self.conv1 = self.Conv_with_paras(in_places, places, kernel_size=3, stride=stride, padding=1, group=self.groups)
        self.conv2 = self.Conv_with_paras(places, places,  kernel_size=3, stride=1, padding=1, group=self.groups)

        if self.downsampling:
            self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places,
                                      kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(places))
        self.relu = nn.ReLU(inplace=True)

    def Conv_with_paras(self, in_places, places, kernel_size, stride, padding, group=1):
        filter_paras = nn.Parameter(self.alphas_para * torch.rand(places), requires_grad=True)
        return nn.Sequential(
            nn.Conv2d_with_parameters(in_channels=in_places, out_channels=places,
                                      kernel_size=kernel_size, stride=stride, padding=padding, groups=group, bias=False, alpha=filter_paras),
            nn.BatchNorm2d(places))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False):
        super(Bottleneck,self).__init__()
        self.downsampling = downsampling
        self.alphas_para = 1
        self.expansion = 4

        self.conv1 = self.Conv_with_paras(in_places, places, kernel_size=1, stride=1, padding=0)
        self.conv2 = self.Conv_with_paras(places, places, kernel_size=3, stride=stride, padding=1)
        self.conv3 = self.Conv_with_paras(places, places*self.expansion, kernel_size=1, stride=1, padding=0)

        if self.downsampling:
            self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion,
                                      kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(places*self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def Conv_with_paras(self, in_places, places, kernel_size, stride, padding):
        filter_paras = nn.Parameter(self.alphas_para * torch.rand(places), requires_grad=True)
        return nn.Sequential(
            nn.Conv2d_with_parameters(in_channels=in_places, out_channels=places,
                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False, alpha=filter_paras),
            nn.BatchNorm2d(places))

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        out = self.conv3(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, module_type=BasicBolock, num_classes=10, G=1):
        super(ResNet,self).__init__()
        self.alphas_para = 1
        self.block = module_type
        if module_type == BasicBolock:
            self.expansion = 1
        else:
            self.expansion = 4

        self.conv1, _ = self.Conv1(in_planes = 3, places= 16)
        self.layer1 = self.make_layer(in_places=16, places=16, block=blocks[0], block_type=self.block, stride=1, group = G)
        self.layer2 = self.make_layer(in_places=16*self.expansion, places=32, block=blocks[1], block_type=self.block, stride=2, group = G)
        self.layer3 = self.make_layer(in_places=32*self.expansion, places=64, block=blocks[2], block_type=self.block, stride=2, group = G)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d_with_parameters):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def all_arch_parameters(self):
        self.arch_parameters = []
        for name, param in self.named_parameters():
            if 'alpha' in name:
                self.arch_parameters.append(param)
            else:
                pass
        return self.arch_parameters

    def Conv1(self, in_planes, places, stride=1):
        filter_paras = nn.Parameter(self.alphas_para * torch.randn(places), requires_grad=True)
        return nn.Sequential(
            nn.Conv2d_with_parameters(in_channels=in_planes, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False, alpha=filter_paras),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True)), filter_paras

    def make_layer(self, in_places, places, block, block_type, stride, group):
        layers = []
        layers.append(block_type(in_places, places, stride, group=group, downsampling =True))
        for i in range(1, block):
            layers.append(block_type(places, places, group=group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

##########   Adaptive_Module   ##########

class BasicBolock_adaptive(nn.Module):
    def __init__(self, len_list, stride=1,downsampling=False):
        super(BasicBolock_adaptive,self).__init__()
        self.downsampling = downsampling
        self.alphas_para = 1
        self.len_list = len_list

        global IND
        self.conv1 = self.Conv_with_paras(self.len_list[IND-1], self.len_list[IND], kernel_size=3, stride=stride, padding=1)
        self.conv2 = self.Conv_with_paras(self.len_list[IND], self.len_list[IND+1], kernel_size=3, stride=1, padding=1)

        if self.downsampling or (self.len_list[IND-1] != self.len_list[IND+1]):
            self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.len_list[IND-1], out_channels=self.len_list[IND+1],
                                      kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.len_list[IND+1]))
            self.downsampling = True
        self.relu = nn.ReLU(inplace=True)
        IND += 2

    def Conv_with_paras(self, in_places, places, kernel_size, stride, padding):
        filter_paras = nn.Parameter(self.alphas_para * torch.rand(places), requires_grad=True)
        return nn.Sequential(
            nn.Conv2d_with_parameters(in_channels=in_places, out_channels=places,
                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False, alpha=filter_paras),
            nn.BatchNorm2d(places))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_adaptive(nn.Module):
    def __init__(self, len_list, stride=1,downsampling=False):
        super(Bottleneck_adaptive,self).__init__()
        self.downsampling = downsampling
        self.alphas_para = 1
        self.len_list = len_list

        global IND
        self.conv1 = self.Conv_with_paras(self.len_list[IND-1], self.len_list[IND], kernel_size=1, stride=1, padding=0)
        self.conv2 = self.Conv_with_paras(self.len_list[IND], self.len_list[IND+1], kernel_size=3, stride=stride, padding=1)
        self.conv2 = self.Conv_with_paras(self.len_list[IND+1], self.len_list[IND+2], kernel_size=1, stride=1, padding=0)

        if self.downsampling or (self.len_list[IND-1] != self.len_list[IND+2]):
            self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.len_list[IND-1], out_channels=self.len_list[IND+2],
                                      kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.len_list[IND+2]))
            self.downsampling = True
        self.relu = nn.ReLU(inplace=True)
        IND += 3

    def Conv_with_paras(self, in_places, places, kernel_size, stride, padding):
        filter_paras = nn.Parameter(self.alphas_para * torch.rand(places), requires_grad=True)
        return nn.Sequential(
            nn.Conv2d_with_parameters(in_channels=in_places, out_channels=places,
                                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False, alpha=filter_paras),
            nn.BatchNorm2d(places))

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        out = self.conv3(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class ResNet_adaptive(nn.Module):
    def __init__(self, blocks, len_list, module_type=BasicBolock_adaptive, num_classes=10, expansion=1):
        super(ResNet_adaptive,self).__init__()
        self.alphas_para = 1
        self.len_list = len_list
        self.expansion = expansion
        self.block = module_type
        global IND
        IND = 0

        self.conv1, _ = self.Conv1(in_planes = 3, places= self.len_list[IND])
        IND += 1
        self.layer1 = self.make_layer(self.len_list, block=blocks[0], block_type=self.block, stride=1)
        self.layer2 = self.make_layer(self.len_list, block=blocks[1], block_type=self.block, stride=2)
        self.layer3 = self.make_layer(self.len_list, block=blocks[2], block_type=self.block, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.len_list[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d_with_parameters):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def all_arch_parameters(self):
        self.arch_parameters = []
        for name, param in self.named_parameters():
            if 'alpha' in name:
                self.arch_parameters.append(param)
            else:
                pass
        return self.arch_parameters

    def Conv1(self, in_planes, places, stride=1):
        filter_paras = nn.Parameter(self.alphas_para * torch.rand(places), requires_grad=True)
        return nn.Sequential(
            nn.Conv2d_with_parameters(in_channels=in_planes, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False, alpha=filter_paras),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True)), filter_paras

    def make_layer(self, list, block, block_type, stride):
        layers = []
        layers.append(block_type(list, stride, downsampling =True))
        for i in range(1, block):
            layers.append(block_type(list))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

##########   Different ResNet Model   ##########
#default block type --- BasicBolock for ResNet20/56/110;
#depper block type--- Bottleneck for ResNet164
def ResNet20(CLASS, len_list=None, subgraph = False, Group = 1):
    if subgraph == False:
        return ResNet([3, 3, 3], num_classes=CLASS, G = Group)
    else:
        return ResNet_adaptive([3, 3, 3], len_list=len_list, num_classes=CLASS, module_type=BasicBolock_adaptive)

def ResNet56(CLASS, len_list=None, subgraph = False):
    if subgraph == False:
        return ResNet([9, 9, 9], num_classes=CLASS)
    else:
        return ResNet_adaptive([9, 9, 9], len_list=len_list, num_classes=CLASS, module_type=BasicBolock_adaptive)

def ResNet110(CLASS, len_list=None, subgraph = False):
    if subgraph == False:
        return ResNet([18, 18, 18], num_classes=CLASS)
    else:
        return ResNet_adaptive([18, 18, 18], len_list=len_list, num_classes=CLASS, module_type=BasicBolock_adaptive)

def ResNet164(CLASS, len_list=None, subgraph=False):
    if subgraph == False:
        return ResNet([18, 18, 18], num_classes=CLASS, module_type=Bottleneck)
    else:
        return ResNet_adaptive([18, 18, 18], len_list=len_list, num_classes=CLASS, module_type=Bottleneck_adaptive)
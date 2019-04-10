#coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as functional

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import math
from sklearn.metrics import accuracy_score

def conv3(input_size, output_size, stride = 1):
    '''Возвращает свертку 3*3'''
    return nn.Conv2d(input_size, output_size, kernel_size = 3, stride = stride, 
                     padding = 1, bias = False)

class BuildBlock(nn.Module):
    '''Обычный building block
            x->conv[3,3]->relu->conv[3,3]+x -> relu->out
    '''

    def __init__(self, input_size, output_size, stride = 1):
        super(BuildBlock, self).__init__()
        self.conv1 = conv3(input_size, output_size, stride)
        self.conv2 = conv3(output_size, output_size)
        self.bn1 = nn.BatchNorm2d(input_size)
        self.bn2 = nn.BatchNorm2d(output_size)
        
        if stride != 1:
            self.expand = nn.Sequential(nn.BatchNorm2d(input_size),
                                nn.Conv2d(input_size, output_size, 1, stride))
        else:
            self.expand = nn.Sequential()
    def get_inception(input_size, out_size, stride = 1):
        return nn.Sequential(nn.BatchNorm2d(input_size),
                            nn.ReLU(),
                            conv3(input_size, out_size, stride))

    
    def forward(self, x):
        #bn->relu->conv->bn->relu->conv
        out = self.bn1(x)

        out = functional.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)

        x = self.expand(x)
        
        out += x
        return out

class SqueezeExicitationBlock(nn.Module):
    '''
    Squeeze-and-Excitation Block
    Встаривается в BottleNeck
    '''
    def __init__(self, filter_count, r = 16):
        super(SqueezeExicitationBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(filter_count, filter_count // r)
        self.fc2 = nn.Linear(filter_count // r, filter_count)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), -1, 1, 1)
        return x

class InceptionSE(nn.Module):
    def __init__(self, input_size, output_size, stride = 1):
        super(InceptionSE, self).__init__()
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()
        self.conv = conv3(input_size, output_size, stride)

        self.se_block = SqueezeExicitationBlock(filter_count = output_size)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        

        se = self.se_block(x)
        return x * se

class BuildBlockWithSE(nn.Module):
    '''building block c SE веткой
            x->conv[3,3]->relu->conv[3,3]+x -> relu->out
    '''
    def __init__(self, input_size, output_size, stride = 1):
        super(BuildBlockWithSE, self).__init__()
        self.conv1 = conv3(input_size, output_size, stride)
        self.conv2 = conv3(output_size, output_size)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()

        self.se_block = SqueezeExicitationBlock(filter_count = output_size)
        
        if stride != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(input_size, output_size, 1, stride),
                nn.BatchNorm2d(output_size),
            )
        else:
            self.expand = nn.Sequential()

    
    def forward(self, x):
        #bn->relu->conv->bn->relu->conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        
        se_result = self.se_block(out)
        out = out * se_result
        
        x = self.expand(x)
        out += x
        
        out = self.relu(out)
        return out
    
    def get_inception(input_size, output_size, stride = 1):
        return InceptionSE(input_size, output_size, stride)

class Cifar10ResNet(nn.Module):
    ''' Rsnet
    '''

    def __init__(self, BlockType, n, out_size = 10):
        '''[summary]
        
        Arguments:
            BlockType {Класс блока} -- building or bottleneck
            n {int} -- Количество блоков
        
        Keyword Arguments:
            out_size {int} -- Количество классов (default: {10})
        '''
        
        super(Cifar10ResNet, self).__init__()

        self.out_size = out_size
        self.inception = BlockType.get_inception(3, 16)
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.block16 = self.get_basic_blocks(BlockType, 16, 16, n, stride = 1)
        self.block32 = self.get_basic_blocks(BlockType, 16, 32, n, stride = 2)
        self.block64 = self.get_basic_blocks(BlockType, 32, 64, n, stride = 2)
        self.fc = nn.Linear(64, out_size)
        conv_count = 0
        linear_count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_count +=1
            elif isinstance(m, nn.BatchNorm2d):
                pass
            elif isinstance(m, nn.Linear):
                linear_count +=1
        print("conv   =", conv_count)
        print("linear =", linear_count)

    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.max_pool(x)
        x = self.inception(x)
        
        x = self.block16(x)
        x = self.block32(x)
        x = self.block64(x)
        x = self.avg(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        

    def get_basic_blocks(self, BlockType, input_size, size, blocks_count, stride):
        layers = []
        layers.append(BlockType(input_size, size, stride))

        for _ in range(blocks_count-1):
            layers.append(BlockType(size, size, stride = 1))
            
        return nn.Sequential(*layers)

def design_resnet110():
    return Cifar10ResNet(BuildBlock, 1, out_size = 10)
def design_se110():
    return Cifar10ResNet(BuildBlockWithSE, 18, out_size=10)
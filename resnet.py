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
        
        self.stride = stride
        
        self.x_conv = nn.Conv2d(input_size, output_size, 1, stride)
        self.x_bn = nn.BatchNorm2d(input_size)

    
    def forward(self, x):
        #bn->relu->conv->bn->relu->conv
        out = self.bn1(x)

        out = functional.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)

        if self.stride != 1:
            x = self.x_bn(x)
            x = self.x_conv(x)
        
        out += x
        return out

class BottleNeckBlock(nn.Module):
    '''BottleNeck блок для оооочень глубоких сетей
    
    Arguments:
        nn {[type]} -- [description]
    '''


    def __init__(self, input_size, size, stride, need_expand = False):
        super(BottleNeckBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_size)
        self.conv1 = nn.Conv2d(input_size, size, kernel_size = 1,
                                bias = False)
        
        self.conv2 = nn.Conv2d(size, size, kernel_size = 3, stride = stride,
                                bias = False, padding = 1)
        self.bn2 = nn.BatchNorm2d(size)

        self.conv3 = nn.Conv2d(size, 4 * size, kernel_size = 1,
                                bias = False)
        self.bn3 = nn.BatchNorm2d(size)

        self.x_conv = nn.Conv2d(input_size, 4*size, 1, stride = stride)
        self.x_bn = nn.BatchNorm2d(input_size)

        self.stride = stride
        self.need_expand = need_expand

    def forward(self, x):
        # first
        out = self.bn1(x)
        out = functional.relu(out)
        out = self.conv1(out)
        # second
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)
        # third 
        out = self.bn3(out)
        out = functional.relu(out)
        out = self.conv3(out)

        if self.need_expand:
            x = self.x_bn(x)
            x = self.x_conv(x)
        
        out += x

        return out

class SqueezeExicitationBlock(nn.Module):
    '''
    Squeeze-and-Excitation Block
    Встаривается в BottleNeck
    '''
    def __init__(self, size, filter_count, r = 16):
        super(SqueezeExicitationBlock, self).__init__()
        self.global_pool = nn.AvgPool2d(kernel_size = size)
        self.fc1 = nn.Linear(filter_count, filter_count // r)
        self.fc2 = nn.Linear(filter_count // 16, filter_count)
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

class InceptionSEBlock(nn.Module):

    def __init__(self):
        super(InceptionSEBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64,
                        kernel_size = 7, stride = 2, padding = 3)

        self.relu = nn.ReLU()
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.se_block = SqueezeExicitationBlock(size = 8, filter_count = 64)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        se_result = self.se_block(x)
        x = x * se_result
        return x



class BottleNeckBlockWithSE(nn.Module):
    '''
    ResNet block с SE-веткой
    '''
    def __init__(self, image_size,  input_size, size, stride, need_expand = False):
        super(BottleNeckBlockWithSE, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_size)
        self.conv1 = nn.Conv2d(input_size, size, kernel_size = 1,
                                bias = False)
        
        self.conv2 = nn.Conv2d(size, size, kernel_size = 3, stride = stride,
                                bias = False, padding = 1)
        self.bn2 = nn.BatchNorm2d(size)

        self.conv3 = nn.Conv2d(size, 4 * size, kernel_size = 1,
                                bias = False)
        self.bn3 = nn.BatchNorm2d(size)

        self.se_block = nn.Sequential(SqueezeExicitationBlock(size = image_size, 
                                                filter_count = 4*size))
        self.x_conv = nn.Conv2d(input_size, 4*size, 1, stride = stride)
        self.x_bn = nn.BatchNorm2d(input_size)

        self.stride = stride
        self.need_expand = need_expand

    def forward(self, x):
        # first
        out = self.bn1(x)
        out = functional.relu(out)
        out = self.conv1(out)
        # second
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)
        # third 
        out = self.bn3(out)
        out = functional.relu(out)
        out = self.conv3(out)

        se_result = self.se_block(out)
        out = out * se_result

        if self.need_expand:
            x = self.x_bn(x)
            x = self.x_conv(x)
        
        out += x

        return out


class ResNet(nn.Module):
    ''' Rsnet
    '''

    def __init__(self, block_bype, block_counts, out_size = 10):
        '''[summary]
        
        Arguments:
            BlockType {Класс блока} -- building or bottleneck
            block_counts {list} -- Количество блоков
        
        Keyword Arguments:
            out_size {int} -- Количество классов (default: {10})
        '''
        
        super(ResNet, self).__init__()

        self.out_size = out_size
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64,
                                kernel_size = 7, stride = 2, padding = 3)
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception = nn.Sequential(self.conv1, self.max_pool)

        if block_bype == "building":
            self.block64 = self.get_basic_blocks(64, 64, block_counts[0], 
                                                stride = 1)
            self.block128 = self.get_basic_blocks(64, 128, block_counts[1],
                                                 stride = 2)
            self.block256 = self.get_basic_blocks(128, 256, block_counts[2],
                                                 stride = 2)
            self.block512 = self.get_basic_blocks(256, 512, block_counts[3], 
                                                    stride = 2)
            self.fc = nn.Linear(512, out_size)
        
        elif block_bype == "bottleneck":
            self.block64 = self.get_bottleneck_blocks(64, 64, block_counts[0],
                                                    stride = 1)
            self.block128 = self.get_bottleneck_blocks(64*4, 128,
                                                     block_counts[1],stride = 2)
            self.block256 = self.get_bottleneck_blocks(128*4, 256, 
                                                    block_counts[2], stride = 2)
            self.block512 = self.get_bottleneck_blocks(256*4, 512, 
                                                    block_counts[3], stride = 2)
            self.fc = nn.Linear(512*4, out_size)
        
        elif block_bype == "se":
            self.inception = nn.Sequential(InceptionSEBlock())

            self.block64 = self.get_bottleneck_se_blocks(8, 64, 64, block_counts[0],
                                                    stride = 1)
            self.block128 = self.get_bottleneck_se_blocks(4, 64*4, 128,
                                                     block_counts[1],stride = 2)
            self.block256 = self.get_bottleneck_se_blocks(2, 128*4, 256, 
                                                    block_counts[2], stride = 2)
            self.block512 = self.get_bottleneck_se_blocks(1, 256*4, 512, 
                                                    block_counts[3], stride = 2)
            self.fc = nn.Linear(512*4, out_size)

            

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.max_pool(x)
        x = self.inception(x)
        
        x = self.block64(x)
        x = self.block128(x)
        x = self.block256(x)
        x = self.block512(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        

        
    def get_basic_blocks(self, input_size, size, blocks_count, stride):
        layers = []
        layers.append(BuildBlock(input_size, size, stride))

        for _ in range(blocks_count-1):
            layers.append(BuildBlock(size, size, stride = 1))
            
        return nn.Sequential(*layers)
    
    def get_bottleneck_blocks(self, input_size, size, bloks_count, stride):
        layers = []
        layers.append(BottleNeckBlock(input_size, size, stride, 
                                        need_expand = True))

        for _ in range(bloks_count-1):
            layers.append(
                BottleNeckBlock(size*4, size, stride = 1, need_expand = False))
        
        return nn.Sequential(*layers)

    def get_bottleneck_se_blocks(self, image_size, input_size, size, blocks_count, stride):
        layers = []
        layers.append(BottleNeckBlockWithSE(image_size, input_size, size, stride,
                                            need_expand = True))

        for _ in range(blocks_count-1):
            layers.append(
                BottleNeckBlockWithSE(image_size, size*4, size, 
                                      stride = 1, need_expand = False)
            )
        return nn.Sequential(*layers)

    
def get_accuracy(clothesNet, x, y):
    '''
    Вычисление точности на выборке
    '''
    output = clothesNet(x)
    _, predicted = torch.max(output, 1)
    
    return accuracy_score(y, predicted)
  
def get_train_params(resnet32,criterion, dataloader, batch_size):
    run_loss = 0
    true_count = 0
    i = 0
    n = 0
    with torch.no_grad():
        for (i, data) in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = resnet32(inputs)
            outputs = outputs.cuda()
          
          
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cuda()
            true_count += int(sum(predicted==labels))
            loss = criterion(outputs, labels)
            run_loss += loss.item()
            i += 1
            n += predicted.size()[0]
    run_loss /= i
    true_count /= n
  
    return run_loss, true_count



def design_resnet34():
    return ResNet("building", [3,4,6,3], out_size = 10)
def design_resnet50():
    return ResNet("bottleneck", [3, 4, 6, 3], out_size = 10)
def design_resnet110():
    return ResNet("bottleneck", [3, 4, 26, 3], out_size = 10)
def design_se110():
    return ResNet("se", [3,4,3,3], out_size=10)

def train_and_test():

    #import data set
    batch_size = 2


    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5),
                            std = (1.0, 1.0, 1.0))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                            std = (1.0, 1.0, 1.0))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

   #torch.cuda.device(1)
    #resnet32 = design_resnet34()
    #resnet32 = design_resnet50()
    resnet32 = design_se110()
          
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # resnet = resnet32.cuda()
    #resnet32.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(resnet32.parameters(), lr = 0.001, momentum = 0.9, 
                             weight_decay = 0.005, nesterov = True)

    
    print(optimizer.state_dict()['param_groups'])
    epoch_count = 200
    print('Start training')
    print (len(trainloader))
    for epoch in range(epoch_count):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            #print(inputs.size())
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            
            optimizer.zero_grad()

            outputs = resnet32(inputs)
            # outputs.cuda()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print(i)
        
        train_loss, train_accuracy = get_train_params(resnet32, criterion,
                                                     trainloader, batch_size)
        test_loss, test_accuracy = get_train_params(resnet32, criterion,
                                                     testloader, batch_size)
        
        print('Epoch #{}'.format(epoch+1))
        print('\tTrain loss    : {0:.10f}'.format(train_loss))
        print('\tTest loss     : {0:.10f}'.format(test_loss))
        print('\tTrain accuracy: {0:.10f}'.format(train_accuracy))
        print('\tTest accuracy : {0:.10f}'.format(test_accuracy))
        print('#', '~'*40, '#')
        # tbc.save_value("loss", "train", epoch+1, train_loss)
        # tbc.save_value("loss", "test", epoch+1, test_loss)
        # tbc.save_value("accuracy", "train", epoch+1, train_accuracy)
        # tbc.save_value("accuracy", "test", epoch+1, test_accuracy)
            
    print('Finish training')

def test_se_block():
    a = torch.randn(5, 16, 2, 2)
    seblock = SqueezeExicitationBlock(2, 16)
    x = seblock(a)
    print('x=',x)
    print('a=',a)
    print('a*x=',a*x)
if __name__ == "__main__":
    train_and_test()
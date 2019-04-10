import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from bisect import bisect_right

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import math
from sklearn.metrics import accuracy_score
import random

from PIL import Image

import numpy as np

import resnet


def get_train_params(resnet32,criterion, dataloader, batch_size):
    run_loss = 0
    true_count = 0
    i = 0
    n = 0
    with torch.no_grad():
        for (i, data) in enumerate(dataloader):
            inputs, labels = data
            outputs = resnet32(inputs)
        
            _, predicted = torch.max(outputs, 1)
            true_count += int(sum(predicted==labels))
            loss = criterion(outputs, labels)
            run_loss += loss.item()
            i += 1
            n += predicted.size()[0]
    run_loss /= i
    true_count /= n
  
    return run_loss, true_count

def get_train_valid_idxs(n : int, train_part : float):
    idxs = list(range(n))
    random.shuffle(idxs)
    train_count = int(train_part * n)
    return idxs[:train_count], idxs[train_count:]

class GaussianNoise:
    def __init__(self, prob, sigma):
        self.prob = prob
        self.sigma = sigma
    def __call__(self, img):
        #print("hey")
        if random.random() < self.prob:
            return self.get_noised(img)
        else:
            return img

    def get_noised(self, img):
        #print("Hi man")
        w, h = img.size
        c = len(img.getbands())

        noise = np.random.normal(0, self.sigma, (h,w,c))
        new_img = np.array(img) + noise
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        new_img = np.uint8(new_img)
        return Image.fromarray(new_img)
 
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        nn.init.normal_(m.weight.data, 0, math.sqrt(2./n))
        nn.init.uniform_(m.bias.data, -math.sqrt(2./n), math.sqrt(2./n))

def get_lr(epoch, milstones, l_rates):
    assert(len(l_rates) == len(milstones) + 1, "Length of rates must be mil +1")
    i = bisect_right(milstones, epoch)
    return l_rates[i]

def get_schedulder(optimizer, milstones, l_rates):
    func = lambda epoch: get_lr(epoch, milstones, l_rates)
    return optim.lr_scheduler.LambdaLR(optimizer, func)

               
def train_and_test():

    #import data set
    
    #
    # ВСЕ КОНСТАНТЫ ПИСАТЬ СЮДА, ЧТОБЫ ПОТОМ НЕ ИСКАТЬ ПО ВСЕМУ ПОЛОТНУ!!!
    #
    
    batch_size = 128
    learning_rate = 1
    #learning_rate_decay = 0.975
    momentum = 0.9
    weight_decay = 0.0001
    epoch_count = 200
    nesterov = False
    
    thresholds = [1, 102, 136]
    learning_rates = [0.01, 0.1, 0.01, 0.001]
    train_part = 0.9
    
    model_path = 'drive/course_work/models/10-04-19-model_new'
    old_path = 'drive/course_work/models/15-03-19-model_new'
    
    # generate train dataset
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        GaussianNoise(0.9, 8),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5),
                            std = (1.0, 1.0, 1.0))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                            std = (1.0, 1.0, 1.0))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    # обучающая выборка без data aug
    trainset_param = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=test_transform)

    train_idxs, valid_idxs = get_train_valid_idxs(len(trainset), train_part)

    train_subset = Subset(trainset, train_idxs)
    train_subset_param = Subset(trainset_param, train_idxs)
    valid_subset = Subset(trainset_param, valid_idxs)
    print("Dataset sizes:", len(train_subset), len(valid_subset))

    train_subset_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, num_workers=2)
    train_subset_param_loader = DataLoader(train_subset_param, batch_size=batch_size,
                                    shuffle=False, num_workers=2)
    valid_subset_loader = DataLoader(valid_subset, batch_size=batch_size,
                                    shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    resnet32 = design_se110()
    resnet32.apply(weight_init)
    resnet32 = resnet32.to(device)
    optimizer = optim.SGD(resnet32.parameters(), lr = learning_rate, 
                          momentum = momentum, weight_decay = weight_decay,
                          nesterov = nesterov)


    
    #resnet32 = resnet32.cuda()
    
    #ЗАГРУЗКА ИЗ ДИСКА!!!
    
#     checkpoint = torch.load(old_path)
#     resnet32.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_epoch = checkpoint['epoch'] + 1
    
    resnet32.train()
    
    scheduler = get_schedulder(optimizer, thresholds, learning_rates)
    #scheduler = optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[82, 123], gamma = gamma)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    #optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        
    losses = {
        'train': [],
        'valid': []
    }
    accuracies = {
        'train': [],
        'valid': []
    }
    
    train_loss, train_accuracy = get_train_params(resnet32, criterion,
                                        train_subset_param_loader, batch_size)
    valid_loss, valid_accuracy = get_train_params(resnet32, criterion,
                                                valid_subset_loader, batch_size)
    print('Epoch #{}'.format(0))
    print('\tTrain loss    : {0:.10f}'.format(train_loss))
    print('\tValid loss    : {0:.10f}'.format(valid_loss))
    print('\tTrain accuracy: {0:.10f}'.format(train_accuracy))
    print('\tValid accuracy: {0:.10f}'.format(valid_accuracy))
    print('#', '~'*40, '#')
    print('Start training')
    #return
    #tbc.save_value("loss", "train", start_epoch, train_loss)
    #tbc.save_value("loss", "valid",start_epoch, valid_loss)
    #tbc.save_value("accuracy", "train", start_epoch, train_accuracy)
    #tbc.save_value("accuracy", "valid", start_epoch, valid_accuracy)
    
    losses['train'].append(train_loss)
    losses['valid'].append(valid_loss)
    accuracies['train'].append(train_accuracy)
    accuracies['valid'].append(valid_accuracy)
    
    

    for epoch in range(start_epoch, epoch_count):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        for i, data in enumerate(train_subset_loader, 0):
            inputs, labels = data
            
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            

            optimizer.zero_grad()
            #print(inputs.size())
            outputs = resnet32(inputs)
            
            loss = criterion(outputs, labels)
            #quit()
            loss.backward()
            optimizer.step()
        
        train_loss, train_accuracy = get_train_params(resnet32, criterion,
                                        train_subset_param_loader, batch_size)
        valid_loss, valid_accuracy = get_train_params(resnet32, criterion,
                                                valid_subset_loader, batch_size)
        

        print('Epoch #{}'.format(epoch+1))
        print('\tTrain loss    : {0:.10f}'.format(train_loss))
        print('\tValid loss    : {0:.10f}'.format(valid_loss))
        print('\tTrain accuracy: {0:.10f}'.format(train_accuracy))
        print('\tValid accuracy: {0:.10f}'.format(valid_accuracy))
        print('#', '~'*40, '#')
        # to Tensorboard
        #tbc.save_value("loss", "train", epoch+1, train_loss)
        #tbc.save_value("loss", "valid", epoch+1, valid_loss)
        #tbc.save_value("accuracy", "train", epoch+1, train_accuracy)
        #tbc.save_value("accuracy", "valid", epoch+1, valid_accuracy)
        # to save
        losses['train'].append(train_loss)
        losses['valid'].append(valid_loss)
        accuracies['train'].append(train_accuracy)
        accuracies['valid'].append(valid_accuracy)
        torch.save({
            'epoch': epoch,
            'model': resnet32.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_idxs': train_idxs,
            'valid_idxs':  valid_idxs,
            'loss': losses,
            'accuracy': accuracies
        }, model_path)

            
    print('Finish training')

train_and_test()


if __name__ == "__main__":
    test_save()
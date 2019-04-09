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

        if need_expand :
            self.expand = nn.Sequential(nn.BatchNorm2d(input_size),
                        nn.Conv2d(input_size, 4*size, 1, stride = stride))
        else:
            self.expand = nn.Sequential()


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

class InceptionSEBlock(nn.Module):

    def __init__(self):
        super(InceptionSEBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64,
                        kernel_size = 7, stride = 2, padding = 3)

        self.relu = nn.ReLU()
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.se_block = SqueezeExicitationBlock(filter_count = 64)
    
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
    def __init__(self, input_size, size, stride, need_expand = False):
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

        self.se_block = SqueezeExicitationBlock(filter_count = 4*size)
        
        if need_expand:
            self.expand = nn.Sequential(nn.BatchNorm2d(input_size),
                    nn.Conv2d(input_size, 4*size, 1, stride = stride))                      
        else:
            self.expand = nn.Sequential()

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
       
        x = self.expand(x)
        
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

            self.block64 = self.get_bottleneck_se_blocks(64, 64, block_counts[0],
                                                    stride = 1)
            self.block128 = self.get_bottleneck_se_blocks(64*4, 128,
                                                     block_counts[1],stride = 2)
            self.block256 = self.get_bottleneck_se_blocks(128*4, 256, 
                                                    block_counts[2], stride = 2)
            self.block512 = self.get_bottleneck_se_blocks(256*4, 512, 
                                                    block_counts[3], stride = 2)
            self.fc = nn.Linear(512*4, out_size)

            

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

    def get_bottleneck_se_blocks(self, input_size, size, blocks_count, stride):
        layers = []
        layers.append(BottleNeckBlockWithSE(input_size, size, stride,
                                            need_expand = True))

        for _ in range(blocks_count-1):
            layers.append(
                BottleNeckBlockWithSE(size*4, size, 
                                      stride = 1, need_expand = False)
            )
        return nn.Sequential(*layers)

def design_resnet34():
    return ResNet("building", [3,4,6,3], out_size = 10)
def design_resnet50():
    return ResNet("bottleneck", [3, 4, 6, 3], out_size = 10)
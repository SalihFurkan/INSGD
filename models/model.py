import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SimpleModel(nn.Module):
    # Simple Model for Normalized LMS Training on MNIST
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding=1,bias=False)
        #self.bath1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3,padding=1, stride=2, bias=False)
        #self.bath2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3,padding=1, stride=2, bias=False)
        #self.bath3 = nn.BatchNorm2d(32)
        #self.drop1 = nn.Dropout(0.2)
        
        self.fc1   = nn.Linear(32*7*7, 10)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        #x = self.bath1(x)
        x = self.conv2(x)
        #x = self.bath2(x)
        x = F.relu(x)
        x = self.conv3(x) 
        #x = self.bath3(x)
        #x = self.drop1(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        return x 
    
    def inputs_ext(self, x):
        
        out1 = self.conv1(x)
        #out2 = self.bath1(out1)
        out3 = self.conv2(out1)
        #out4 = self.bath2(out3)
        out5 = F.relu(out3)
        out6 = self.conv3(out5) 
        #out7 = self.bath3(out6)
        #out8 = self.drop1(out7)
        out9 = F.relu(out6)
        out10 = torch.flatten(out9, 1)
        
        return [x, out1, out5, out10]

class SimpleModel_Cifar(nn.Module):
    # Simple Model for Normalized LMS Training
    def __init__(self):
        super(SimpleModel_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3,8,3,padding=1,bias=False)
        self.bath1 = nn.BatchNorm2d(8)
        self.conv21 = nn.Conv2d(8,16,1,padding=0, stride=2, bias=False)
        self.bath21 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(8,16,3,padding=1, stride=2, bias=False)
        self.bath2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3,padding=1, stride=2, bias=False)
        self.bath3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,64,3,padding=1, stride=2, bias=False)
        self.bath4 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.2)
        
        self.fc1   = nn.Linear(64*4*4, 10)
        
        
    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.bath1(x)
        x1 = self.conv21(x)
        x1 = self.bath21(x1)
        x = self.conv2(x)
        x = self.bath2(x)
        x = F.relu(x)
        x = x + x1
        x = self.conv3(x) 
        x = self.bath3(x)
        x = F.relu(x)
        x = self.conv4(x) 
        x = self.bath4(x)
        x = self.drop1(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        return x 
    
    def inputs_ext(self, x):
        out1 = self.conv1(x)
        out2 = self.bath1(out1)
        out3 = self.conv2(out2)
        out4 = self.bath2(out3)
        out5 = F.relu(out4)
        out6 = self.conv3(out5) 
        out7 = self.bath3(out6)
        out8 = F.relu(out7)
        out9 = self.conv4(out8) 
        out10 = self.bath4(out9)
        out11 = self.drop1(out10)
        out12 = F.relu(out11)
        out13 = torch.flatten(out12, 1)
        
        out14 = self.fc1(out13)
        
        return [x, out2, out5, out8, out13, out14]
    
def get_directional_filters_initializer_np():
  dir_filter_minus63 = torch.unsqueeze(torch.tensor([[0,0.0313, 0.0313,0,0,0,0],[0,0,0,0,0,0,0],[0,0,-0.2813,-0.2813,0,0,0],[0,0,0,1,0,0,0],[0,0,0,-0.2813,-0.2813,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0.0313, 0.0313,0]]),0)
  dir_filter_minus45 = torch.unsqueeze(torch.tensor([[0.0625,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,-0.5625,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,-0.5625,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0.0625]]),0)
  dir_filter_minus26 = torch.transpose(dir_filter_minus63, 1, 2)
  dir_filter0 = torch.unsqueeze(torch.tensor([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0.0625,0,-0.5625,1,-0.5625,0,0.0625],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]),0)
  dir_filter26 = torch.fliplr(dir_filter_minus26)
  dir_filter45 = torch.fliplr(dir_filter_minus45)
  dir_filter63 = torch.fliplr(dir_filter_minus63)
  dir_filter90 = torch.transpose(dir_filter0, 1, 2)
  dir_filters = torch.cat((dir_filter_minus63,dir_filter_minus45,dir_filter_minus26,dir_filter0,dir_filter26,dir_filter45,dir_filter63,dir_filter90), dim=0)
  dir_filters = torch.unsqueeze(dir_filters,1)
  return torch.nn.Parameter(dir_filters)

def get_matched_filters_initializer_np():
  matched_filter_minus63 = torch.unsqueeze(torch.tensor([[0,-0.25,0,0,0,0,0],[0,-0.5,-0.5,0,0,0,0],[0,0,1,1,0,0,0],[0,0,0,2,0,0,0],[0,0,0,1,1,0,0],[0,0,0,0,-0.5, -0.5,0],[0,0,0,0,0,-0.25,0]]),0)
  matched_filter_minus45 = torch.unsqueeze(torch.tensor([[-0.25,0,0,0,0,0,0],[0,-0.5,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,2,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,-0.5,0],[0,0,0,0,0,0,-0.25]]),0)
  matched_filter_minus26 = torch.transpose(matched_filter_minus63, 1, 2)
  matched_filter0 = torch.unsqueeze(torch.tensor([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[-0.25,-0.5,1,2,1,-0.5,-0.25],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]),0)
  matched_filter26 = torch.fliplr(matched_filter_minus26)
  matched_filter45 = torch.fliplr(matched_filter_minus45)
  matched_filter63 = torch.fliplr(matched_filter_minus63)
  matched_filter90 = torch.transpose(matched_filter0, 1, 2)
  matched_filters = torch.cat((matched_filter_minus63,matched_filter_minus45,matched_filter_minus26,matched_filter0,matched_filter26,matched_filter45,matched_filter63,matched_filter90), dim=0)
  matched_filters = torch.unsqueeze(matched_filters,1)
  return torch.nn.Parameter(matched_filters)


class Directional_Block(nn.Module):
    def __init__(self):
        super(Directional_Block, self).__init__()
        self.conv1 = nn.Conv2d(1,8,7,padding=3,bias=False)
        self.conv1.weight = get_directional_filters_initializer_np()
        self.conv2 = nn.Conv2d(8,8,7,padding=3,bias=False,groups=8)
        self.conv2.weight = get_matched_filters_initializer_np()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) 
        x = F.leaky_relu(x)
        return x 

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, dropp=0.1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropp)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        return out

class MyModel(nn.Module):
    def __init__(self, ResidualBlock, num_classes=6):
        super(MyModel, self).__init__()
        
        self.inchannel = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8)
        )
        self.direcb = Directional_Block()
        self.maxp1  = nn.MaxPool2d(2,2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 32, 1, stride=1)
        self.maxp2  = nn.MaxPool2d(2,2, padding=(1,0))
        self.layer2 = self.make_layer(ResidualBlock, 64, 1, stride=1)
        self.maxp3  = nn.MaxPool2d(2,2, padding=(0,1))
        self.layer3 = self.make_layer(ResidualBlock, 128, 1, stride=1, dropp=0.2)
        self.maxp4  = nn.MaxPool2d(2,2)
        self.fc = nn.Sequential(nn.Linear(128*10*5, 64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5))
        self.fc2 = nn.Linear(64, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride, dropp=0.1):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, dropp))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        d1 = self.direcb(x)
        out = c1 + d1
        #print(out.size())
        out = self.maxp1(out)
        #print(out.size())
        out = self.layer1(out)
        #print(out.size())
        out = self.maxp2(out)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = self.maxp3(out)
        #print(out.size())
        out = self.layer3(out)
        #out = self.maxp4(out)
        #print(out.size())
        out = torch.flatten(out, 1)
        #print(out.size())
        out = self.fc(out)
        out = self.fc2(out)
        out = F.softmax(out)
        return out
    

def MyModel_Residual():

    return MyModel(ResidualBlock)
    
class MyModelBase(nn.Module):
    def __init__(self, ResidualBlock, num_classes=6):
        super(MyModelBase, self).__init__()
        
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    
class MyModelx3(nn.Module):
    def __init__(self, num_classes=6):
        super(MyModelx3, self).__init__()
        self.base1 = MyModelBase(ResidualBlock)
        self.base2 = MyModelBase(ResidualBlock)
        self.base3 = MyModelBase(ResidualBlock)
        self.fc = nn.Linear(64*3, num_classes*4)
        self.fc2 = nn.Linear(num_classes*4+1, num_classes)
        
    def forward(self, x):
        dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        age = x[1]
        x   = x[0]
        x1 = torch.unsqueeze(x[:,0,:,:],1)
        x2 = torch.unsqueeze(x[:,1,:,:],1)
        x3 = torch.unsqueeze(x[:,2,:,:],1)
        x1 = self.rot_img(x1, 0, dtype)
        x2 = self.rot_img(x2, np.pi/36, dtype)
        x3 = self.rot_img(x3, np.pi/18, dtype)
        y = torch.cat([self.base1(x1), self.base2(x2), self.base3(x3)], dim=-1)
        out = self.fc(y)
        out = self.fc2(torch.cat([out, age], dim=-1))
        return out
    
    def get_rot_mat(self, theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])


    def rot_img(self, x, theta, dtype):
        rot_mat = self.get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid)
        return x
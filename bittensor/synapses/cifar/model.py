import bittensor
from bittensor.synapses.cifar.model import CIFARSynapse
from bittensor.utils.model_utils import ModelToolbox

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class CIFARSynapse(bittensor.Synapse):
    def __init__(self):
        super(CIFAR, self).__init__()
        model_config = self.DPN26()
        in_planes, out_planes = model_config['in_planes'], model_config['out_planes']
        num_blocks, dense_depth = model_config['num_blocks'], model_config['dense_depth']

        # Image encoder
        self._transform = bittensor.utils.batch_transforms.Normalize((0.1307,), (0.3081,))
        self._adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        # Main Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=1)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear((out_planes[3] * 4)+(((num_blocks[3]+1) * 4)*dense_depth[3]), 10)

        # Distill Network
        self.dist_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dist_bn1 = nn.BatchNorm2d(64)
        self.dist_layer1 = copy.deepcopy(self.layer1)
        self.dist_layer2 = copy.deepcopy(self.layer2)
        self.dist_layer3 = copy.deepcopy(self.layer3)
        self.dist_layer4 = copy.deepcopy(self.layer4)

        # Logit Network
        self.logit_layer1 = nn.Linear((out_planes[3] * 4)+(((num_blocks[3]+1) * 4)*dense_depth[3]), 512)
        self.logit_layer2 = nn.Linear(512, 256)
        self.logit_layer3 = nn.Linear(256, 10)


    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(self.Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def DPN92(self):
        cfg = {
            'in_planes': (96,192,384,768),
            'out_planes': (256,512,1024,2048),
            'num_blocks': (3,4,20,3),
            'dense_depth': (16,32,24,128)
        }
        return cfg

    def DPN26(self):
        cfg = {
            'in_planes': (96,192,384,768),
            'out_planes': (256,512,1024,2048),
            'num_blocks': (2,2,2,2),
            'dense_depth': (16,32,24,128)
        }
        return cfg

    def distill(self, x):
        x = x.to(self.device)
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.dist_bn1(self.dist_conv1(x)))
        x = self.dist_layer1(x)
        x = self.dist_layer2(x)
        x = self.dist_layer3(x)
        x = self.dist_layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def logits (self, x):
        x = F.relu(self.logit_layer1 (x))
        x = F.relu(self.logit_layer2 (x))
        x = F.log_softmax(x)
        return x

    def forward(self, x, y = None):        
        y = self.distill(x) if y == None else y
        x = x.to(self.device)
        y = y.to(self.device)
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def encode_image(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._transform(inputs)
        inputs = self._adaptive_pool(inputs)
        return torch.flatten(inputs, start_dim=1)

    @property
    def input_shape(self):
        return [-1, 3072]

    @property
    def output_shape(self):
        return [-1, 10]

    class Bottleneck(nn.Module):
        def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
            super(CIFAR.Bottleneck, self).__init__()
            self.out_planes = out_planes
            self.dense_depth = dense_depth

            self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
            self.bn2 = nn.BatchNorm2d(in_planes)
            self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)

            self.shortcut = nn.Sequential()
            if first_layer:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes + dense_depth)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            x = self.shortcut(x)
            d = self.out_planes
            out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
            out = F.relu(out)
            return out

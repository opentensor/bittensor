import torch
import torchvision
from typing import List, Tuple, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F

import bittensor

class MnistSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on 28, 28 pixel images to detect handwritten characters.
    """
    def __init__(self, config: bittensor.Config):
        super(MnistSynapse, self).__init__(config)
        # Image encoder
        self._adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))
        
        # Forward Network
        self.forward_layer1 = nn.Linear(784, 1024)
        self.forward_layer2 = nn.Linear(1024, 1024)
        
        # Distillation Network
        self.dist_layer1 = nn.Linear(784, 1024)
        self.dist_layer2 = nn.Linear(1024, 1024)
        
        # Logit Network 
        self.logit_conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1)
        self.logit_average1 = nn.AvgPool2d(2, stride=2)
        self.logit_conv2 = nn.Conv2d(4, 16, kernel_size=5, stride=1)
        self.logit_average2 = nn.AvgPool2d(2, stride=2)
        self.logit_conv3 = nn.Conv2d(16, 256, kernel_size=4, stride=1)
        self.logit_layer1 = nn.Linear(1024, 1024)
    
    def distill(self, x):
        x = F.relu(self.dist_layer1 (x))
        x = F.relu(self.dist_layer2 (x))
        return x
    
    def logits(self, x):
        x = x.view(-1, 1, 32, 32)
        x = torch.tanh(self.logit_conv1(x))
        x = self.logit_average1(x)
        x = torch.tanh(self.logit_conv2(x))
        x = self.logit_average2(x)
        x = torch.tanh(self.logit_conv3(x))
        x = torch.flatten(x, start_dim = 1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.logit_layer1(x))
        x = F.log_softmax(x, dim=1)
        return x
        
    def forward (self, x, y = None):
        y = self.distill(x) if y == None else y
        x = F.relu(self.forward_layer1 (x))
        x = F.relu(self.forward_layer2 (x))
        x = x # + y
        return x  
      
    def encode_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs 
 
    def encode_string(self, inputs: List [str] ) -> torch.Tensor:
        # Output to [batch_size, 1024]
        return torch.zeros( (len(inputs), 1024) )
    
    def encode_image(self, inputs: List [object]) -> torch.Tensor:
        transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
        image_batch = []
        for image in inputs:
            image_batch.append(transform(image))
        image_batch = torch.cat(image_batch, dim=0)
        
        # Encode images to consistent size.
        image_batch = self._adaptive_pool(image_batch)
    
        # Output to [batch_size, 1024]
        return torch.flatten(image_batch, start_dim = 1)
  
    
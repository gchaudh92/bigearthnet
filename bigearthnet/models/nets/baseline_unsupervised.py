import torch
from byol_pytorch import BYOL
from torchvision import models


class Byol(torch.nn.Module):
    def __init__(self, image_size)
        self.resnet =  models.resnet50(pretrained=True) 
        self.learner = BYOL(self.resnet,image_size = 256,hidden_layer = 'avgpool')

    def forward(self, x):
        return self.learner(x) 
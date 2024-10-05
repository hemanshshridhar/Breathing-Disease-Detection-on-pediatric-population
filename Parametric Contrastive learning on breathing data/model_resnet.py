import torch.nn as nn
import torchvision.models as models
import torch
class Resnet50(nn.Module):
    '''
    Resnet 50.

    Args:
        dim (int): Dimension of the last layer.
    '''
    def __init__(self, dim=128):
        super(Resnet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # Modify the first convolutional layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, dim)

    def forward(self, x):
        out = self.resnet(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        return out / norm

from torchvision.models import resnet18
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 46)

        if freeze:
            param_names = [
                name for name, _ in self.resnet.named_parameters() if "fc" in name
            ]
            self.freeze_layers(param_names=param_names)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def freeze_layers(self, param_names):
        for name, param in self.resnet.named_parameters():
            if name not in param_names:
                param.requires_grad = False

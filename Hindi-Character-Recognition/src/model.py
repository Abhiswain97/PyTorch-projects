from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch


def calculate_conv_output(IH, IW, KH, KW, P, S):
    return ((IH - KH + 2 * P) / S + 1, (IW - KW + 2 * P) / S + 1)


class HNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 32 x 32 x 3 => 28 x 28 x 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))

        # 28 x 28 x 16 => 26 x 26 x 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))

        # 26 x 26 x 32 => 46
        self.fc1 = nn.Linear(26 * 26 * 32, 46)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 26 * 26 * 32)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

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

# Intitialize the model
#####################################
## DO NOT CHANGE THE VARIABLE NAME ##
#####################################
model = HNet()

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvSame(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConvSame, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet_2(nn.Module):
    def __init__(self, c_in, c_out):
        super(UNet_2, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.conv2 = DoubleConvSame(c_in=64, c_out=128)
        self.conv3 = DoubleConvSame(c_in=128, c_out=256)
        self.conv4 = DoubleConvSame(c_in=256, c_out=512)
        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConvSame(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConvSame(c_in=512, c_out=256)
        self.up_conv3 = DoubleConvSame(c_in=256, c_out=128)
        self.up_conv4 = DoubleConvSame(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER"""

        u1 = self.up1(c5)
        cat1 = torch.cat([u1, c4], dim=1)
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        cat2 = torch.cat([u2, c3], dim=1)
        uc2 = self.up_conv2(cat2)

        u3 = self.up3(uc2)
        cat3 = torch.cat([u3, c2], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        cat4 = torch.cat([u4, c1], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs


class UNet_OG(nn.Module):
    def __init__(self, c_in, c_out):
        super(UNet_OG, self).__init__()

        self.conv1 = DoubleConv(c_in=c_in, c_out=64)
        self.conv2 = DoubleConv(c_in=64, c_out=128)
        self.conv3 = DoubleConv(c_in=128, c_out=256)
        self.conv4 = DoubleConv(c_in=256, c_out=512)
        self.conv5 = DoubleConv(c_in=512, c_out=1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConv(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConv(c_in=512, c_out=256)
        self.up_conv3 = DoubleConv(c_in=256, c_out=128)
        self.up_conv4 = DoubleConv(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

    def crop_tensor(self, up_tensor, target_tensor):
        _, _, H, W = up_tensor.shape

        x = CenterCrop(size=(H, W))(target_tensor)

        return x

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        """BOTTLE-NECK"""

        c5 = self.conv5(p4)
        """DECODER"""

        u1 = self.up1(c5)
        crop1 = self.crop_tensor(u1, c4)
        cat1 = torch.cat([u1, crop1], dim=1)
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        crop2 = self.crop_tensor(u2, c3)
        cat2 = torch.cat([u2, crop2], dim=1)
        uc2 = self.up_conv2(cat2)

        u3 = self.up3(uc2)
        crop3 = self.crop_tensor(u3, c2)
        cat3 = torch.cat([u3, crop3], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        crop4 = self.crop_tensor(u4, c1)
        cat4 = torch.cat([u4, crop4], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs


class AttentionBlock(nn.Module):
    """
    Class for creating Attention module
    Takes in gating signal `g` and `x`
    """

    def __init__(self, g_chl, x_chl):
        super(AttentionBlock, self).__init__()

        inter_shape = x_chl // 4

        # Conv 1x1 with stride 2 for `x`
        self.conv_x = nn.Conv2d(
            in_channels=x_chl,
            out_channels=inter_shape,
            kernel_size=1,
            stride=2,
        )

        # Conv 1x1 with stride 1 for `g` (gating signal)
        self.conv_g = nn.Conv2d(
            in_channels=g_chl,
            out_channels=inter_shape,
            kernel_size=1,
            stride=1,
        )

        # Conv 1x1 for `psi` the output after `g` + `x`
        self.psi = nn.Conv2d(
            in_channels=2 * inter_shape,
            out_channels=1,
            kernel_size=1,
            stride=1,
        )

        # For upsampling the attention output to size of `x`
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, g, x):

        # perform the convs on `x` and `g`
        theta_x = self.conv_x(x)
        gate = self.conv_g(g)

        # `theta_x` + `gate`
        add = torch.cat([self.conv_x(x), self.conv_g(g)], axis=1)

        # ReLU on the add operation
        relu = torch.relu(add)

        # the 1x1 Conv
        psi = self.psi(relu)

        # Sigmoid the squash the outputs/attention weights
        sig = torch.sigmoid(psi)

        # Upsample to original size of `x` to perform multiplication
        upsample = self.upsample(sig)

        # return the attention weights!
        return upsample


class AttentionUNet(nn.Module):
    def __init__(self, c_in, c_out):
        super(AttentionUNet, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.conv2 = DoubleConvSame(c_in=64, c_out=128)
        self.conv3 = DoubleConvSame(c_in=128, c_out=256)
        self.conv4 = DoubleConvSame(c_in=256, c_out=512)
        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.attn1 = AttentionBlock(1024, 512)
        self.attn2 = AttentionBlock(512, 256)
        self.attn3 = AttentionBlock(256, 128)
        self.attn4 = AttentionBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConvSame(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConvSame(c_in=512, c_out=256)
        self.up_conv3 = DoubleConvSame(c_in=256, c_out=128)
        self.up_conv4 = DoubleConvSame(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER - WITH ATTENTION"""

        att1 = self.attn1(c5, c4)
        u1 = self.up1(c5)
        mult1 = torch.multiply(att1, u1)
        cat1 = torch.cat([mult1, c4], dim=1)
        uc1 = self.up_conv1(cat1)

        att2 = self.attn2(uc1, c3)
        u2 = self.up2(uc1)
        mult2 = torch.multiply(att2, u2)
        cat2 = torch.cat([mult2, c3], dim=1)
        uc2 = self.up_conv2(cat2)

        att3 = self.attn3(uc2, c2)
        u3 = self.up3(uc2)
        mult3 = torch.multiply(att3, u3)
        cat3 = torch.cat([mult3, c2], dim=1)
        uc3 = self.up_conv3(cat3)

        att4 = self.attn4(uc3, c1)
        u4 = self.up4(uc3)
        mult4 = torch.multiply(att4, u4)
        cat4 = torch.cat([mult4, c1], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs


if __name__ == "__main__":
    model = AttentionUNet(c_in=3, c_out=1)

    summary(
        model,
        (3, 512, 512),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        device=torch.device("cpu"),
    )

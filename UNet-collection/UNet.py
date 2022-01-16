import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop, RandomCrop
from torchvision import models
from torchsummary import summary

# --------------------------------------------------GENREIC - BLOCKS ------------------------------------------------------------


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


class DoubleConvSame3D(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConvSame3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=c_out,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.conv = DoubleConvSame(c_in=in_channels, c_out=in_channels * 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)

        return c, p


class Encoder3D(nn.Module):
    def __init__(self, in_channels):
        super(Encoder3D, self).__init__()

        self.conv = DoubleConvSame3D(c_in=in_channels, c_out=in_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)

        return c, p


# -------------------------------------------------- UNET- SAME PADDING ------------------------------------------------------------


class UNet_2(nn.Module):
    def __init__(self, c_in, c_out):
        super(UNet_2, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = Encoder(64)
        self.enc2 = Encoder(128)
        self.enc3 = Encoder(256)
        self.enc4 = Encoder(512)

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

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

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


# -------------------------------------------------- UNET- ORIGINAL ------------------------------------------------------------


class UNet_OG(nn.Module):
    def __init__(self, c_in, c_out):
        super(UNet_OG, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = Encoder(64)
        self.enc2 = Encoder(128)
        self.enc3 = Encoder(256)
        self.enc4 = Encoder(512)

        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

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

        x = RandomCrop(size=(H, W))(target_tensor)

        return x

    def forward(self, x):

        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

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


# -------------------------------------------------- ATTENTION - UNET ------------------------------------------------------------


class AttentionDecoder(nn.Module):
    def __init__(self, in_channels):
        super(AttentionDecoder, self).__init__()

        self.up_conv = DoubleConvSame(c_in=in_channels, c_out=in_channels // 2)
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )

    def forward(self, conv1, conv2, attn):
        up = self.up(conv1)
        mult = torch.multiply(attn, up)
        cat = torch.cat([mult, conv2], dim=1)
        uc = self.up_conv(cat)

        return uc


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
        add = torch.cat([gate, theta_x], dim=1)

        # ReLU on the add operation
        relu = torch.relu(add)

        # the 1x1 Conv
        psi = self.psi(relu)

        # Sigmoid to squash the outputs/attention weights
        sig = torch.sigmoid(psi)

        # Upsample to original size of `x` to perform multiplication
        upsample = self.upsample(sig)

        # return the attention weights!
        return upsample


class AttentionUNet(nn.Module):
    def __init__(self, c_in, c_out):
        super(AttentionUNet, self).__init__()

        self.conv1 = DoubleConvSame(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = Encoder(64)
        self.enc2 = Encoder(128)
        self.enc3 = Encoder(256)
        self.enc4 = Encoder(512)

        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

        self.attn1 = AttentionBlock(1024, 512)
        self.attn2 = AttentionBlock(512, 256)
        self.attn3 = AttentionBlock(256, 128)
        self.attn4 = AttentionBlock(128, 64)

        self.attndeco1 = AttentionDecoder(1024)
        self.attndeco2 = AttentionDecoder(512)
        self.attndeco3 = AttentionDecoder(256)
        self.attndeco4 = AttentionDecoder(128)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER - WITH ATTENTION"""

        att1 = self.attn1(c5, c4)
        uc1 = self.attndeco1(c5, c4, att1)

        att2 = self.attn2(uc1, c3)
        uc2 = self.attndeco2(c4, c3, att2)

        att3 = self.attn3(uc2, c2)
        uc3 = self.attndeco3(c3, c2, att3)

        att4 = self.attn4(uc3, c1)
        uc4 = self.attndeco4(c2, c1, att4)

        outputs = self.conv_1x1(uc4)

        return outputs


# -------------------------------------------------- ATTENTION - UNET - 3D -------------------------------------------------------------------------


class AttentionDecoder3D(nn.Module):
    def __init__(self, in_channels):
        super(AttentionDecoder3D, self).__init__()

        self.up_conv = DoubleConvSame3D(c_in=in_channels, c_out=in_channels // 2)
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )

    def forward(self, conv1, conv2, attn):
        up = self.up(conv1)
        mult = torch.multiply(attn, up)
        cat = torch.cat([mult, conv2], dim=1)
        uc = self.up_conv(cat)

        return uc


class AttentionBlock3D(nn.Module):
    """
    Class for creating Attention module
    Takes in gating signal `g` and `x`
    """

    def __init__(self, g_chl, x_chl):
        super(AttentionBlock3D, self).__init__()

        inter_shape = x_chl // 4

        # Conv 1x1 with stride 2 for `x`
        self.conv_x = nn.Conv3d(
            in_channels=x_chl,
            out_channels=inter_shape,
            kernel_size=1,
            stride=2,
        )

        # Conv 1x1 with stride 1 for `g` (gating signal)
        self.conv_g = nn.Conv3d(
            in_channels=g_chl,
            out_channels=inter_shape,
            kernel_size=1,
            stride=1,
        )

        # Conv 1x1 for `psi` the output after `g` + `x`
        self.psi = nn.Conv3d(
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
        add = torch.cat([gate, theta_x], dim=1)

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


class AttentionUNet3D(nn.Module):
    def __init__(self, c_in, c_out):
        super(AttentionUNet3D, self).__init__()

        self.conv1 = DoubleConvSame3D(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = Encoder3D(64)
        self.enc2 = Encoder3D(128)
        self.enc3 = Encoder3D(256)
        self.enc4 = Encoder3D(512)

        self.conv5 = DoubleConvSame3D(c_in=512, c_out=1024)

        self.attn1 = AttentionBlock3D(1024, 512)
        self.attn2 = AttentionBlock3D(512, 256)
        self.attn3 = AttentionBlock3D(256, 128)
        self.attn4 = AttentionBlock3D(128, 64)

        self.attndeco1 = AttentionDecoder3D(1024)
        self.attndeco2 = AttentionDecoder3D(512)
        self.attndeco3 = AttentionDecoder3D(256)
        self.attndeco4 = AttentionDecoder3D(128)

        self.conv_1x1 = nn.Conv3d(in_channels=64, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER - WITH ATTENTION"""

        att1 = self.attn1(c5, c4)
        uc1 = self.attndeco1(c5, c4, att1)

        att2 = self.attn2(uc1, c3)
        uc2 = self.attndeco2(c4, c3, att2)

        att3 = self.attn3(uc2, c2)
        uc3 = self.attndeco3(c3, c2, att3)

        att4 = self.attn4(uc3, c1)
        uc4 = self.attndeco4(c2, c1, att4)

        outputs = self.conv_1x1(uc4)

        return outputs


# -------------------------------------------------- 3D - UNET -------------------------------------------------------------------------


class UNet3D(nn.Module):
    def __init__(self, c_in, c_out):
        super(UNet3D, self).__init__()

        self.conv1 = DoubleConvSame3D(c_in=c_in, c_out=64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = Encoder3D(64)
        self.enc2 = Encoder3D(128)
        self.enc3 = Encoder3D(256)
        self.enc4 = Encoder3D(512)

        self.conv5 = DoubleConvSame3D(c_in=512, c_out=1024)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose3d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose3d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose3d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up4 = nn.ConvTranspose3d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.up_conv1 = DoubleConvSame3D(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConvSame3D(c_in=512, c_out=256)
        self.up_conv3 = DoubleConvSame3D(c_in=256, c_out=128)
        self.up_conv4 = DoubleConvSame3D(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv3d(in_channels=64, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

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


if __name__ == "__main__":
    model = UNet_2(c_in=3, c_out=1)
    attn_unet = AttentionUNet(c_in=3, c_out=1)
    unet3d = UNet3D(c_in=3, c_out=1)
    attn_unet3d = AttentionUNet3D(3, 1)

    batch = torch.randn(1, 3, 128, 128, 128)

    print(attn_unet3d(batch).size())

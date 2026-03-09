# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Block modules
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, DCN, DConv
from .transformer import TransformerBlock
from .attention import *
from .rep_block import DiverseBranchBlock,ACBlockDBB,DeepACBlockDBB,DeepDiverseBranchBlock,WideDiverseBranchBlock,ACBlock,FusedDiverseBranchBlock,RecursionDiverseBranchBlock,RecursionDiverseBranchBlockInner,MixedDBB
__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3','C3SECA',
           'C2f_ACDBB', 'C2f_DeepACDBB', 'C2f_DeepDBB', 'C2f_DeepACDBBMix', 'C2f_DBB', 'C2f_ACNET', 'C2f_WDBB','C2f_RDBB','C2f_MixedDBB',
           'C3_MixedDBB','C3_RDBB', 'C2f_CGA', 'C3k2', 'C3k', 'C2PSA', "Attention","DC2f",
           )

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))



# -----------------------------SECABottleneck START-----------------------------

class SimECA(nn.Module):
    def __init__(self, c1, k_size=5,gamma=2, b=1):  # k_size: Adaptive selection of kernel size
        super(SimECA, self).__init__()

        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(c1, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.sigmoid(y)

        return x + y.expand_as(x)



class SECABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.SimECA1=SimECA(c1, k_size=5,gamma=2, b=1)
        self.SimECA2 = SimECA(c2, k_size=5, gamma=2, b=1)

    def forward(self, x):
        return self.SimECA1(x) + self.SimECA2(self.cv2(self.cv1(x))) if self.add else self.SimECA2(self.cv2(self.cv1(x)))

class C3SECA(C3):
    # C3 module with DCNv2
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(SECABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))



class FocalECA(nn.Module):
    def __init__(self, c1, k_size=5,gamma=2, b=1,focal=2):  # k_size: Adaptive selection of kernel size
        super(FocalECA, self).__init__()

        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(c1, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = y.pow(2)
        return x + y.expand_as(x)



class FocalECABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.SimECA1=FocalECA(c1, k_size=5,gamma=2, b=1)
        self.SimECA2 = FocalECA(c2, k_size=5, gamma=2, b=1)

    def forward(self, x):
        return self.SimECA1(x) + self.SimECA2(self.cv2(self.cv1(x))) if self.add else self.SimECA2(self.cv2(self.cv1(x)))

class C3FECA(C3):
    # C3 module with DCNv2
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(FocalECABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))



######################################## C2f-DDB begin ########################################

class Bottleneck_DBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C2f_DBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class Bottleneck_WDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = WideDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = WideDiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C2f_WDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_WDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class Bottleneck_DeepDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeepDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DeepDiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C2f_DeepDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DeepDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class Bottleneck_DeepACDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeepACBlockDBB(c1, c_, k[0], 1)
        self.cv2 = DeepACBlockDBB(c_, c2, k[1], 1, groups=g)

class C2f_DeepACDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DeepACDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))



class Bottleneck_DeepACDBBMix(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeepACBlockDBB(c1, c_, k[0], 1)
        self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C2f_DeepACDBBMix(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DeepACDBBMix(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))





class Bottleneck_ACDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ACBlockDBB(c1, c_, k[0], 1)
        self.cv2 = ACBlockDBB(c_, c2, k[1], 1, groups=g)

class C2f_ACDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ACDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class C3_DBB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DBB(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class Bottleneck_ACNET(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ACBlock(c1, c_, k[0], 1)
        self.cv2 = ACBlock(c_, c2, k[1], 1, groups=g)

class C2f_ACNET(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ACNET(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class C3_ACNET(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ACNET(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))



class Bottleneck_RDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RecursionDiverseBranchBlock(c1, c_, k[0], 1,recursion_layer=2)
        self.cv2 = RecursionDiverseBranchBlock(c_, c2, k[1], 1, groups=g,recursion_layer=2)

class C2f_RDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_RDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class C3_RDBB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RDBB(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


class Bottleneck_MixedDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MixedDBB(c1, c_, k[0], 1)
        self.cv2 = MixedDBB(c_, c2, k[1], 1, groups=g)

class C2f_MixedDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_MixedDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class C3_MixedDBB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MixedDBB(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


class C2f_CGA(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CGAFusion(self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1], y[0]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1], y[0]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

######################################## C2f-DDB end ########################################
# --------------------------------------------------------
# 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# --------------------------------------------------------

# 二维空间注意力（Spatial Attention）模块
class SpatialAttention(nn.Module):
    # 定义一个名为 SpatialAttention 的类，它继承自 PyTorch 的 nn.Module，用于创建一个空间注意力模块
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 在模块中定义一个二维卷积层，用于空间注意力的计算。
        # 输入通道数为 2，输出通道数为 1。
        # 卷积核大小为 7x7，使用 'reflect' 填充模式，填充大小为 3。
        # 卷积层包含偏置项（bias）。
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        # 定义前向传播函数，它接受输入 x 并返回空间注意力的输出。

        x_avg = torch.mean(x, dim=1, keepdim=True)  # torch.Size([3, 1, 64, 64])
        # 计算输入 x 的通道平均值，dim=1 表示沿着通道维度计算平均值。
        # keepdim=True 表示保持输出的维度与输入相同。

        x_max, _ = torch.max(x, dim=1, keepdim=True)  # torch.Size([3, 1, 64, 64])
        # 计算输入 x 的通道最大值，dim=1 表示沿着通道维度寻找最大值。
        # _ 表示我们不关心最大值的索引。

        x2 = torch.cat([x_avg, x_max], dim=1)  # torch.Size([3, 2, 64, 64])
        # 将平均值和最大值沿着通道维度拼接起来，形成一个新的张量 x2。

        sattn = self.sa(x2)  # torch.Size([3, 1, 64, 64])
        # 将拼接后的张量 x2 输入到之前定义的卷积层 self.sa 中，得到空间注意力图。

        # 返回空间注意力图，它将作为模块的输出。
        return sattn


# 定义一个名为 ChannelAttention 的类，继承自 PyTorch 的 nn.Module，用于创建通道注意力模块。
class ChannelAttention(nn.Module):

    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()

        # 使用全局平均池化层，将输入特征图的每个通道的空间维度压缩到一个单一的数值。
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 定义一个通道注意力子网络，使用顺序容器来堆叠层。
        self.ca = nn.Sequential(

            # 第一个卷积层，输入通道数为 dim，输出通道数为 dim // reduction，用于降维。
            # 卷积核大小为 1x1，没有填充（padding=0），包含偏置项（bias=True）。
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),

            # 使用 ReLU 激活函数，inplace=True 表示在原地进行计算，减少内存使用。
            nn.ReLU(inplace=True),

            # 第二个卷积层，输入通道数为经过降维的 dim // reduction，输出通道数恢复为原始的 dim。
            # 同样使用 1x1 卷积核，没有填充，包含偏置项。
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    # 定义前向传播函数，接受输入 x 并返回通道注意力的输出。
    def forward(self, x):
        # x.shape = torch.Size([3, 32, 64, 64])

        # 通过全局平均池化层处理输入 x，得到每个通道的全局空间信息。
        x_gap = self.gap(x)  # torch.Size([3, 32, 1, 1])

        cattn = self.ca(x_gap)  # torch.Size([3, 32, 1, 1])
        # 将全局平均池化的结果输入到通道注意力子网络中，计算每个通道的权重。

        return cattn
        # 返回通道注意力的权重，这些权重将用于调节输入特征图的通道响应


# 定义一个名为 PixelAttention 的类，继承自 PyTorch 的 nn.Module，用于创建像素注意力模块。
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # 调用父类的构造函数。

        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        # 定义一个二维卷积层，用于像素注意力的计算。
        # 输入通道数为 2 * dim，输出通道数为 dim
        # 卷积核大小为 7x7，使用 'reflect' 填充模式，填充大小为 3。
        # 卷积层使用分组卷积，groups=dim，每个组独立卷积。
        # 卷积层包含偏置项（bias=True）。

        self.sigmoid = nn.Sigmoid()
        # 定义 Sigmoid 激活函数，用于将卷积层的输出转换为概率分布。

    # 定义前向传播函数，接受输入特征图 x 和第一个注意力特征 pattn1，并返回像素注意力的输出。
    def forward(self, x, pattn1):
        # 获取输入特征图 x 的形状。
        B, C, H, W = x.shape

        x = x.unsqueeze(dim=2)  # B, C, 1, H, W torch.Size([3, 32, 1, 64, 64])
        # 扩展输入特征图的维度，增加一个维度，用于与 pattn1 拼接。

        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W torch.Size([3, 32, 1, 64, 64])
        # 扩展 pattn1 的维度，与 x 进行拼接。
        # x2.shape = torch.Size([3, 32, 2, 64, 64])
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        # 将 x 和 pattn1 沿着通道维度拼接。
        # x2.shape = torch.Size([3, 64, 64, 64])
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        # 使用 Rearrange 函数重新排列 x2 的形状，这里应该是一个错误，因为 PyTorch 没有内置的 Rearrange 函数。
        # 正确的操作可能是使用 view 或 permute 来重新排列张量。

        pattn2 = self.pa2(x2)  # pattn2 torch.Size([3, 32, 64, 64])
        # 将拼接和重新排列后的张量 x2 输入到之前定义的卷积层 self.pa2 中。

        pattn2 = self.sigmoid(pattn2)  # pattn2 torch.Size([3, 32, 64, 64])
        # 通过 Sigmoid 激活函数处理 self.pa2 的输出，得到像素级别的注意力权重。

        return pattn2
        # 返回像素注意力权重。


class CGAFusion(nn.Module):
    # 定义 CGAFusion 类，继承自 PyTorch 的 nn.Module，用于创建特征融合模块。
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()

        self.sa = SpatialAttention()
        # 创建空间注意力模块。

        self.ca = ChannelAttention(dim, reduction)
        # 创建通道注意力模块，dim 表示输入特征的维度，reduction 用于控制通道降维的比例。

        self.pa = PixelAttention(dim)
        # 创建像素注意力模块，dim 表示输入特征的维度。

        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        # 创建一个二维卷积层，用于处理融合后的特征图，卷积核大小为 1。

        self.sigmoid = nn.Sigmoid()
        # 创建 Sigmoid 激活函数，用于将输出转换为概率分布。

    def forward(self, x, y):
        # 定义前向传播函数，接受两个输入特征图 x 和 y。
        # x.shape = torch.Size([3, 32, 64, 64])
        initial = x + y  # initial.shape torch.Size([3, 32, 64, 64])
        # 将两个输入特征图相加，作为初始融合特征。

        cattn = self.ca(initial)  # cattn torch.Size([3, 32, 1, 1])
        # 计算初始融合特征的通道注意力。

        sattn = self.sa(initial)  # sattn torch.Size([3, 1, 64, 64])
        # 计算初始融合特征的空间注意力。

        pattn1 = sattn + cattn  # torch.Size([3, 32, 64, 64])

        # 将空间注意力和通道注意力相加
        # pattn2 torch.Size([3, 32, 64, 64]) self.pa(initial, pattn1) torch.Size([3, 32, 64, 64])
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        # 计算像素级别的注意力权重，并通过 Sigmoid 函数进行归一化。

        result = initial + pattn2 * x + (1 - pattn2) * y
        # result torch.Size([3, 32, 64, 64])
        # 根据像素注意力权重调整初始融合特征和输入特征的组合，生成最终的融合结果。

        result = self.conv(result)
        # result torch.Size([3, 32, 64, 64])
        # 使用卷积层进一步处理融合结果。

        return result
        # 返回最终的特征融合结果

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class DBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, de=1.0, gc=8):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DConv(c1, c_, k[0], e=de, gc=gc)
        self.cv2 = DConv(c_, c2, k[1], g=g, e=de, gc=gc)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, gc=8):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
         # 8:1.2, 16:1.4
         # dbottleneck2 16: 1.4
        self.m = nn.ModuleList(DBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, de=1.1, gc=gc) for _ in range(n))
        # self.m = nn.ModuleList(DBottleneck2(self.c, self.c, shortcut, g, k=(3, 3), e=1.5, gc=gc) for _ in range(n))


    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
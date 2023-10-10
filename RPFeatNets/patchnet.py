import torch.nn as nn
import torch.nn.functional as F


class BaseNet (nn.Module):
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors=F.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(urepeatability),
                    reliability=self.softmax(ureliability))

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        res = {k: [r[k] for r in res if k in r]
               for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)


class PatchNet (BaseNet):
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append(nn.Conv2d(self.curchan, outd,
                        kernel_size=k, **conv_params))
        if bn and self.bn:
            self.ops.append(self._make_bn(outd))
        if relu:
            self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)


class BackboneNet (PatchNet):
    def __init__(self, dim=128, mchan=4, relu22=False, **kw):
        PatchNet.__init__(self, **kw)
        self._add_conv(8*mchan)
        self._add_conv(8*mchan)
        self._add_conv(16*mchan, stride=2)
        self._add_conv(16*mchan)
        self._add_conv(32*mchan, stride=2)
        self._add_conv(32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv(32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim


class SE4Layer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SE4Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RPFeat (BackboneNet):
    def __init__(self, **kw):
        BackboneNet.__init__(self, **kw)
        self.se1 = SE4Layer(self.out_dim)
        self.se2 = SE4Layer(self.out_dim)
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        ureliability = self.clf(self.se1(x**2))
        urepeatability = self.sal(self.se2(x**2))
        return self.normalize(x, ureliability.mul(urepeatability), urepeatability)

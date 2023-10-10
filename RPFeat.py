from PIL import Image
import torch
from tools import common
from tools.dataloader import norm_RGB
from RPFeatNets.patchnet import *


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    net = eval(checkpoint['net'])
    weights = checkpoint['state_dict']
    net.load_state_dict(
        {k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, scale_f=2**0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s+0.001 >= max(min_scale, min_size / max(H, W)):
        if s-0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            #if verbose:
            #    print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh, nw), mode='bilinear',
                            align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    # scores = reliability * repeatability
    scores = torch.cat(C) * torch.cat(Q)
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores

def get_rpfeat_feature(net, img, detector, scale_f=2**0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       top_k=2000,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s+0.001 >= max(min_scale, min_size / max(H, W)):
        if s-0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            #if verbose:
            #    print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            kpts = detector(**res).T[:,[1,0]]
            index = repeatability[0,0][kpts[:,1],kpts[:,0]].argsort()[-top_k:]
            x = x[index]
            y = y[index]
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm
    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    # scores = reliability * repeatability
    scores = torch.cat(C) * torch.cat(Q)
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(img_path, config):
    iscuda = common.torch_set_gpu(config.gpu)

    # load the network...
    net = load_network(config.model)
    if iscuda:
        net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr=config.reliability_thr,
        rep_thr=config.repeatability_thr)

    #print(f"\nExtracting features for {img_path}")
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    img = norm_RGB(img)[None]
    if iscuda:
        img = img.cuda()

    # extract keypoints/descriptors for a single image
    # extract_multiscale
    xys, desc, scores = get_rpfeat_feature(net, img, detector,
                                           scale_f=config.scale_f,
                                           min_scale=config.min_scale,
                                           max_scale=config.max_scale,
                                           min_size=config.min_size,
                                           max_size=config.max_size,
                                           verbose=True)

    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-config.top_k or None:]
    return xys, desc, scores


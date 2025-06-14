# Timestep is given as a layer as in flownet5_v001
# removed normalization in head
# FiLM modulation on resolution in EMB resblocks

class Model:

    info = {
        'name': 'Flownet5_v010',
        'file': 'flownet5_v010.py',
        'ratio_support': True
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        class myPReLU(Module):
            def __init__(self, c):
                super().__init__()
                self.alpha = 0.2
                self.beta = 0.69
                self.prelu = torch.nn.PReLU(c, 0.2)
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                x = x / self.alpha - self.beta
                tanh_x = self.tanh(x)
                x = torch.where(
                    x > 0, 
                    x, 
                    tanh_x + abs(tanh_x) * self.prelu(x)
                )
                return self.alpha * (x + self.beta)

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, 
                    out_planes, 
                    kernel_size=kernel_size, 
                    stride=stride,
                    padding=padding, 
                    dilation=dilation,
                    padding_mode = 'zeros',
                    bias=True
                ),
                myPReLU(out_planes)
            )

        def warp(tenInput, tenFlow):
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            tenGrid = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
            g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bicubic', padding_mode='border', align_corners=True)

        def normalize(tensor, min_val, max_val):
            src_dtype = tensor.dtype
            tensor = tensor.float()
            t_min = tensor.min()
            t_max = tensor.max()

            if t_min == t_max:
                return torch.full_like(tensor, (min_val + max_val) / 2.0)
            
            tensor = ((tensor - t_min) / (t_max - t_min)) * (max_val - min_val) + min_val
            tensor = tensor.to(dtype = src_dtype)
            return tensor

        def hpass(img):
            # src_dtype = img.dtype
            # img = img.float()
            def gauss_kernel(size=5, channels=6):
                kernel = torch.tensor([[1., 4., 6., 4., 1],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]])
                kernel /= 256.
                kernel = kernel.repeat(channels, 1, 1, 1)
                return kernel
            
            def conv_gauss(img, kernel):
                img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
                out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
                return out

            gkernel = gauss_kernel()
            gkernel = gkernel.to(device=img.device, dtype=img.dtype)
            hp = img - conv_gauss(img, gkernel) + 0.5
            hp = torch.clamp(hp, 0.48, 0.52)
            hp = normalize(hp, 0, 1)
            # hp = torch.max(hp, dim=1, keepdim=True).values
            # hp = hp.to(dtype = src_dtype)
            return hp

        def compress(x):
            src_dtype = x.dtype
            x = x.float()
            x = x * 2 - 1
            scale = torch.tanh(torch.tensor(1.0))
            x = torch.where(
                (x >= -1) & (x <= 1), scale * x,
                torch.tanh(x)
            ) + 0.01 * x
            x = (0.99 * x / scale + 1) / 2
            x = x.to(dtype = src_dtype)
            return x

        def ACEScg2cct(image):
            condition = image <= 0.0078125
            value_if_true = image * 10.5402377416545 + 0.0729055341958155 
            ACEScct = torch.where(condition, value_if_true, image)

            condition = image > 0.0078125
            value_if_true = (torch.log2(image) + 9.72) / 17.52
            ACEScct = torch.where(condition, value_if_true, ACEScct)

            return torch.clamp(ACEScct, 0, 1)

        def ACEScct2cg(image):
            condition = image < 0.155251141552511
            value_if_true = (image - 0.0729055341958155) / 10.5402377416545
            ACEScg = torch.where(condition, value_if_true, image)

            condition = (image >= 0.155251141552511) & (image < (torch.log2(torch.tensor(65504.0)) + 9.72) / 17.52)
            value_if_true = torch.exp2(image * 17.52 - 9.72)
            ACEScg = torch.where(condition, value_if_true, ACEScg)

            ACEScg = torch.clamp(ACEScg, max=65504.0)

            return ACEScg

        class Head(Module):
            def __init__(self, c=32):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(3, c, 5, 2, 2),
                    myPReLU(c),
                    torch.nn.Conv2d(c, c, 3, 1, 1),
                    myPReLU(c),
                    ResConv(c),
                    ResConv(c),
                    torch.nn.ConvTranspose2d(c, 9, 4, 2, 1)
                )
                self.maxdepth = 2

            def forward(self, x):
                n, c, h, w = x.shape
                ph = self.maxdepth - (h % self.maxdepth)
                pw = self.maxdepth - (w % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)
                x = self.encode(x)
                return x[:, :, :h, :w]
            
        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = myPReLU(c)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)
            
        class ResConvEmb(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.PReLU(c, 0.2)
                self.mlp = FeatureModulator(1, c)

            def forward(self, x):
                x_scalar = x[1]
                x = x[0]
                x = self.relu(self.mlp(x_scalar, self.conv(x)) * self.beta + x)
                return x, x_scalar

        class UpMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.up = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
                self.relu = myPReLU(c)

            def forward(self, x, x_deep):
                x_deep = self.up(x_deep)
                return self.relu(self.conv(x_deep) * self.beta + x)

        class Mix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv0 = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.conv1 = torch.nn.Conv2d(c, c, 3, 1, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.up = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
                self.relu = myPReLU(c)

            def forward(self, x, x_deep):
                x_deep = self.up(x_deep)
                return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

        class DownMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, cd, 7, 4, 3, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = myPReLU(cd)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x) * self.beta + x_deep)

        class FeatureModulator(torch.nn.Module):
            def __init__(self, scalar_dim, feature_channels):
                super().__init__()
                self.scale_net = torch.nn.Sequential(
                    torch.nn.Linear(scalar_dim, feature_channels),
                    torch.nn.PReLU(feature_channels, 1),  # or no activation
                )
                self.shift_net = torch.nn.Linear(scalar_dim, feature_channels)
                self.c = feature_channels

            def forward(self, x_scalar, features):
                scale = self.scale_net(x_scalar).view(-1, self.c, 1, 1)
                shift = self.shift_net(x_scalar).view(-1, self.c, 1, 1)
                return features * scale + shift

        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 5, 2, 2),
                    conv(c//2, c, 3, 2, 1),
                    )
                self.convblock = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, conf, flow, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                
                if flow is None:
                    x = torch.cat((img0, img1, f0, f1, timestep), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask, conf), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=True, antialias=True) * 1. / scale
                    x = torch.cat((x, flow), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=True, antialias=True)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetDeepEmb(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)
                self.conv0 = conv(6, c//4, 3, 1, 1)              
                self.conv1 = conv(in_planes, c//2, 5, 2, 2)
                self.conv2 = conv(c//2, c, 5, 2, 2)
                self.conv3 = conv(c, cd, 5, 2, 2)
                self.conv4 = conv(cd, cd, 5, 2, 2)
                self.convblock1 = torch.nn.Sequential(
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                )
                self.convblock2 = torch.nn.Sequential(
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                )
                self.convblock3 = torch.nn.Sequential(
                    ResConvEmb(c),
                    ResConvEmb(c),
                )
                self.convblock1f = torch.nn.Sequential(
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                )
                self.convblock2f = torch.nn.Sequential(
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                )
                self.convblock3f = torch.nn.Sequential(
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                )
                self.convblock_last = torch.nn.Sequential(
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                )
                self.convblock_last_shallow = torch.nn.Sequential(
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                    ResConvEmb(c//4),
                )
                self.convblock_deep1 = torch.nn.Sequential(
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                )
                self.convblock_deep2 = torch.nn.Sequential(
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                )
                self.convblock_deep3 = torch.nn.Sequential(
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                )
                
                self.mix1 = UpMix(c, cd)
                self.mix1f = DownMix(c//4, c)
                self.mix2 = UpMix(c, cd)
                self.mix2f = DownMix(c//4, c)
                self.mix3 = Mix(c, cd)
                self.mix3f = DownMix(c//4, c)
                self.mix4f = DownMix(c//4, c)
                self.revmix1 = DownMix(c, cd)
                self.revmix1f = UpMix(c//4, c)
                self.revmix2 = DownMix(c, cd)
                self.revmix2f = UpMix(c//4, c)
                self.revmix3f = UpMix(c//4, c)
                self.mix4 = Mix(c//4, c)
                self.lastconv = torch.nn.Sequential(
                    torch.nn.Conv2d(c//4, 6, 5, 1, 2),
                )

                self.maxdepth = 16

            def forward(self, img0, img1, f0, f1, timestep, mask, conf, flow, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)

                if flow is None:
                    imgs = torch.cat((img0, img1), 1)
                    # imgs = normalize(imgs, 0, 1) * 2 - 1
                    x = torch.cat((imgs, f0, f1), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                    x = torch.nn.functional.pad(x, padding)
                else:
                    merged = warp(img0, flow[:, :2]) * torch.sigmoid(mask) + warp(img1, flow[:, 2:4]) * (1 - torch.sigmoid(mask))
                    imgs = torch.cat((img0, img1, merged), 1)
                    # imgs = normalize(imgs, 0, 1) * 2 - 1
                    x = torch.cat((imgs, f0, f1, mask, conf), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True) * 1. / scale
                    x = torch.cat((x, flow), 1)
                    x = torch.nn.functional.pad(x, padding)

                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1).to(device=img0.device, dtype=img0.dtype)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.nn.functional.pad(tenGrid, padding, mode='replicate')
                x = torch.cat((x, tenGrid), 1)

                # max_res = max(x.shape[-2:])
                # max_res = torch.full((x.shape[0], 1), 1e-4 * float(max_res)).to(img0.device)
                timestep_tensor = torch.full((x.shape[0], 1), float(timestep)).to(img0.device)
                x_scalar = timestep_tensor
                # x_scalar = torch.cat([max_res, timestep_tensor], dim=1)

                featF = self.conv0(hpass(x[:, :6, :, :]))
                featF, _ = self.convblock1f((featF, x_scalar))

                feat = self.conv1(x)
                feat = self.conv2(feat)
                feat_deep = self.conv3(feat)
                feat_deep = self.conv4(feat_deep)

                feat, _ = self.convblock1((feat, x_scalar))
                feat_deep, _ = self.convblock_deep1((feat_deep, x_scalar))
                
                feat = self.mix1f(featF, feat)

                feat_tmp = self.mix1(feat, feat_deep)
                feat_deep = self.revmix1(feat, feat_deep)

                featF = self.revmix1f(featF, feat_tmp)

                featF, _ = self.convblock2f((featF, x_scalar))
                feat, _ = self.convblock2((feat_tmp, x_scalar))
                feat_deep, _ = self.convblock_deep2((feat_deep, x_scalar))

                feat = self.mix2f(featF, feat)
                feat_tmp = self.mix2(feat, feat_deep)
                feat_deep = self.revmix2(feat, feat_deep)
                featF = self.revmix2f(featF, feat_tmp)

                featF, _ = self.convblock3f((featF, x_scalar))
                feat, _ = self.convblock3((feat_tmp, x_scalar))
                feat_deep, _ = self.convblock_deep3((feat_deep, x_scalar))
                feat = self.mix3f(featF, feat)
                feat = self.mix3(feat, feat_deep)
                
                featF = self.revmix3f(featF, feat)

                feat, _ = self.convblock_last((feat, x_scalar))
                featF, _ = self.convblock_last_shallow((featF, x_scalar))

                feat = self.mix4(featF, feat)

                feat = self.lastconv(feat)
                feat = torch.nn.functional.interpolate(feat[:, :, :sh, :sw], size=(h, w), mode="bicubic", align_corners=True, antialias=True)
                flow = feat[:, :4] * scale
                mask = feat[:, 4:5]
                conf = feat[:, 5:6]
                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeepEmb(24+2, c=144)
                # self.block1 = FlownetDeep(24+5+4+2, c=192)
                # self.block2 = FlownetDeep(24+5+4+2, c=144)
                # self.block3 = Flownet(31, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[1, 8, 4, 1], iterations=4, gt=None):

                img0 = compress(img0)
                img1 = compress(img1)

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                scale[0] = 1

                flow1, mask1, conf1 = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])
                # with torch.no_grad():
                flow2, mask2, conf2 = self.block0(img1, img0, f1, f0, 1-timestep, None, None, None, scale=scale[0])

                flow = (flow1 + torch.cat((flow2[:, 2:4], flow2[:, :2]), 1)) / 2
                mask = (mask1 + (-mask2)) / 2
                conf = (conf1 + conf2) / 2

                mask = torch.sigmoid(mask) #
                conf = torch.sigmoid(conf) #

                result = {
                    'flow_list': [flow],
                    'mask_list': [mask],
                    'conf_list': [conf],
                    'merged': [merged]
                }

                return result

                flow_list[0] = flow.clone()
                conf_list[0] = torch.sigmoid(conf.clone())
                mask_list[0] = torch.sigmoid(mask.clone())
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                '''
                result = {
                    'flow_list': flow_list,
                    'mask_list': mask_list,
                    'conf_list': conf_list,
                    'merged': merged
                }
                return result
                '''

                flow, mask, conf = self.block1(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    mask, 
                    conf, 
                    flow,
                    scale=scale[1])

                flow_list[1] = flow.clone()
                conf_list[1] = torch.sigmoid(conf.clone())
                mask_list[1] = torch.sigmoid(mask.clone())
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                flow, mask, conf = self.block2(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    mask, 
                    conf, 
                    flow,
                    scale=scale[2])
                
                flow_list[2] = flow.clone()
                conf_list[2] = torch.sigmoid(conf.clone())
                mask_list[2] = torch.sigmoid(mask.clone())
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_d, mask_d, conf_d = self.block3(img0, img1, f0, f1, timestep, mask, conf, flow, scale=scale[3])
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[3] = flow
                conf_list[3] = torch.sigmoid(conf) #
                mask_list[3] = torch.sigmoid(mask) #
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                result = {
                    'flow_list': flow_list,
                    'mask_list': mask_list,
                    'conf_list': conf_list,
                    'merged': merged
                }

                return result

        class FlownetCasEval(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeep(24+2, c=192)
                self.block1 = FlownetDeep(24+5+4+2, c=144)
                self.block2 = FlownetDeep(24+5+4+2, c=96)
                self.block3 = Flownet(31, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[24, 16, 12, 8, 4, 1], iterations=1, gt=None):

                img0 = ACEScg2cct(compress(img0))
                img1 = ACEScg2cct(compress(img1))

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])

                flow, mask, conf = self.block1(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    mask, 
                    conf, 
                    flow,
                    scale=scale[1])

                flow, mask, conf = self.block1(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    mask, 
                    conf, 
                    flow,
                    scale=scale[2])

                flow, mask, conf = self.block2(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    mask, 
                    conf, 
                    flow,
                    scale=scale[3])

                flow, mask, conf = self.block2(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    mask, 
                    conf, 
                    flow,
                    scale=scale[4])

                flow_d, mask_d, conf_d = self.block3(img0, img1, f0, f1, timestep, mask, conf, flow, scale=scale[5])
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                mask = torch.sigmoid(mask) #
                conf = torch.sigmoid(conf) #
                merged = warp(img0, flow[:, :2]) * mask + warp(img1, flow[:, 2:4]) * (1 - mask)

                result = {
                    'flow_list': [flow],
                    'mask_list': [mask],
                    'conf_list': [conf],
                    'merged': [merged]
                }

                return result

        self.model = FlownetCas
        self.training_model = FlownetCas

    @staticmethod
    def get_info():
        return Model.info

    @staticmethod
    def get_name():
        return Model.info.get('name')

    @staticmethod
    def input_channels(model_state_dict):
        channels = 3
        try:
            channels = model_state_dict.get('multiresblock1.conv_3x3.conv1.weight').shape[1]
        except Exception as e:
            print (f'Unable to get model dict input channels - setting to 3, {e}')
        return channels

    @staticmethod
    def output_channels(model_state_dict):
        channels = 5
        try:
            channels = model_state_dict.get('conv_final.conv1.weight').shape[0]
        except Exception as e:
            print (f'Unable to get model dict output channels - setting to 3, {e}')
        return channels

    def get_model(self):
        import platform
        if platform.system() == 'Darwin':
            return self.training_model
        return self.model

    def get_training_model(self):
        return self.training_model

    def load_model(self, path, flownet, rank=0):
        import torch
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                flownet.load_state_dict(convert(torch.load(path)), False)
            else:
                flownet.load_state_dict(convert(torch.load(path, map_location ='cpu')), False)
# Orig v001 changed to v002 main flow and signatures
# Back from SiLU to LeakyReLU to test data flow
# Warps moved to flownet forward
# Different Tail from flownet 2lh (ConvTr 6x6, conv 1x1, ConvTr 4x4, conv 1x1)

class Model:

    info = {
        'name': 'Flownet4_v001hpass',
        'file': 'flownet4_v001hpass.py',
        'ratio_support': True
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        backwarp_tenGrid = {}

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, 
                    out_planes, 
                    kernel_size=kernel_size, 
                    stride=stride,
                    padding=padding, 
                    dilation=dilation,
                    padding_mode = 'reflect',
                    bias=True
                ),
                torch.nn.LeakyReLU(0.2, True)
                # torch.nn.SELU(inplace = True)
            )

        def warp(tenInput, tenFlow):
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        def normalize(tensor, min_val, max_val):
            return (tensor - min_val) / (max_val - min_val)

        def compress(x):
            scale = torch.tanh(torch.tensor(1.0))
            x = torch.where(
                (x >= -1) & (x <= 1), scale * x,
                torch.tanh(x)
            )
            x = (x + 1) / 2
            return x

        def blur(img):  
            def gauss_kernel(size=5, channels=3):
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
            return conv_gauss(img, gkernel)

        def blur(img):  
            def gauss_kernel(size=5, channels=3):
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
            return conv_gauss(img, gkernel)

        def hpass(img):
            return img - blur(img)

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(4, 48, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(48, 48, 3, 1, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(48, 48, 3, 1, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.ConvTranspose2d(48, 10, 4, 2, 1)
                )

            def forward(self, x, feat=False):
                x = normalize(x, x.min(), x.max())
                hp = hpass(x)
                hp = normalize(hp, hp.min(), hp.max())
                hp = torch.max(hp, dim=1, keepdim=True).values
                x = torch.cat((x, hp), 1)
                return self.encode(x)

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class ResConvMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x_deep) * self.beta + x)

        class ResConvRevMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x) * self.beta + x_deep)

        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = round(1.618 * c)
                self.conv0 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes, c//2, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(c//2, c, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True)
                    )
                self.conv0D = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes, c//2, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(c//2, c, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.LeakyReLU(0.2, True),
                    )
                self.convblock0 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock0D = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock1 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock1D = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.mix = ResConvMix(c, cd)
                self.revmix = ResConvRevMix(c, cd)

                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2),
                )
                self.lastconvD = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    torch.nn.ConvTranspose2d(cd, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2),
                )
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                if flow is None:
                    x = torch.cat((img0, img1, f0, f1), 1)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, mask), 1)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)
                
                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw)
                tenGrid = torch.cat((
                    tenHorizontal * ((sw - 1.0) / 2.0), 
                    tenVertical * ((sh - 1.0) / 2.0)
                    ), 1).to(device=img0.device, dtype=img0.dtype)
                timestep = (tenGrid[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((x, timestep, tenGrid), 1)

                # x[:, :6, :, :] = normalize(x[:, :6, :, :], x[:, :6, :, :].min(), x[:, :6, :, :].max())

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                feat = self.conv0(x)
                featD = self.conv0D(x)
                feat = self.convblock0(feat)
                featD = self.convblock0D(featD)

                tmp = self.mix(feat, featD)
                featD = self.revmix(feat, featD)

                feat = self.convblock1(tmp)
                featD = self.convblock1D(featD)

                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]

                tmpD = self.lastconvD(featD)
                tmpD = torch.nn.functional.interpolate(tmpD[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)

                flowD = tmpD[:, :4] * scale
                maskD = tmpD[:, 4:5]
                confD = tmpD[:, 5:6]

                return flow, mask, conf, flowD, maskD, confD

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(6+20+1+2, c=96) # images + feat + timestep + lineargrid
                self.block1 = Flownet(8+4+16, c=128)
                self.block2 = Flownet(8+4+16, c=96)
                self.block3 = Flownet(8+4+16, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], iterations=1, gt=None):
                img0 = compress(img0 * 2 - 1)
                img1 = compress(img1 * 2 - 1)
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                flow_list_hpass = [None] * 4
                mask_list_hpass = [None] * 4
                conf_list_hpass = [None] * 4
                merged_hpass = [None] * 4

                scale[0] = 1

                flow, mask, conf, flowD, maskD, confD = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0])

                flow_list_hpass[3] = flow
                conf_list_hpass[3] = torch.sigmoid(conf)
                mask_list_hpass[3] = torch.sigmoid(mask)
                merged_hpass[3] = warp(img0, flow[:, :2]) * mask_list_hpass[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list_hpass[3])

                flow_list[3] = flowD
                conf_list[3] = torch.sigmoid(confD)
                mask_list[3] = torch.sigmoid(maskD)
                merged[3] = warp(img0, flowD[:, :2]) * mask_list[3] + warp(img1, flowD[:, 2:4]) * (1 - mask_list[3])
                
                result = {
                    'flow_list': flow_list,
                    'mask_list': mask_list,
                    'conf_list': conf_list,
                    'merged': merged,
                    'flow_list_hpass': flow_list_hpass,
                    'mask_list_hpass': mask_list_hpass,
                    'conf_list_hpass': conf_list_hpass,
                    'merged_hpass': merged_hpass
                }

                return result



                flow_list[0] = flow.clone()
                conf_list[0] = torch.sigmoid(conf.clone())
                mask_list[0] = torch.sigmoid(mask.clone())
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                for iteration in range(iterations):
                    flow_d, mask, conf = self.block1(
                        img0, 
                        img1,
                        f0,
                        f1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[1]
                    )
                    flow = flow + flow_d

                flow_list[1] = flow.clone()
                conf_list[1] = torch.sigmoid(conf.clone())
                mask_list[1] = torch.sigmoid(mask.clone())
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                for iteration in range(iterations):
                    flow_d, mask, conf = self.block2(
                        img0, 
                        img1,
                        f0,
                        f1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[2]
                    )
                    flow = flow + flow_d

                flow_list[2] = flow.clone()
                conf_list[2] = torch.sigmoid(conf.clone())
                mask_list[2] = torch.sigmoid(mask.clone())
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                for iteration in range(iterations):
                    flow_d, mask, conf = self.block3(
                        img0, 
                        img1,
                        f0,
                        f1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[3]
                    )
                    flow = flow + flow_d

                flow_list[3] = flow
                conf_list[3] = torch.sigmoid(conf)
                mask_list[3] = torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                return flow_list, mask_list, conf_list, merged

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
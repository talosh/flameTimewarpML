class Model:
    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        backwarp_tenGrid = {}

        def hpass(img):
            def gauss_kernel(channels):
                kernel = torch.tensor([[1., 4., 6., 4., 1.],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]])
                kernel /= 256.
                return kernel.repeat(channels, 1, 1, 1)

            def conv_gauss(img, kernel):
                img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
                return torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])

            gkernel = gauss_kernel(img.shape[1])
            gkernel = gkernel.to(device=img.device, dtype=img.dtype)
            hp = img - conv_gauss(img, gkernel)
            
            hp = hp + 0.5

            return hp

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
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, feat=False):
                x = x + 2.0 * (hpass(x)) - 1.0

                x0 = self.cnn0(x)
                x = self.relu(x0)
                x1 = self.cnn1(x)
                x = self.relu(x1)
                x2 = self.cnn2(x)
                x = self.relu(x2)
                x3 = self.cnn3(x)
                if feat:
                    return [x0, x1, x2, x3]
                return x3

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 3, 2, 1),
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

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((img0, img1, f0, f1, timestep), 1)
                x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                if flow is not None:
                    mask = torch.nn.functional.interpolate(mask, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, mask, flow), 1)
                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(7+16, c=192)
                self.block1 = Flownet(8+4+16, c=128)
                self.block2 = Flownet(8+4+16, c=96)
                self.block3 = Flownet(8+4+16, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], iterations=1, bidirectional=False, matte0=None, matte1=None):
                def halving_steps(start, n_steps=4):
                    vals = [max(start // (2 ** i), 1) for i in range(n_steps - 1)]
                    vals.append(1)
                    return vals
 
                def swap_pairs(f):
                    # flow packs two 2-channel fields: [to-img0 | to-img1].
                    # A reverse pass sees the inputs swapped, so its own
                    # convention is [to-img1 | to-img0]. This converts between
                    # the two.
                    return torch.cat((f[:, 2:4], f[:, :2]), 1)

                def black_matte(reference):
                    # Either input matte missing means the output matte is
                    # black — there is nothing meaningful to warp and blend
                    # with only one side present. Channel count follows
                    # whichever matte was actually supplied, defaulting to 1.
                    channels = 1
                    if matte0 is not None:
                        channels = matte0.shape[1]
                    elif matte1 is not None:
                        channels = matte1.shape[1]
                    return torch.zeros(
                        (reference.shape[0], channels, reference.shape[2], reference.shape[3]),
                        dtype=reference.dtype, device=reference.device,
                    )

                have_mattes = matte0 is not None and matte1 is not None
 
                scale = halving_steps(scale[0], 4)
 
                if timestep == 0:
                    matte_out = matte0 if have_mattes else black_matte(img0)
                    return img0, torch.ones_like(img0[:, :1]), matte_out
                if timestep == 1:
                    matte_out = matte1 if have_mattes else black_matte(img1)
                    return img1, torch.ones_like(img1[:, :1]), matte_out
 
                f0 = self.encode(img0)
                f1 = self.encode(img1)
 
                # Stage 0 — no prior flow/mask to refine from.
                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0])
 
                if bidirectional:
                    flow_r, mask_r, conf_r = self.block0(
                        img1, img0, f1, f0, 1 - timestep, None, None, scale=scale[0]
                    )
                    # mask is a pre-sigmoid logit favouring img0; the reverse
                    # pass favours img1, so negate to put it on the same axis.
                    flow = (flow + swap_pairs(flow_r)) * 0.5
                    mask = (mask - mask_r) * 0.5
                    conf = (conf + conf_r) * 0.5
 
                # Stages 1-3 — each refines the flow from the previous stage.
                for block, s in ((self.block1, scale[1]), (self.block2, scale[2]), (self.block3, scale[3])):
                    for _ in range(iterations):
                        flow_d, mask_next, conf = block(
                            warp(img0, flow[:, :2]),
                            warp(img1, flow[:, 2:4]),
                            warp(f0, flow[:, :2]),
                            warp(f1, flow[:, 2:4]),
                            timestep,
                            mask,
                            flow,
                            scale=s,
                        )
 
                        if bidirectional:
                            # Same refinement seen from the other end. Every
                            # argument is swapped, including the flow and mask
                            # priors, which must be handed over in the reverse
                            # pass's own convention.
                            flow_dr, mask_r, conf_r = block(
                                warp(img1, flow[:, 2:4]),
                                warp(img0, flow[:, :2]),
                                warp(f1, flow[:, 2:4]),
                                warp(f0, flow[:, :2]),
                                1 - timestep,
                                -mask,
                                swap_pairs(flow),
                                scale=s,
                            )
                            flow_d    = (flow_d + swap_pairs(flow_dr)) * 0.5
                            mask_next = (mask_next - mask_r) * 0.5
                            conf      = (conf + conf_r) * 0.5
 
                        flow = flow + flow_d
                        mask = mask_next
 
                mask = torch.sigmoid(mask)
                merged = warp(img0, flow[:, :2]) * mask + warp(img1, flow[:, 2:4]) * (1 - mask)

                # conf comes straight off the conv stack and is unbounded;
                # squash it so it is usable as a 0-1 matte.
                conf = torch.sigmoid(conf)

                # Matte is warped exactly like RGB — same flow, same blend
                # weights — so it stays pixel-aligned with the merged frame.
                if have_mattes:
                    matte_out = (warp(matte0, flow[:, :2]) * mask
                                 + warp(matte1, flow[:, 2:4]) * (1.0 - mask))
                else:
                    matte_out = black_matte(merged)

                return merged, conf, matte_out

        self.model = FlownetCas
        self.training_model = FlownetCas

    @staticmethod
    def get_info():
        info = {
            'name': 'Flownet4_v001',
            'file': 'flownet4_v001.py',
            'ratio_support': True
        }
        return info

    @staticmethod
    def get_name():
        return 'TWML_Flownet_v001'

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
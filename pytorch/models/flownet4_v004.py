# Soft-roll PReLU in heads
# 5x5 convs in heads
# Feauture modulator in resblocks
# torch.nn.PReLU channel-wise in head and resblocks
# tenGrid as input
# bicubic aligned scale

class Model:

    info = {
        'name': 'Flownet4_v004',
        'file': 'flownet4_v004.py',
        'ratio_support': True
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module

        class myPReLU(Module):
            def __init__(self, c):
                super().__init__()
                self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 0.2), requires_grad=True)
                self.beta = self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 0.69), requires_grad=True)
                self.prelu = torch.nn.PReLU(c, 0.2)
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                alpha = self.alpha.clamp(min=1e-8)
                x = x / alpha - self.beta
                tanh_x = self.tanh(x)
                x = torch.where(
                    x > 0, 
                    x, 
                    tanh_x + abs(tanh_x) * self.prelu(x)
                )
                return alpha * (x + self.beta)

        class FeatureModulator(torch.nn.Module):
            def __init__(self, scalar_dim, feature_channels):
                super().__init__()
                self.scale_net = torch.nn.Sequential(
                    torch.nn.Linear(scalar_dim, feature_channels),
                    # torch.nn.PReLU(feature_channels, 1),  # or no activation
                )
                self.shift_net = torch.nn.Linear(scalar_dim, feature_channels)
                self.c = feature_channels

            def forward(self, x_scalar, features):
                scale = self.scale_net(x_scalar).view(-1, self.c, 1, 1)
                shift = self.shift_net(x_scalar).view(-1, self.c, 1, 1)
                return features * scale + shift

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
    
        def warp(tenInput, tenFlow):
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            tenGrid = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
            g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bicubic', padding_mode='border', align_corners=True)

        class Head(Module):
            def __init__(self, c=48):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(3, c, 5, 2, 2),
                    myPReLU(c),
                    torch.nn.Conv2d(c, c, 3, 1, 1),
                    torch.nn.PReLU(c, 0.2),
                    torch.nn.Conv2d(c, c, 3, 1, 1),
                    torch.nn.PReLU(c, 0.2),
                    torch.nn.ConvTranspose2d(c, 8, 4, 2, 1)
                )
                self.maxdepth = 2

            def forward(self, x):
                n, c, h, w = x.shape
                ph = self.maxdepth - (h % self.maxdepth)
                pw = self.maxdepth - (w % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)
                x = self.encode(x)[:, :, :h, :w]
                return x
        class FourierChannelAttention(Module):
            def __init__(self, c, latent_dim):
                super().__init__()

                out_channels = max(1, c // 8)

                # self.weight_real = torch.nn.Parameter(torch.ones(1, c, 1, 1))
                # self.weight_imag = torch.nn.Parameter(torch.ones(1, c, 1, 1))

                self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 0.2), requires_grad=True)

                self.encoder = torch.nn.Sequential(
                    # torch.nn.Conv2d(c+2, out_channels, 3, 2, 1),
                    # torch.nn.PReLU(out_channels, 0.2),
                    torch.nn.AdaptiveAvgPool2d((11, 11)),
                    torch.nn.Conv2d(c, out_channels, 1, 1, 0),
                    torch.nn.PReLU(out_channels, 0.2),
                    # torch.nn.Conv2d(c, c, 3, 2, 1),
                    # torch.nn.PReLU(c, 0.2),
                    # torch.nn.Conv2d(c, out_channels, 3, 2, 1),
                    # torch.nn.PReLU(out_channels, 0.2),
                    torch.nn.Flatten(),
                    torch.nn.Linear(121 * out_channels, latent_dim),
                    torch.nn.PReLU(latent_dim, 0.2)
                )
                self.fc1 = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, 121 * c),
                    torch.nn.Softplus(), # torch.nn.Sigmoid(),
                )
                '''
                self.fc1up = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(2*c, c, 4, 2, 1),
                    torch.nn.Softplus(),
                )
                '''
                
                self.fc2 = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, c),
                    torch.nn.Softplus()
                )
                self.fc3 = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, c),
                )

                # self.weight_spat = torch.nn.Parameter(torch.ones(1, c, 1, 1))
                # self.weight_chan = torch.nn.Parameter(torch.ones(1, c, 1, 1))
                # self.weight_bias = torch.nn.Parameter(torch.zeros(1, c, 1, 1))

                self.c = c

            def forward(self, x):
                B, C, H, W = x.shape
                x_fft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2 + 1]
                _, _, sh, sw = x_fft.shape

                # weight_complex = torch.complex(self.weight_real, self.weight_imag)
                # x_fft = x_fft * weight_complex

                mag = x_fft.abs()
                phase = x_fft.angle()

                # grid_x = torch.linspace(0, 1, sw, device=x.device).view(1, 1, 1, sw).expand(B, 1, sh, sw)
                # grid_y = torch.linspace(0, 1, sh, device=x.device).view(1, 1, sh, 1).expand(B, 1, sh, sw)
                # latent = self.encoder(torch.cat([torch.log1p(mag), grid_x, grid_y], dim=1))
                latent = self.encoder(torch.log1p(mag) + 0.1 * mag)
                spat_at = self.fc1(latent).view(-1, self.c, 11, 11)
                # spat_at = self.fc1up(spat_at)
                spat_at = torch.nn.functional.interpolate(
                    spat_at, 
                    size=(sh, sw), 
                    mode="bilinear",
                    align_corners=True, 
                    antialias=True
                    )
                mag = mag * spat_at # * self.weight_spat

                x_fft = torch.polar(mag, phase)
                x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')

                chan_scale = self.fc2(latent).view(-1, self.c, 1, 1)
                chan_bias = self.fc3(latent).view(-1, self.c, 1, 1)
                x = x * chan_scale + chan_bias

                return x

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(c, 0.2)
                self.mlp = FeatureModulator(1, c)
                self.atn = FourierChannelAttention(c, c//4)

            def forward(self, x):
                x_scalar = x[1]
                x = x[0]
                x = self.relu(self.mlp(x_scalar, self.atn(self.conv(x))) * self.beta + x)
                return x, x_scalar
        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes, c//2, 5, 2, 2, padding_mode = 'zeros'),
                    myPReLU(c//2),
                    torch.nn.Conv2d(c//2, c, 5, 2, 2, padding_mode = 'reflect'),
                    torch.nn.PReLU(c, 0.2),
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
                        
                if flow is None:
                    x = torch.cat((img0, img1, f0, f1), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                    tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1).to(device=img0.device, dtype=img0.dtype)
                    tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw).to(device=img0.device, dtype=img0.dtype)
                    tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                    x = torch.cat((x, tenGrid), 1)
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, mask, conf), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True) * 1. / scale
                    x = torch.cat((x, flow), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                timestep = torch.full((x.shape[0], 1), float(timestep)).to(img0.device)

                feat = self.conv0(x)
                feat, _ = self.convblock((feat, timestep))
                tmp = self.lastconv(feat)

                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bicubic", align_corners=True, antialias=True)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(6+16+2, c=192)
                self.block1 = Flownet(6+16+2+4, c=128)
                self.block2 = Flownet(6+16+2+4, c=96)
                self.block3 = Flownet(6+16+2+4, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[12, 8, 4, 1], iterations=1, gt=None):
                img0 = compress(img0)
                img1 = compress(img1)
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                flow, mask, conf = self.block0(
                    img0,
                    img1,
                    f0,
                    f1,
                    timestep,
                    None,
                    None, 
                    None,
                    scale=scale[0])

                flow_list[0] = flow.clone()
                conf_list[0] = torch.sigmoid(conf.clone())
                mask_list[0] = torch.sigmoid(mask.clone())
                # merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                flow_d, mask_d, conf_d = self.block1(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    conf,
                    flow, 
                    scale=scale[1]
                )
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[1] = flow.clone()
                conf_list[1] = torch.sigmoid(conf.clone())
                mask_list[1] = torch.sigmoid(mask.clone())
                # merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                flow_d, mask, conf = self.block2(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    conf,
                    flow, 
                    scale=scale[2]
                )
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[2] = flow.clone()
                conf_list[2] = torch.sigmoid(conf.clone())
                mask_list[2] = torch.sigmoid(mask.clone())
                # merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_d, mask, conf = self.block3(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    conf,
                    flow, 
                    scale=scale[3]
                )
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[3] = flow
                conf_list[3] = torch.sigmoid(conf)
                mask_list[3] = torch.sigmoid(mask)
                # merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

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
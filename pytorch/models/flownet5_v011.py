# Orig v001 changed to v002 main flow and signatures
# Back from SiLU to LeakyReLU to test data flow
# Warps moved to flownet forward
# Different Tail from flownet 2lh (ConvTr 6x6, conv 1x1, ConvTr 4x4, conv 1x1)

class Model:

    info = {
        'name': 'Flownet5_v011',
        'file': 'flownet5_v011.py',
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
                torch.nn.PReLU(out_planes, 0.2)
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

        def diffmatte(tensor1, tensor2):
            difference = torch.norm(tensor1 - tensor2, p=2, dim=1, keepdim=True)
            max_val = difference.view(difference.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            difference_normalized = difference / (max_val + 1e-8)
            return difference_normalized

        class HighPassFilter(Module):
            def __init__(self):
                super(HighPassFilter, self).__init__()
                self.register_buffer('gkernel', self.gauss_kernel())

            def gauss_kernel(self, channels=1):
                kernel = torch.tensor([
                    [1., 4., 6., 4., 1],
                    [4., 16., 24., 16., 4.],
                    [6., 24., 36., 24., 6.],
                    [4., 16., 24., 16., 4.],
                    [1., 4., 6., 4., 1.]
                ])
                kernel /= 256.
                kernel = kernel.repeat(channels, 1, 1, 1)
                return kernel

            def conv_gauss(self, img, kernel):
                img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
                out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
                return out

            def rgb_to_luminance(self, rgb):
                weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
                return (rgb * weights).sum(dim=1, keepdim=True)

            def normalize(self, tensor, min_val, max_val):
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
                tensor = tensor * (max_val - min_val) + min_val
                return tensor

            def forward(self, img):
                img = self.rgb_to_luminance(img)
                hp = img - self.conv_gauss(img, self.gkernel) + 0.5
                hp = torch.clamp(hp, 0.48, 0.52)
                hp = self.normalize(hp, 0, 1)
                return hp

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

        class ChannelAttention(Module):
            def __init__(self, c, latent_dim=48, reduction=8, spat=3):
                super().__init__()
                out_channels = max(1, c // reduction)

                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(c, c//2, 3, 2, 1),
                    torch.nn.PReLU(c//2, 0.2),
                    torch.nn.Conv2d(c//2, c//4, 3, 1, 1),
                    torch.nn.PReLU(c//4, 0.2),
                    torch.nn.AdaptiveAvgPool2d((3, 3)),
                    torch.nn.Conv2d(c//4, out_channels, 1, 1, 0),
                    torch.nn.PReLU(out_channels, 0.2),
                    torch.nn.Flatten(start_dim=1),
                    torch.nn.Linear(3 * 3 * out_channels, latent_dim),
                    torch.nn.PReLU(latent_dim, 0.2)
                )
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, c),
                    torch.nn.Sigmoid()
                )
                self.c = c
            
            def forward(self, x):
                latent = self.encoder(x)
                chan_scale = self.fc(latent)
                chan_scale = self.fc(latent).view(-1, self.c, 1, 1)
                x = x * chan_scale.clamp(min=1e-6)
                return x

        class FourierChannelAttention(Module):
            def __init__(self, c, latent_dim, out_channels, bands = 11, norm = False):
                super().__init__()

                self.bands = bands
                self.norm = norm
                self.c = c

                self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 1.0), requires_grad=True)

                self.precomp = torch.nn.Sequential(
                    torch.nn.Conv2d(c + 2, c, 3, 1, 1),
                    torch.nn.PReLU(c, 0.2),
                    torch.nn.Conv2d(c, c, 3, 1, 1),
                    torch.nn.PReLU(c, 0.2),
                )

                self.encoder = torch.nn.Sequential(
                    torch.nn.AdaptiveMaxPool2d((bands, bands)),
                    torch.nn.Conv2d(c, out_channels, 1, 1, 0),
                    torch.nn.PReLU(out_channels, 0.2),
                    torch.nn.Flatten(start_dim=1),
                    torch.nn.Linear(bands * bands * out_channels, latent_dim),
                    torch.nn.PReLU(latent_dim, 0.2)
                )
                self.fc1 = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, bands * bands * c),
                    torch.nn.Sigmoid(),
                )
                self.fc1_scaler = torch.nn.Sequential(
                    torch.nn.Conv2d(c, c, 1, 1, 0),
                    torch.nn.ReLU()
                )
                self.fc2 = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, c),
                    torch.nn.Sigmoid(),
                )
                self.fc2_scaler = torch.nn.Sequential(
                    torch.nn.Conv2d(c, c, 1, 1, 0),
                    torch.nn.ReLU()
                )

            def normalize_fft_magnitude(self, mag, sh, sw, target_size=(64, 64)):
                """
                mag: [B, C, sh, sw]
                Returns: [B, C, Fy, Fx]
                """
                B, C, _, _ = mag.shape
                Fy, Fx = target_size

                mag_reshaped = mag.view(B * C, 1, sh, sw)
                norm_mag = torch.nn.functional.interpolate(
                    mag_reshaped, size=(Fy, Fx), mode='bilinear', align_corners=False
                )
                norm_mag = norm_mag.view(B, C, Fy, Fx)
                return norm_mag

            def denormalize_fft_magnitude(self, norm_mag, sh, sw):
                """
                norm_mag: [B, C, Fy, Fx]
                Returns: [B, C, sh, sw]
                """
                B, C, Fy, Fx = norm_mag.shape

                norm_mag = norm_mag.view(B * C, 1, Fy, Fx)
                mag = torch.nn.functional.interpolate(
                    norm_mag, size=(sh, sw), mode='bilinear', align_corners=False
                )
                mag = mag.view(B, C, sh, sw)
                return mag
            
            def forward(self, x):
                B, C, H, W = x.shape
                x_fft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2 + 1]
                _, _, sh, sw = x_fft.shape

                mag = x_fft.abs()
                phase = x_fft.angle()

                if self.norm:
                    mag_n = self.normalize_fft_magnitude(mag, sh, sw, target_size=(64, 64))
                else:
                    mag_n = torch.nn.functional.interpolate(
                        mag, 
                        size=(64, 64), 
                        mode="bilinear",
                        align_corners=False, 
                        )

                mag_n = torch.log1p(mag_n) + self.alpha * mag_n
                grid_x = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 1, 64).expand(B, 1, 64, 64)
                grid_y = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 64, 1).expand(B, 1, 64, 64)
                mag_n = self.precomp(torch.cat([mag_n, grid_x, grid_y], dim=1))

                latent = self.encoder(mag_n)
                spat_at = self.fc1(latent).view(-1, self.c, self.bands, self.bands)
                spat_at = self.fc1_scaler(spat_at)
                if self.norm:
                    spat_at = self.denormalize_fft_magnitude(spat_at, sh, sw)
                else:
                    spat_at = torch.nn.functional.interpolate(
                        spat_at, 
                        size=(sh, sw), 
                        mode="bilinear",
                        align_corners=False, 
                        )

                mag = mag * spat_at.clamp(min=1e-6)

                x_fft = torch.polar(mag, phase)
                x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')

                chan_scale = self.fc2(latent).view(-1, self.c, 1, 1)
                chan_scale = self.fc2_scaler(chan_scale)
                x = x * chan_scale.clamp(min=1e-6)
                return x

        class Head(Module):
            def __init__(self, c=48):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(4, c, 5, 2, 2),
                    myPReLU(c),
                    torch.nn.Conv2d(c, c, 3, 1, 1),
                    torch.nn.PReLU(c, 0.2),
                    torch.nn.Conv2d(c, c, 3, 1, 1),
                    torch.nn.PReLU(c, 0.2),
                )
                self.attn = FourierChannelAttention(c, c, 11)
                self.lastconv = torch.nn.ConvTranspose2d(c, 9, 4, 2, 1)
                self.hpass = HighPassFilter()
                self.maxdepth = 2

            def forward(self, x):
                hp = self.hpass(x)
                x = torch.cat((x, hp), 1)
                n, c, h, w = x.shape
                ph = self.maxdepth - (h % self.maxdepth)
                pw = self.maxdepth - (w % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)
                x = self.encode(x)
                x = self.attn(x)
                x = self.lastconv(x)[:, :, :h, :w]
                return x

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.PReLU(c, 0.2)

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

        class ResConvDummy(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(c, 0.2)

            def forward(self, x):
                x_scalar = x[1]
                x = x[0]
                x = self.relu(self.conv(x) * self.beta + x)
                return x, x_scalar

        class UpMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(c, 0.2)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x_deep) * self.beta + x)

        class Mix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv0 = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.conv1 = torch.nn.Conv2d(c, c, 3, 1, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(c, 0.2)

            def forward(self, x, x_deep):
                return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

        class DownMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(cd, 0.2)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x) * self.beta + x_deep)

        class FlownetDeep(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)

                self.register_buffer("forward_counter", torch.tensor(0, dtype=torch.long))

                self.conv00 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes + 1, c//2, 5, 2, 2, padding_mode = 'zeros'),
                    myPReLU(c//2),
                    )                
                self.conv10 = torch.nn.Sequential(
                    torch.nn.Conv2d(c//2, c, 5, 2, 2, padding_mode = 'reflect'),
                    torch.nn.PReLU(c, 0.2),
                    )
                self.conv20 = torch.nn.Sequential(
                    torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect'),
                    torch.nn.PReLU(cd, 0.2),     
                )

                self.convblock1 = torch.nn.Sequential(
                    ResConvDummy(c),
                    ResConvDummy(c),
                    ResConvDummy(c),
                    ResConvDummy(c),
                )

                self.convblock10 = torch.nn.Sequential(
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                    ResConvEmb(c),
                )
                self.convblock10f = torch.nn.Sequential(
                    ResConvEmb(c//2),
                    ResConvEmb(c//2),
                    ResConvEmb(c//2),
                    ResConvEmb(c//2),
                )
                self.convblock_deep10 = torch.nn.Sequential(
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                    ResConvEmb(cd),
                )

                self.mix10 = UpMix(c, cd)
                self.mix10f = DownMix(c//2, c)
                self.revmix10 = DownMix(c, cd)
                self.revmix10f = UpMix(c//2, c)

                self.convblock2 = torch.nn.Sequential(
                    ResConvDummy(c),
                    ResConvDummy(c),
                    ResConvDummy(c),
                )
                self.convblock3 = torch.nn.Sequential(
                    ResConvDummy(c),
                    ResConvDummy(c),
                )
                self.convblock1f = torch.nn.Sequential(
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                )
                self.convblock2f = torch.nn.Sequential(
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                )
                self.convblock3f = torch.nn.Sequential(
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                )
                self.convblock_last = torch.nn.Sequential(
                    ResConvDummy(c),
                    ResConvDummy(c),
                    ResConvDummy(c),
                    ResConvDummy(c),
                )
                self.convblock_last_shallow = torch.nn.Sequential(
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                    ResConvDummy(c//2),
                )
                self.convblock_deep1 = torch.nn.Sequential(
                    ResConvDummy(cd),
                    ResConvDummy(cd),
                    ResConvDummy(cd),
                    ResConvDummy(cd),
                )
                self.convblock_deep2 = torch.nn.Sequential(
                    ResConvDummy(cd),
                    ResConvDummy(cd),
                    ResConvDummy(cd),
                )
                self.convblock_deep3 = torch.nn.Sequential(
                    ResConvDummy(cd),
                    ResConvDummy(cd),
                )

                self.attn_deep = ChannelAttention(cd)

                self.mix1 = UpMix(c, cd)
                self.mix1f = DownMix(c//2, c)
                self.mix2 = UpMix(c, cd)
                self.mix2f = DownMix(c//2, c)
                self.mix3 = Mix(c, cd)
                self.mix3f = DownMix(c//2, c)
                self.mix4f = DownMix(c//2, c)
                self.revmix1 = DownMix(c, cd)
                self.revmix1f = UpMix(c//2, c)
                self.revmix2 = DownMix(c, cd)
                self.revmix2f = UpMix(c//2, c)
                self.revmix3f = UpMix(c//2, c)
                self.mix4 = Mix(c//2, c)
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c//2, 6, 4, 2, 1),
                )
                self.maxdepth = 16

                self.mix_ratio = 0

            def resize_min_side(self, tensor, size):
                B, C, H, W = tensor.shape

                if H <= W:
                    new_h = size
                    new_w = int(round(W * (56 / H)))
                else:
                    new_w = size
                    new_h = int(round(H * (56 / W)))

                return torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=True)

            def forward(self, img0, img1, f0, f1, timestep, mask, conf, flow, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)

                imgs = torch.cat((img0, img1), 1)
                x = torch.cat((imgs, f0, f1, diffmatte(img0, img1)), 1)
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=True, antialias=True)
                x = torch.nn.functional.pad(x, padding)

                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1).to(device=img0.device, dtype=img0.dtype)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.nn.functional.pad(tenGrid, padding, mode='replicate')
                timestep_emb = torch.full((x.shape[0], 1), float(timestep)).to(img0.device)
                timestep = (tenGrid[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((timestep, x, tenGrid), 1)


                self.forward_counter += 1

                # Sigmoid-based schedule
                midpoint = 20000.0
                steepness = 0.00011
                counter_f = self.forward_counter.float()
                self.mix_ratio = torch.sigmoid(steepness * (counter_f - midpoint))

                feat = self.conv00(x)

                featF, _ = self.convblock1f((feat, timestep_emb))
                featF_emb, _ = self.convblock10f((feat, timestep_emb))

                feat_ = self.conv10(feat)
                feat_deep_ = self.conv20(feat_)

                _, _, dh, dw = feat_deep.shape
                # feat_deep = self.resize_min_side(feat_deep, 48)
                feat_deep = self.attn_deep(feat_deep)

                feat, _ = self.convblock1((feat_, timestep_emb))
                feat_emb, _ = self.convblock10((feat_, timestep_emb))

                feat_deep, _ = self.convblock_deep1((feat_deep_, timestep_emb))
                feat_deep_emb, _ = self.convblock_deep10((feat_deep_, timestep_emb))

                feat = self.mix1f(featF, feat)
                feat_tmp = self.mix1(feat, feat_deep)
                feat_deep = self.revmix1(feat, feat_deep)
                featF = self.revmix1f(featF, feat_tmp)

                feat_emb = self.mix10f(featF_emb, feat_emb)
                feat_tmp_emb = self.mix10(feat_emb, feat_deep_emb)
                feat_deep_emb = self.revmix10(feat_emb, feat_deep_emb)
                featF_emb = self.revmix10f(featF_emb, feat_tmp_emb)

                featF = (1 - self.mix_ratio) * featF + self.mix_ratio * featF_emb
                feat = (1 - self.mix_ratio) * feat + self.mix_ratio * feat_emb
                feat_deep = (1 - self.mix_ratio) * feat_deep + self.mix_ratio * feat_deep_emb

                featF, _ = self.convblock2f((featF, timestep_emb))
                feat, _ = self.convblock2((feat_tmp, timestep_emb))
                feat_deep, _ = self.convblock_deep2((feat_deep, timestep_emb))

                feat = self.mix2f(featF, feat)
                feat_tmp = self.mix2(
                    feat,
                    feat_deep 
                    # torch.nn.functional.interpolate(feat_deep, size=(dh, dw), mode='bilinear', align_corners=True)
                    )
                feat_deep = self.revmix2(
                    feat,
                    feat_deep 
                    # torch.nn.functional.interpolate(feat_deep, size=(dh, dw), mode='bilinear', align_corners=True)
                    )
                featF = self.revmix2f(featF, feat_tmp)

                featF, _ = self.convblock3f((featF, timestep_emb))
                feat, _ = self.convblock3((feat_tmp, timestep_emb))
                feat_deep, _ = self.convblock_deep3((feat_deep, timestep_emb))
                
                feat = self.mix3f(featF, feat)
                feat = self.mix3(
                    feat,
                    feat_deep
                    # torch.nn.functional.interpolate(feat_deep, size=(dh, dw), mode='bilinear', align_corners=True)
                    )
                
                featF = self.revmix3f(featF, feat)

                feat, _ = self.convblock_last((feat, timestep_emb))
                featF, _ = self.convblock_last_shallow((featF, timestep_emb))

                feat = self.mix4(featF, feat)

                feat = self.lastconv(feat)
                feat = torch.nn.functional.interpolate(feat[:, :, :sh, :sw], size=(h, w), mode="bicubic", align_corners=True, antialias=True)
                flow = feat[:, :4] * scale
                mask = feat[:, 4:5]
                conf = feat[:, 5:6]
                return flow, mask, conf

        def find_scale(x_query):
            x_known = [576, 1024, 2048, 4096]
            y_known = [3, 5, 9, 12]
            n = len(x_known)
            if x_query <= x_known[0]:
                x0, x1 = x_known[0], x_known[1]
                y0, y1 = y_known[0], y_known[1]
            elif x_query >= x_known[-1]:
                x0, x1 = x_known[-2], x_known[-1]
                y0, y1 = y_known[-2], y_known[-1]
            else:
                for i in range(1, n):
                    if x_query < x_known[i]:
                        x0, x1 = x_known[i-1], x_known[i]
                        y0, y1 = y_known[i-1], y_known[i]
                        break
            t = (x_query - x0) / (x1 - x0)
            start = y0 + t * (y1 - y0)
            scale_list = torch.linspace(start, 1.0, steps=4).tolist()
            scale_list = [round(v) for v in scale_list]
            return scale_list

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeep(24+2+1, c=192)
                self.block1 = None # FlownetDeep(24+5+4+2+1, c=128)
                self.block2 = None # FlownetDeep(24+5+4+2+1, c=96)
                self.block3 = None # Flownet(31, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[12, 8, 4, 1], iterations=4, gt=None):

                img0 = compress(img0)
                img1 = compress(img1)

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                '''
                flow1, mask1, conf1 = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])
                flow2, mask2, conf2 = self.block0(img1, img0, f1, f0, 1-timestep, None, None, None, scale=scale[0])

                flow = (flow1 + torch.cat((flow2[:, 2:4], flow2[:, :2]), 1)) / 2
                mask = (mask1 + (-mask2)) / 2
                conf = (conf1 + conf2) / 2
                '''

                scale[0] = 1

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])

                mask = torch.sigmoid(mask) #
                conf = torch.sigmoid(conf) #
                # merged = warp(img0, flow[:, :2]) * mask + warp(img1, flow[:, 2:4]) * (1 - mask)

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
                # merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

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
                # merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

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
                # merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_d, mask_d, conf_d = self.block3(img0, img1, f0, f1, timestep, mask, conf, flow, scale=scale[3])
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[3] = flow
                conf_list[3] = torch.sigmoid(conf) #
                mask_list[3] = torch.sigmoid(mask) #
                # merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

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
                self.block0 = FlownetDeep(24+2+1, c=192)
                self.block1 = FlownetDeep(24+5+4+2+1, c=128)
                self.block2 = FlownetDeep(24+5+4+2+1, c=96)
                self.block3 = Flownet(31, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[32, 32, 32, 32, 32, 32], iterations=4, gt=None):

                size = max(img0.shape[2], img0.shape[3])
                scale = find_scale(size)

                img0 = compress(img0)
                img1 = compress(img1)

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow1, mask1, conf1 = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])
                flow2, mask2, conf2 = self.block0(img1, img0, f1, f0, 1-timestep, None, None, None, scale=scale[0])

                flow = (flow1 + torch.cat((flow2[:, 2:4], flow2[:, :2]), 1)) / 2
                mask = (mask1 + (-mask2)) / 2
                conf = (conf1 + conf2) / 2

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
    
    @staticmethod
    def freeze(net = None):
        for param in net.block0.parameters():
            param.requires_grad = False

        for param in net.block0.conv00.parameters():
            param.requires_grad = True
        for param in net.block0.conv10.parameters():
            param.requires_grad = True
        for param in net.block0.conv20.parameters():
            param.requires_grad = True
        for param in net.block0.attn_deep.parameters():
            param.requires_grad = True

        for param in net.block0.convblock10f.parameters():
            param.requires_grad = True
        for param in net.block0.convblock10.parameters():
            param.requires_grad = True
        for param in net.block0.convblock_deep10.parameters():
            param.requires_grad = True

        for param in net.block0.mix10.parameters():
            param.requires_grad = True
        for param in net.block0.mix10f.parameters():
            param.requires_grad = True
        for param in net.block0.revmix10.parameters():
            param.requires_grad = True
        for param in net.block0.revmix10f.parameters():
            param.requires_grad = True

        # for param in net.encode.parameters():
        #    param.requires_grad = False

        #for param in net.block0.convblock_last[0].mlp.parameters():
        #    param.requires_grad = True
        #for param in net.block0.convblock_last[1].mlp.parameters():
        #    param.requires_grad = True
        #for param in net.block0.convblock_last[2].mlp.parameters():
        #    param.requires_grad = True
        # for param in net.block0.convblock_last[3].parameters():
        #    param.requires_grad = True
        #for param in net.block0.convblock_last_shallow[0].mlp.parameters():
        #    param.requires_grad = True
        # for param in net.block0.convblock_last_shallow[1].mlp.parameters():
        #    param.requires_grad = True
        
        '''
        for param in net.block0.convblock_last_shallow[2].parameters():
            param.requires_grad = True
        for param in net.block0.convblock_last_shallow[3].parameters():
            param.requires_grad = True

        for param in net.block0.lastconv.parameters():
            param.requires_grad = True
        for param in net.block0.mix4.parameters():
            param.requires_grad = True
        '''

        '''
        
        for param in net.block0.attn_deep.parameters():
            param.requires_grad = True        

        for param in net.block0.conv0.parameters():
            param.requires_grad = True
        for param in net.block0.conv1.parameters():
            param.requires_grad = True
        for param in net.block0.conv2.parameters():
            param.requires_grad = True
        for param in net.block0.convblock1.parameters():
            param.requires_grad = True
        for param in net.block0.convblock1f.parameters():
            param.requires_grad = True
        for param in net.block0.convblock_deep1.parameters():
            param.requires_grad = True
        '''

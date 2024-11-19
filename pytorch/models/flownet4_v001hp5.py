class Model:

    info = {
        'name': 'Flownet4_v001hp5',
        'file': 'flownet4_v001hp5.py',
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
                    padding_mode = 'zeros',
                    bias=True
                ),
                torch.nn.LeakyReLU(0.2, True)
                # torch.nn.SELU(inplace = True)
            )

        def warp(tenInput, tenFlow):

            print (tenInput.dtype)
            print (tenFlow.dtype)

            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device)
            # tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        class ChannelAttention(Module):
            def __init__(self, c, reduction_ratio=4):
                super(ChannelAttention, self).__init__()
                self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # Global average pooling
                self.max_pool = torch.nn.AdaptiveMaxPool2d(1)  # Global max pooling

                # Shared MLP layers
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(c, c // reduction_ratio, bias=False),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Linear(c // reduction_ratio, c, bias=False)
                )
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                batch_size, channels, _, _ = x.size()

                # Apply average pooling and reshape
                avg_pool = self.avg_pool(x).view(batch_size, channels)
                # Apply max pooling and reshape
                max_pool = self.max_pool(x).view(batch_size, channels)

                # Forward pass through MLP
                avg_out = self.mlp(avg_pool)
                max_out = self.mlp(max_pool)

                # Combine outputs and apply sigmoid activation
                out = avg_out + max_out
                out = self.sigmoid(out).view(batch_size, channels, 1, 1)
                out = out.expand_as(x)

                # Scale input feature maps
                return out


        class SpatialAttention(Module):
            def __init__(self, kernel_size=5):
                super(SpatialAttention, self).__init__()
                padding = kernel_size // 2  # Ensure same spatial dimensions
                self.conv0 = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode='zeros', bias=False)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                # Compute average and max along the channel dimension
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)

                # Concatenate along channel dimension
                x_cat = torch.cat([avg_out, max_out], dim=1)

                # Apply convolution and sigmoid activation
                out = self.conv0(x_cat)
                out = self.sigmoid(out)
                out = out.expand_as(x)

                return out


        class CBAM(Module):
            def __init__(self, c, reduction_ratio=4, spatial_kernel_size=7, channel_scale=0., spatial_scale=0.):
                super(CBAM, self).__init__()
                self.channel_attention = ChannelAttention(c, reduction_ratio)
                self.spatial_attention = SpatialAttention(kernel_size = spatial_kernel_size)
                self.channel_scale = torch.nn.Parameter(torch.full((1, 1, 1, 1), channel_scale), requires_grad=True)
                self.channel_offset = torch.nn.Parameter(torch.full((1, 1, 1, 1), 1 - abs(channel_scale)/2), requires_grad=True)
                self.spatial_scale = torch.nn.Parameter(torch.full((1, 1, 1, 1), spatial_scale), requires_grad=True)
                self.spatial_offset = torch.nn.Parameter(torch.full((1, 1, 1, 1), 1 - abs(spatial_scale)/2), requires_grad=True)

            def forward(self, x):
                channel_attention = self.channel_attention(x) * self.channel_scale + self.channel_offset
                spatial_attention = self.spatial_attention(x) * self.spatial_scale + self.spatial_offset

                x = x * channel_attention
                x = x * spatial_attention

                return x

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(4, 32, 5, 2, 2)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)
                self.attn = CBAM(32)

            def highpass(self, img):            
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

                def normalize(tensor, min_val, max_val):
                    return (tensor - min_val) / (max_val - min_val)

                gkernel = gauss_kernel()
                gkernel = gkernel.to(device=img.device, dtype=img.dtype)
                blurred = conv_gauss(img, gkernel)
                # img = torch.nn.functional.interpolate(img, scale_factor = 1 / 2, mode="bilinear", align_corners=False)
                # blurred = torch.nn.functional.interpolate(blurred, scale_factor = 1 / 2, mode="bilinear", align_corners=False)
                hp = img - blurred + 0.5
                # hp = torch.nn.functional.interpolate(hp, scale_factor = 2, mode="bicubic", align_corners=False)
                hp = torch.clamp(hp, 0.48, 0.52)
                hp = normalize(hp, hp.min(), hp.max())
                hp = torch.max(hp, dim=1, keepdim=True).values
                return hp

            def blur(self, img):
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
                return  conv_gauss(img, gkernel)

            def forward(self, x):
                x = torch.cat((x, self.highpass(x)), 1)
                x = self.cnn0(x)
                x = self.relu(x)
                x = self.cnn1(x)
                x = self.relu(x)
                x = self.attn(x)
                x = self.cnn2(x)
                x = self.relu(x)
                x = self.cnn3(x)
                return x

        class ResConvAttn(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.attn = CBAM(c)       
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)

            def forward(self, x):
                return self.relu(self.attn(self.conv(x)) * self.beta + x)

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)

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
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                print ('hello')
                print (img0.shape)

                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((img0, img1, f0, f1, timestep), 1)
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                if flow is not None:
                    mask = torch.nn.functional.interpolate(mask, size=(sh, sw), mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, mask, flow), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

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
                self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x) * self.beta + x_deep)


        class FlownetDeepSingleHead(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = int(1.618 * c)
                self.conv0 = conv(in_planes, c, 5, 2, 2)
                self.conv1 = conv(c, c, 3, 2, 1)
                self.conv2 = conv(c, cd, 3, 2, 1)
                self.convblock_shallow = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock1 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock2 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock3 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock4 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock_mask = torch.nn.Sequential(
                    torch.nn.Conv2d(c, c//2, 1, 1, 0),
                    ResConv(c//2),
                    ResConv(c//2),
                )
                self.convblock_deep1 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep2 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep3 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep4 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                )
                self.mix1 = ResConvMix(c, cd)
                self.mix2 = ResConvMix(c, cd)
                self.mix3 = ResConvMix(c, cd)
                self.mix4 = ResConvMix(c, cd)
                self.revmix1 = ResConvRevMix(c, cd)
                self.revmix2 = ResConvRevMix(c, cd)
                self.lastconv_mask = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c//2, 4*2, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.lastconv_flow = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*4, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.hardtanh = torch.nn.Hardtanh()
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, timestep, mask, conf, flow, scale=1, encoder = None):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep

                if flow is None:
                    conf = img0[:, :1].clone() * 0 + 1
                    x = torch.cat((img0, img1, f0, f1, conf, timestep), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                else:
                    merged = warp(img0, flow[:, :2]) * torch.sigmoid(mask) + warp(img1, flow[:, 2:4]) * (1 - torch.sigmoid(mask))
                    fm = encoder(merged)

                    x = torch.cat((
                        img0, 
                        merged, 
                        img1,
                        f0,
                        fm,
                        f1,
                        timestep,
                        mask,
                        conf), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) # * 1. / scale
                    x = torch.cat((x, flow), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                n, c, xh, xw = x.shape
                tenHorizontal = torch.linspace(-1.0, 1.0, xw).view(1, 1, 1, xw).expand(n, -1, xh, -1)
                tenVertical = torch.linspace(-1.0, 1.0, xh).view(1, 1, xh, 1).expand(n, -1, -1, xw)
                tenGrid = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=img0.device, dtype=img0.dtype)

                x = torch.cat((x, tenGrid), 1)

                # noise = torch.rand_like(feat[:, :2, :, :]) * 2 - 1
                feat = self.conv0(x)
                feat = self.convblock_shallow(feat)
                feat = self.conv1(feat)

                feat_deep = self.conv2(feat)
                feat_deep = self.convblock_deep1(feat_deep)
                feat = self.mix1(feat, feat_deep)

                feat_deep = self.convblock_deep2(feat_deep)
                feat = self.convblock1(feat)

                tmp = self.revmix1(feat, feat_deep)
                feat = self.mix2(feat, feat_deep)

                feat_deep = self.convblock_deep3(tmp)
                feat = self.convblock2(feat)

                tmp = self.revmix2(feat, feat_deep)
                feat = self.mix3(feat, feat_deep)

                feat_deep = self.convblock_deep4(tmp)
                feat = self.convblock3(feat)
                feat = self.mix4(feat, feat_deep)

                feat = self.convblock4(feat)

                feat_mask = self.convblock_mask(feat)
                tmp_mask = self.lastconv_mask(feat_mask)

                flow = self.lastconv_flow(feat)
                tmp_mask = torch.nn.functional.interpolate(tmp_mask[:, :, :sh, :sw], size=(h, w), mode="bicubic", align_corners=False)
                flow = torch.nn.functional.interpolate(flow[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)

                flow = self.hardtanh(flow) # * scale
                mask = tmp_mask[:, 0:1]
                conf = tmp_mask[:, 1:2]

                return flow, mask, conf


        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeepSingleHead(6+16+1+1+2, c=144)
                self.block1 = FlownetDeepSingleHead(9+24+1+1+1+4+2, c=112)
                self.block2 = FlownetDeepSingleHead(9+24+1+1+1+4+2, c=96)
                self.block3 = FlownetDeepSingleHead(9+24+1+1+1+4+2, c=64)
                self.block4 = FlownetDeepSingleHead(9+24+1+1+1+4+2, c=48) # Flownet(8+4+16+1, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[8, 4, 2, 1], iterations=1):
                # scale = [8, 4, 2, 1]
                img0 = img0
                img1 = img1
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                scale = [8, 5, 3, 2] if scale == [8, 4, 2, 1] else scale

                # scale[0] = 1

                # scale[1] = 1

                # scale[0] = scale[2]
                # scale[1] = scale[3]
                # scale[2] = 1

                # scale[0] = scale[1]
                # scale[1] = scale[2]
                # scale[2] = scale[3]
                # scale[3] = 1

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])

                '''
                flow_list[0] = flow.clone()
                mask_list[0] = torch.sigmoid(mask.clone())
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])
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
                    scale=scale[1],
                    encoder = self.encode
                )

                '''
                flow_list[1] = flow.clone()
                mask_list[1] = torch.sigmoid(mask.clone())
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])
                '''
                flow, mask, conf = self.block2(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    conf,
                    flow, 
                    scale=scale[2],
                    encoder = self.encode
                )

                '''
                flow_list[2] = flow.clone()
                mask_list[2] = torch.sigmoid(mask.clone())
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])
                '''

                flow, mask, conf = self.block3(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    conf,
                    flow,
                    scale=scale[3],
                    encoder = self.encode
                )

                flow, mask, conf = self.block4(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    conf,
                    flow,
                    scale=1,
                    encoder = self.encode
                )

                # flow = flow + flow_d
                # mask = mask + mask_d
                # conf = conf + conf_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                conf_list[3] = torch.sigmoid(conf)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                flow_list[3][:, 0:1, :, :] = flow_list[3][:, 0:1, :, :] * ((flow.shape[3] - 1.0) / 2.0)
                flow_list[3][:, 1:2, :, :] = flow_list[3][:, 1:2, :, :] * ((flow.shape[2] - 1.0) / 2.0)
                flow_list[3][:, 2:3, :, :] = flow_list[3][:, 2:3, :, :] * ((flow.shape[3] - 1.0) / 2.0)
                flow_list[3][:, 3:4, :, :] = flow_list[3][:, 3:4, :, :] * ((flow.shape[2] - 1.0) / 2.0)

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
# Orig v001 changed to v002 main flow and signatures
# Back from SiLU to LeakyReLU to test data flow
# Warps moved to flownet forward
# Different Tail from flownet 2lh (ConvTr 6x6, conv 1x1, ConvTr 4x4, conv 1x1)
# Padding moved into Flownet modules
# Initial flow estimator changed to Deep module
# First two refine steps are Deep modules
# Two more refine steps are standart modules with no flow given
# Channles are 192, 96, 64 for deep and 48, 32 for standart
# Head feature encoder added attention block

class Model:

    info = {
        'name': 'Flownet4_v001eb_dv04e',
        'file': 'flownet4_v001eb_dv04e.py',
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
                    padding_mode='reflect',
                    bias=True
                ),
                # torch.nn.Mish(True)
                torch.nn.LeakyReLU(0.2, True)
            )

        def conv_mish(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, 
                    out_planes, 
                    kernel_size=kernel_size, 
                    stride=stride,
                    padding=padding, 
                    dilation=dilation,
                    padding_mode='reflect',
                    bias=True
                ),
                torch.nn.Mish(True)
                # torch.nn.LeakyReLU(0.2, True)
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
                self.conv0 = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode='reflect', bias=False)
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
            def __init__(self, c, reduction_ratio=4, spatial_kernel_size=5, channel_scale=0.2, spatial_scale=0.2):
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
                self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.attn = CBAM(32, channel_scale=-0.1, spatial_scale=0.1)
                self.relu = torch.nn.Mish(True)

            def forward(self, x):
                x = self.cnn0(x * 2 - 1)
                x = self.relu(x)
                x = self.attn(x)
                x = self.cnn1(x)
                x = self.relu(x)
                x = self.cnn2(x)
                x = self.relu(x)
                x = self.cnn3(x)
                return x

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class ResConvMish(Module):
            def __init__(self, c):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.Mish(True)

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
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c, 3, 2, 1),
                    conv(c, c, 3, 2, 1),
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
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.noise_level = torch.nn.Parameter(torch.full((1, 1, 1, 1), 1e-4), requires_grad=True)
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                pvalue = scale * self.maxdepth
                _, _, h, w = img0.shape
                ph = ((h - 1) // pvalue + 1) * pvalue
                pw = ((w - 1) // pvalue + 1) * pvalue
                padding = (0, pw - w, 0, ph - h)

                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, :2])
                warped_f0 = warp(f0, flow[:, :2])
                warped_f1 = warp(f1, flow[:, 2:4])
                x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1)
                x = torch.nn.functional.pad(x, padding)
                x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)

                flow = torch.nn.functional.pad(flow, padding)
                flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                x = torch.cat((x, flow), 1)

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4][:, :, :h, :w] * scale
                mask = tmp[:, 4:5][:, :, :h, :w]
                conf = tmp[:, 5:6][:, :, :h, :w]
                return flow, mask, conf

        class FlownetDeepSingleHead(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = int(1.618 * c)
                self.conv0 = conv(in_planes, c, 7, 2, 3)
                self.conv1 = conv(c, c, 3, 2, 1)
                self.conv2 = conv(c, cd, 3, 2, 1)
                self.attn = CBAM(c, channel_scale=-0.1, spatial_scale=0.1)
                self.convblock_shallow = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock1 = torch.nn.Sequential(
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
                    CBAM(c, channel_scale=-0.1, spatial_scale=0.1),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock_deep1 = torch.nn.Sequential(
                    CBAM(cd, channel_scale=-0.1, spatial_scale=0.1),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep2 = torch.nn.Sequential(
                    CBAM(cd, channel_scale=-0.1, spatial_scale=0.1),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep3 = torch.nn.Sequential(
                    CBAM(cd, channel_scale=-0.1, spatial_scale=0.1),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep4 = torch.nn.Sequential(
                    CBAM(cd, channel_scale=-0.1, spatial_scale=0.1),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.mix1 = ResConvMix(c, cd)
                self.mix2 = ResConvMix(c, cd)
                self.mix3 = ResConvMix(c, cd)
                self.mix4 = ResConvMix(c, cd)
                self.revmix1 = ResConvRevMix(c, cd)
                self.revmix2 = ResConvRevMix(c, cd)
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 6*4, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.lastconv_long = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                    torch.nn.Conv2d(c//2, c//2, kernel_size=3, stride=1, padding=1, padding_mode = 'reflect', bias=False),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.ConvTranspose2d(c//2, c//4, 4, 2, 1),
                    torch.nn.Conv2d(c//4, 6, kernel_size=3, stride=1, padding=1, padding_mode = 'reflect', bias=False),
                )
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):

                pvalue = scale * self.maxdepth
                _, _, h, w = img0.shape
                ph = ((h - 1) // pvalue + 1) * pvalue
                pw = ((w - 1) // pvalue + 1) * pvalue
                padding = (0, pw - w, 0, ph - h)
                
                if flow is None:
                    x = torch.cat((img0, img1, f0, f1, timestep), 1)
                    x = torch.nn.functional.pad(x, padding)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    
                    x = torch.cat((img0, img1, f0, f1, warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1)
                    x = torch.nn.functional.pad(x, padding)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)

                    flow = torch.nn.functional.pad(flow, padding)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)

                feat = self.conv0(x)
                feat = self.attn(feat)
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
                tmp = self.lastconv_long(feat)

                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4][:, :, :h, :w] * scale
                mask = tmp[:, 4:5][:, :, :h, :w]
                conf = tmp[:, 5:6][:, :, :h, :w]

                return flow, mask, conf

        class FlownetDeepDualHead(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = conv(in_planes, c, 3, 2, 1)
                self.conv1 = conv(c, c, 3, 2, 1)

                self.conv0_clean = conv_mish(in_planes-5, c, 3, 2, 1)
                self.conv1_clean = conv_mish(c, c, 3, 2, 1)
                self.conv2_deep = conv_mish(c, c*2, 3, 2, 1)
                self.attn = CBAM(c)

                self.conv_deep = torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect', bias=True)
                self.beta_deep = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu_deep = torch.nn.LeakyReLU(0.2, True)

                self.conv_mix = torch.nn.Conv2d(c*2, c, 3, 1, 1, padding_mode = 'reflect', bias=True)
                self.beta_mix = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu_mix = torch.nn.LeakyReLU(0.2, True)

                self.convblock_deep = torch.nn.Sequential(
                    ResConv(c*2),
                    ResConv(c*2),
                    ResConv(c*2),
                    ResConv(c*2),
                    torch.nn.ConvTranspose2d(c*2, c, 4, 2, 1),
                )
                self.convblock_clean = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock_mix = torch.nn.Sequential(
                    ResConv(c*2),
                    ResConv(c*2),
                    ResConv(c*2),
                    ResConv(c*2),
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
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):

                pvalue = scale * self.maxdepth
                _, _, h, w = img0.shape
                ph = ((h - 1) // pvalue + 1) * pvalue
                pw = ((w - 1) // pvalue + 1) * pvalue
                padding = (0, pw - w, 0, ph - h)
                
                x_clean = torch.cat((img0, img1, f0, f1, timestep), 1)
                x_clean = torch.nn.functional.pad(x_clean, padding)
                x_clean = torch.nn.functional.interpolate(x_clean, scale_factor= 1. / scale, mode="bilinear", align_corners=False)

                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

                warped_f0 = warp(f0, flow[:, :2])
                warped_f1 = warp(f1, flow[:, 2:4])
                
                x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1)
                x = torch.nn.functional.pad(x, padding)
                x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)

                flow = torch.nn.functional.pad(flow, padding)
                flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                x = torch.cat((x, flow), 1)

                feat = self.conv0(x)
                feat = self.conv1(feat)

                feat_clean = self.conv0_clean(x_clean)
                feat_clean = self.attn(feat_clean)
                feat_clean = self.conv1_clean(feat_clean)

                feat_deep = self.conv2_deep(feat_clean)
                feat_deep = self.convblock_deep(feat_deep)

                feat_clean = self.relu_deep(self.conv_deep(feat_deep) * self.beta_deep + feat_clean)
                feat_clean = self.convblock_clean(feat_clean)
                
                feat_mix = self.convblock_mix(torch.cat((feat_clean, feat), 1))
                feat = self.relu_mix(self.conv_mix(feat_mix) * self.beta_mix + feat)

                feat = self.convblock(feat)
                tmp = self.lastconv(feat)

                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4][:, :, :h, :w] * scale
                mask = tmp[:, 4:5][:, :, :h, :w]
                conf = tmp[:, 5:6][:, :, :h, :w]

                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeepSingleHead(23, c=192)
                self.block1 = Flownet(28, c=96)
                self.block2 = Flownet(28, c=64)
                self.block3 = Flownet(28, c=48)
                self.block4 = Flownet(28, c=32)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[8, 4, 2, 1], iterations=1):
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                if not torch.is_tensor(timestep):
                    timestep = (img0[:, :1].clone() * 0 + 1) * timestep

                flow_list = [None] * 5
                mask_list = [None] * 5
                conf_list = [None] * 5
                merged = [None] * 5

                scale = [8, 5, 3, 2] if scale == [8, 4, 2, 1] else scale                
                
                # step training

                # stage 1
                scale[0] = 1

                # stage 2
                # scale[0] = scale[3]
                # scale[1] = 1

                # stage 3
                # scale[0] = scale[2]
                # scale[1] = scale[3]
                # scale[2] = 1

                flow, mask, conf = self.block0(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    None, 
                    None,
                    scale=scale[0]
                    )
                
                flow_list[0] = flow.clone()
                mask_list[0] = torch.sigmoid(mask)
                conf_list[0] = torch.sigmoid(conf)
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                # '''
                # step training stage 1
                flow_list[4] = flow_list[0]
                mask_list[4] = mask_list[0]
                conf_list[4] = conf_list[0]
                merged[4] = merged[0]

                return flow_list, mask_list, conf_list, merged
                # '''

                # refine step 1
                flow_d, mask, conf_d = self.block1(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow, 
                    scale=scale[1]
                )

                conf = conf + conf_d
                flow = flow + flow_d

                flow_list[1] = flow.clone()
                mask_list[1] = torch.sigmoid(mask)
                conf_list[1] = torch.sigmoid(conf)
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                # '''
                # step training stage 2
                flow_list[4] = flow_list[1]
                mask_list[4] = mask_list[1]
                conf_list[4] = conf_list[1]
                merged[4] = merged[1]

                return flow_list, mask_list, conf_list, merged
                # '''

                # refine step 2
                flow_d, mask, conf_d = self.block2(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow, 
                    scale=scale[2]
                )

                conf = conf + conf_d
                flow = flow + flow_d

                flow_list[2] = flow.clone()
                mask_list[2] = torch.sigmoid(mask)
                conf_list[2] = torch.sigmoid(conf)
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                '''
                # step training stage 03
                flow_list[4] = flow_list[2]
                mask_list[4] = mask_list[2]
                conf_list[4] = conf_list[2]
                merged[4] = merged[2]

                return flow_list, mask_list, conf_list, merged
                '''

                # refine step 3
                flow_d, mask, conf_d = self.block3(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow,
                    scale=scale[3]
                )

                conf = conf + conf_d
                flow = flow + flow_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                conf_list[3] = torch.sigmoid(conf)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                # refine step 4
                flow_d, mask, conf_d = self.block4(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow,
                    scale=1
                )

                conf = conf + conf_d
                flow = flow + flow_d

                flow_list[4] = flow
                mask_list[4] = torch.sigmoid(mask)
                conf_list[4] = torch.sigmoid(conf)
                merged[4] = warp(img0, flow[:, :2]) * mask_list[4] + warp(img1, flow[:, 2:4]) * (1 - mask_list[4])

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
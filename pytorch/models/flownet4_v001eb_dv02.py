# Orig v001 changed to v002 main flow and signatures
# Back from SiLU to LeakyReLU to test data flow
# Warps moved to flownet forward
# Different Tail from flownet 2lh (ConvTr 6x6, conv 1x1, ConvTr 4x4, conv 1x1)

class Model:

    info = {
        'name': 'Flownet4_v001eb_dv02',
        'file': 'flownet4_v001eb_dv02.py',
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

        class ChannelAttention(Module):
            def __init__(self, c, reduction_ratio=16):
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
            def __init__(self, kernel_size=7):
                super(SpatialAttention, self).__init__()
                padding = kernel_size // 2  # Ensure same spatial dimensions
                self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode='reflect', bias=False)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                # Compute average and max along the channel dimension
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)

                # Concatenate along channel dimension
                x_cat = torch.cat([avg_out, max_out], dim=1)

                # Apply convolution and sigmoid activation
                out = self.conv(x_cat)
                out = self.sigmoid(out)
                out = out.expand_as(x)

                # Scale input feature maps
                return out

        class CBAM(Module):
            def __init__(self, c, reduction_ratio=16, spatial_kernel_size=9):
                super(CBAM, self).__init__()
                self.channel_attention = ChannelAttention(c, reduction_ratio)
                self.spatial_attention = SpatialAttention(spatial_kernel_size)
                self.channel_scale = torch.nn.Parameter(torch.full((1, 1, 1, 1), 0.5), requires_grad=True)
                self.channel_offset = torch.nn.Parameter(torch.full((1, 1, 1, 1), 0.75), requires_grad=True)
                self.spatial_scale = torch.nn.Parameter(torch.full((1, 1, 1, 1), 0.5), requires_grad=True)
                self.spatial_offset = torch.nn.Parameter(torch.full((1, 1, 1, 1), 0.75), requires_grad=True)

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
                self.relu = torch.nn.LeakyReLU(0.2, True)
                self.encode = torch.nn.Sequential(
                    self.cnn0,
                    self.relu,
                    self.cnn1,
                    self.relu,
                    self.cnn2,
                    self.relu,
                    self.cnn3
                )

            def forward(self, x):
                return self.encode(x)
                '''
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
                '''

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)

                torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
                self.conv.weight.data *= 1e-2
                if self.conv.bias is not None:
                    torch.nn.init.constant_(self.conv.bias, 0)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class FlownetShallow(Module):
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
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, c, 6, 2, 2),
                    torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
                    torch.nn.ConvTranspose2d(c, c, 4, 2, 1),
                    torch.nn.Conv2d(c, 6, kernel_size=1, stride=1, padding=0, bias=True),
                )

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, :2])
                warped_f0 = warp(f0, flow[:, :2])
                warped_f1 = warp(f1, flow[:, 2:4])
                x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1)
                x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                x = torch.cat((x, flow), 1)

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class Flownet(Module):
            def __init__(self, in_planes, c=64, cd=96):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 3, 2, 1),
                    conv(c//2, c, 3, 2, 1),
                    )
                self.conv1 = torch.nn.Sequential(
                    conv(in_planes, cd//2, 3, 2, 1),
                    conv(cd//2, cd, 3, 2, 1),
                    conv(cd, cd, 3, 2, 1),
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
                self.convblock_deep = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    torch.nn.ConvTranspose2d(cd, cd, 4, 2, 1)
                )

                self.mix = torch.nn.Conv2d(c+cd, c+cd, kernel_size=1, stride=1, padding=0, bias=True)

                self.attention = CBAM(c+cd)
                
                self.convblock_mix = torch.nn.Sequential(
                    ResConv(c+cd),
                    ResConv(c+cd),
                    ResConv(c+cd),
                    ResConv(c+cd),
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c+cd, c, 6, 2, 2),
                    torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
                    torch.nn.ConvTranspose2d(c, c, 4, 2, 1),
                    torch.nn.Conv2d(c, 6, kernel_size=1, stride=1, padding=0, bias=True),
                )
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, timestep, mask, conf, flow, scale=1):

                pvalue = scale * self.maxdepth
                _, _, h, w = img0.shape
                ph = ((h - 1) // pvalue + 1) * pvalue
                pw = ((w - 1) // pvalue + 1) * pvalue
                padding = (0, pw - w, 0, ph - h)
                
                if flow is None:
                    x = torch.cat((img0, img1, f0, f1, timestep), 1)
                    x = torch.nn.functional.pad(x, padding)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    x_deep = x
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])

                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    
                    x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask, flow), 1)
                    x = torch.nn.functional.pad(x, padding)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    x_deep = x

                    '''
                    x_deep = torch.cat((img0, img1, f0, f1, timestep, conf, flow), 1)
                    x_deep = torch.nn.functional.pad(x_deep, padding)
                    x_deep = torch.nn.functional.interpolate(x_deep, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    '''


                feat = self.conv0(x)
                feat = self.convblock(feat)

                feat_deep = self.conv1(x_deep)
                feat_deep = self.convblock_deep(feat_deep)

                feat = torch.cat((feat_deep, feat), 1)
                feat = self.mix(feat)
                feat = self.attention(feat)
                feat = self.convblock_mix(feat)
                tmp = self.lastconv(feat)

                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4][:, :, :h, :w] * scale
                mask = tmp[:, 4:5][:, :, :h, :w]
                conf = tmp[:, 5:6][:, :, :h, :w]

                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(23, c=192, cd=192)
                self.block1 = Flownet(28, c=96, cd=96)
                self.block2 = Flownet(28, c=64, cd=64)
                self.block3 = FlownetShallow(28, c=48)
                self.block4 = FlownetShallow(28, c=32)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[8, 4, 2, 1], iterations=1):
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                if not torch.is_tensor(timestep):
                    timestep = (img0[:, :1].clone() * 0 + 1) * timestep

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                # scale = [5, 3, 2, 1] if scale == [8, 4, 2, 1] else scale

                scale = [x if x == 8 else x + 1 for x in scale]
                
                # step training

                # 2 steps
                scale[0] = scale[2]
                scale[0] = scale[3]
                scale[1] = 1

                flow, mask, conf = self.block0(
                    img0, 
                    img1, 
                    f0, 
                    f1, 
                    timestep, 
                    None, 
                    None,
                    None,
                    scale=scale[0]
                    )
                
                flow_list[0] = flow.clone()
                mask_list[0] = torch.sigmoid(mask.clone())
                conf_list[0] = torch.sigmoid(conf.clone())
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                '''
                # step training
                flow_list[3] = flow_list[0]
                mask_list[3] = mask_list[0]
                conf_list[3] = conf_list[0]
                merged[3] = merged[0]

                return flow_list, mask_list, conf_list, merged
                '''

                # refine step 1
                flow_d, mask, conf_d = self.block1(
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

                conf = conf
                flow = flow + flow_d

                flow_list[1] = flow.clone()
                mask_list[1] = torch.sigmoid(mask.clone())
                conf_list[1] = torch.sigmoid(conf.clone())
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                # refine step 2
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

                conf = conf
                flow = flow + flow_d

                flow_list[2] = flow.clone()
                mask_list[2] = torch.sigmoid(mask.clone())
                conf_list[2] = torch.sigmoid(conf.clone())
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_list[3] = flow_list[2]
                mask_list[3] = mask_list[2]
                conf_list[3] = conf_list[2]
                merged[3] = merged[2]

                return flow_list, mask_list, conf_list, merged

                # refine step 3
                pvalue = scale[3] * self.maxdepth
                ph = ((h - 1) // pvalue + 1) * pvalue
                pw = ((w - 1) // pvalue + 1) * pvalue
                padding = (0, pw - w, 0, ph - h)

                flow_d, mask, conf = self.block3(
                    torch.nn.functional.pad(img0, padding), 
                    torch.nn.functional.pad(img1, padding),
                    torch.nn.functional.pad(f0, padding),
                    torch.nn.functional.pad(f1, padding),
                    torch.nn.functional.pad(timestep, padding, mode='replicate'),
                    torch.nn.functional.pad(mask, padding),
                    torch.nn.functional.pad(flow, padding), 
                    scale=scale[3]
                )

                flow_d = flow_d[:, :, :h, :w]
                mask = mask[:, :, :h, :w]
                conf = conf[:, :, :h, :w]
                flow = flow + flow_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                # refine step 4
                pvalue = 1 * self.maxdepth
                ph = ((h - 1) // pvalue + 1) * pvalue
                pw = ((w - 1) // pvalue + 1) * pvalue
                padding = (0, pw - w, 0, ph - h)

                flow_d, mask, conf = self.block4(
                    torch.nn.functional.pad(img0, padding), 
                    torch.nn.functional.pad(img1, padding),
                    torch.nn.functional.pad(f0, padding),
                    torch.nn.functional.pad(f1, padding),
                    torch.nn.functional.pad(timestep, padding, mode='replicate'),
                    torch.nn.functional.pad(mask, padding),
                    torch.nn.functional.pad(flow, padding), 
                    scale=1
                )

                flow_d = flow_d[:, :, :h, :w]
                mask = mask[:, :, :h, :w]
                conf = conf[:, :, :h, :w]
                flow = flow + flow_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                return flow_list, mask_list, merged

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
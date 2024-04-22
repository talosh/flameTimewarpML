class Model:
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
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        class Conv2d_batchnorm(Module):
            '''
            2D Convolutional layers

            Arguments:
                num_in_filters {int} -- number of input filters
                num_out_filters {int} -- number of output filters
                kernel_size {tuple} -- size of the convolving kernel
                stride {tuple} -- stride of the convolution (default: {(1, 1)})
                activation {str} -- activation function (default: {'relu'})

            '''

            def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = 'relu'):
                super().__init__()
                self.activation = activation
                self.conv1 = torch.nn.Conv2d(
                    in_channels=num_in_filters,
                    out_channels=num_out_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding = 'same',
                    padding_mode = 'reflect',
                    bias=False
                    )
                # self.act = torch.nn.ELU()
                # self.act = torch.nn.LeakyReLU(0.1)
                self.act = torch.nn.LeakyReLU(0.2)

            def forward(self,x):
                x = self.conv1(x)
                if self.activation == 'relu':
                    return self.act(x)
                else:
                    return x

        class Multiresblock(Module):
            '''
            MultiRes Block
            
            Arguments:
                num_in_channels {int} -- Number of channels coming into mutlires block
                num_filters {int} -- Number of filters in a corrsponding UNet stage
                alpha {float} -- alpha hyperparameter (default: 1.67)
            
            '''

            def __init__(self, num_in_channels, num_filters, alpha=1.69):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_3x3 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                filt_cnt_7x7 = int(self.W*0.5)
                filt_cnt_9x9 = int(self.W*0.69)
                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7 # + filt_cnt_9x9

                self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

                self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

                self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
                
                self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

                # self.conv_9x9 = Conv2d_batchnorm(filt_cnt_7x7, filt_cnt_9x9, kernel_size = (3,3), activation='relu')

                # self.act = torch.nn.ELU()
                # self.act = torch.nn.LeakyReLU(0.1)
                self.act = torch.nn.LeakyReLU(0.2)


            def forward(self,x):

                shrtct = self.shortcut(x)
                
                a = self.conv_3x3(x)
                b = self.conv_5x5(a)
                c = self.conv_7x7(b)
                # d = self.conv_9x9(c)

                # x = torch.cat([a,b,c,d],axis=1)
                x = torch.cat([a,b,c],axis=1)
                x = x + shrtct
                x = self.act(x)
            
                return x


        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, feat=False):
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
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=False)
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
                self.alpha = 1.69
                self.upsample01 = torch.nn.ConvTranspose2d(c, c, 4, 2, 1)
                self.multires01 = Multiresblock(c, c*2)
                self.multires01_filters = int(c*2*self.alpha*0.167)+int(c*2*self.alpha*0.333)+int(c*2*self.alpha* 0.5)
                self.upsample02 = torch.nn.ConvTranspose2d(self.multires01_filters, c//2, 4, 2, 1)
                self.multires02 = Multiresblock(c//2, c)
                self.multires02_filters = int(c*self.alpha*0.167)+int(c*self.alpha*0.333)+int(c*self.alpha* 0.5)
                self.conv_final = Conv2d_batchnorm(self.multires02_filters, 8, kernel_size = (3,3), activation='None')

            def forward(self, x, flow, scale=1):
                x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                if flow is not None:
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)
                feat = self.conv0(x)
                feat = self.convblock(feat)

                print (f'feat shape: {feat.shape}')
            
                tmp = self.lastconv(feat)

                print (f'tmp shape: {tmp.shape}')

                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)

                print (f'tmp scaled shape: {tmp.shape}')

                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]

                # additional lastconv block
                up01 = self.upsample01(feat)
                print (f'feat shape: {feat.shape}')
                print (f'up01 shape: {up01.shape}')

                x01 = self.multires01(up01)
                up02 = self.upsample02(x01)

                print (f'up02 shape: {up02.shape}')

                x02 = self.multires02(up02)
                up03 = self.conv_final(x02)

                print (f'up03 shape: {up03.shape}')

                # up03 = torch.nn.functional.interpolate(up03, scale_factor=scale, mode="bilinear", align_corners=False)

                addflow = up03[:, :4] * scale
                addmask = up03[:, 4:5]
                flowmix01 = torch.sigmoid(up03[:, 5:6])
                flowmix02 = torch.sigmoid(up03[:, 6:7])
                maskmix = torch.sigmoid(up03[:, 7:8])

                flow0 = flow[:, :2] * flowmix01 + addflow[:, :2] * (1 - flowmix01)
                flow1 = flow[:, 2:4] * flowmix02 + addflow[:, 2:4] * (1 - flowmix02)
                mask = mask * maskmix + addmask * (1 - maskmix)

                flow = torch.cat((flow0, flow1), dim=1)
                # end of additional lastconv block

                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(7+16, c=192)
                self.block1 = Flownet(8+4+16, c=128)
                self.block2 = Flownet(8+4+16, c=96)
                self.block3 = Flownet(8+4+16, c=64)
                self.encode = Head()

            def forward(self, img0, gt, img1, f0, f1, timestep=0.5, scale=[8, 4, 2, 1]):
                # return self.encode(img0)
                img0 = img0
                img1 = img1
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                if not torch.is_tensor(timestep):
                    timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                else:
                    timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
                flow_list = []
                merged = []
                mask_list = []
                conf_list = []
                teacher_list = []
                flow_list_teacher = []
                warped_img0 = img0
                warped_img1 = img1
                flow = None
                loss_cons = 0
                stu = [self.block0, self.block1, self.block2, self.block3]
                flow = None
                for i in range(4):
                    if flow is not None:
                        flow_d, mask, conf = stu[i](torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1), flow, scale=scale[i])
                        flow = flow + flow_d
                    else:
                        flow, mask, conf = stu[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=scale[i])

                    mask_list.append(mask)
                    flow_list.append(flow)
                    conf_list.append(conf)
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    merged_student = (warped_img0, warped_img1)
                    merged.append(merged_student)
                conf = torch.sigmoid(torch.cat(conf_list, 1))
                conf = conf / (conf.sum(1, True) + 1e-3)
                if gt is not None:
                    flow_teacher = 0
                    mask_teacher = 0
                    for i in range(4):
                        flow_teacher += conf[:, i:i+1] * flow_list[i]
                        mask_teacher += conf[:, i:i+1] * mask_list[i]
                    warped_img0_teacher = warp(img0, flow_teacher[:, :2])
                    warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
                    mask_teacher = torch.sigmoid(mask_teacher)
                    merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
                    teacher_list.append(merged_teacher)
                    flow_list_teacher.append(flow_teacher)

                for i in range(4):
                    mask_list[i] = torch.sigmoid(mask_list[i])
                    merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
                    if gt is not None:
                        loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 1e-2).float().detach()
                        loss_cons += (((flow_teacher.detach() - flow_list[i]) ** 2).sum(1, True) ** 0.5 * loss_mask).mean() * 0.001

                return flow_list, mask_list, merged

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
try:
    import torch
    from torch.nn import Module
    import torch.nn.functional as F
except:
    torch = object
    Module = object
    F = object

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class unetConv2(Module):
    def __init__(self, in_size, out_size, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        for i in range(1, n + 1):
            conv = torch.nn.Sequential(torch.nn.Conv2d(in_size, out_size, ks, s, p, padding_mode = 'reflect'),
                                    torch.nn.SELU(inplace=True), )
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

class DoubleConv2(Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
			in_channels=in_size,
			out_channels=out_size,
			kernel_size=(3,3),
			stride=1,
			padding = 'same',
			padding_mode = 'reflect',
			bias=False
			)

        self.conv2 = torch.nn.Conv2d(
			in_channels=out_size,
			out_channels=out_size,
			kernel_size=(3,3),
			stride=1,
			padding = 'same',
			padding_mode = 'reflect',
			bias=False
			)
        
        self.act = torch.nn.SELU(inplace=True)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x

class Conv2d(Module):
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1)):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(
			in_channels=num_in_filters,
			out_channels=num_out_filters,
			kernel_size=kernel_size,
			stride=stride,
			padding = 'same',
			padding_mode = 'reflect',
			bias=False
			)
		# torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='selu')
		torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
		# torch.nn.init.dirac_(self.conv1.weight)

	def forward(self,x):
		x = self.conv1(x)
		return x

class Conv2d_ReLU(Module):
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1)):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(
			in_channels=num_in_filters,
			out_channels=num_out_filters,
			kernel_size=kernel_size,
			stride=stride,
			padding = 'same',
			padding_mode = 'reflect',
			bias=False
			)
		self.act = torch.nn.SELU(inplace = True)
		# torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='selu')
		torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
		# torch.nn.init.dirac_(self.conv1.weight)

	def forward(self,x):
		x = self.conv1(x)
		x = self.act(x)
		return x


'''
    UNet 3+
'''
class UNet_3Plus(Module):

    def __init__(self, in_channels=3, n_classes=3, feature_scale=4):
        super(UNet_3Plus, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0])
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1])
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2])
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3])
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4])

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = torch.nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = torch.nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd4_relu = torch.nn.SELU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = torch.nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = torch.nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_PT_hd4_relu = torch.nn.SELU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = torch.nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h3_PT_hd4_relu = torch.nn.SELU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = torch.nn.Conv2d(filters[3], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h4_Cat_hd4_relu = torch.nn.SELU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = torch.nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd4_relu = torch.nn.SELU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.relu4d_1 = torch.nn.SELU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = torch.nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = torch.nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd3_relu = torch.nn.SELU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = torch.nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_PT_hd3_relu = torch.nn.SELU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = torch.nn.Conv2d(filters[2], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h3_Cat_hd3_relu = torch.nn.SELU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd3_relu = torch.nn.SELU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = torch.nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd3_relu = torch.nn.SELU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.relu3d_1 = torch.nn.SELU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = torch.nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_PT_hd2_relu = torch.nn.SELU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = torch.nn.Conv2d(filters[1], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h2_Cat_hd2_relu = torch.nn.SELU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd3_UT_hd2_relu = torch.nn.SELU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd2_relu = torch.nn.SELU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = torch.nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = torch.nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd2_relu = torch.nn.SELU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.relu2d_1 = torch.nn.SELU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = torch.nn.Conv2d(filters[0], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.h1_Cat_hd1_relu = torch.nn.SELU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd2_UT_hd1_relu = torch.nn.SELU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd3_UT_hd1_relu = torch.nn.SELU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = torch.nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = torch.nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd4_UT_hd1_relu = torch.nn.SELU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = torch.nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = torch.nn.Conv2d(filters[4], self.CatChannels, 3, padding=1, padding_mode = 'reflect')
        self.hd5_UT_hd1_relu = torch.nn.SELU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = torch.nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1, padding_mode = 'reflect')  # 16
        self.relu1d_1 = torch.nn.SELU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = torch.nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = torch.nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = torch.nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = torch.nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = torch.nn.Conv2d(filters[4], n_classes, 3, padding=1)

        # output
        self.outconv1 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1, padding_mode = 'reflect')

        # initialise weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, torch.nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_conv(h4))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        hd4 = self.relu4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_conv(h3))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        hd3 = self.relu3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_conv(h2))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        hd2 = self.relu2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_conv(h1))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        hd1 = self.relu1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))) # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        out = self.outconv1(hd1)  # d1->320*320*n_classes

        '''
        _, _, h, w = inputs.shape
        norm_flow0 = torch.tanh(out[:, :2])
        res_mask = torch.sigmoid((out[: , 2:3] + 1) / 2)
        norm_flow1 = torch.tanh(out[:, 3:5])

        horisontal_flow0 = norm_flow0[:, 0:1, :, :] * w # Horizontal component
        vertical_flow0 = norm_flow0[:, 1:2, :, :] * h # Vertical component
        res_flow0 = torch.cat([horisontal_flow0, vertical_flow0], dim=1)
        
        horisontal_flow1 = norm_flow1[:, 0:1, :, :] * w # Horizontal component
        vertical_flow1 = norm_flow1[:, 1:2, :, :] * h # Vertical component
        res_flow1 = torch.cat([horisontal_flow1, vertical_flow1], dim=1)
        '''

        deepsup = []

        # res_flow0_d5 = torch.tanh(d5[:, :2])
        # mask_d5 = torch.sigmoid((d5[: , 2:3]+ 1) / 2)
        # res_flow1_d5 = torch.tanh(d5[:, 3:5])
        deepsup.append((d5))

        # res_flow0_d4 = torch.tanh(d4[:, :2])
        # mask_d4 = torch.sigmoid((d4[: , 2:3]+ 1) / 2)
        # res_flow1_d4 = torch.tanh(d4[:, 3:5])
        deepsup.append((d4))

        # res_flow0_d3 = torch.tanh(d3[:, :2])
        # mask_d3 = torch.sigmoid((d3[: , 2:3]+ 1) / 2)
        # res_flow1_d3 = torch.tanh(d3[:, 3:5])
        deepsup.append((d3))

        # res_flow0_d2 = torch.tanh(d2[:, :2])
        # mask_d2 = torch.sigmoid((d2[: , 2:3]+ 1) / 2)
         #res_flow1_d2 = torch.tanh(d2[:, 3:5])
        deepsup.append((d2))

        # res_flow0 = torch.tanh(out[:, :2])
        # mask = out[: , 2:3]
         #res_mask = torch.sigmoid((mask + 1) / 2)
        # res_flow1 = torch.tanh(out[:, 3:5])

        return out, deepsup

class Model:
    @staticmethod
    def get_name():
        return 'UNet_3Plus_v002'

    def get_model(self):
        return UNet_3Plus

    def get_training_model(self):
        return UNet_3Plus

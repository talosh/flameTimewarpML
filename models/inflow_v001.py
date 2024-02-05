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

        # output
        self.outconv1 = torch.nn.Conv2d(self.UpChannels, n_classes, 3, padding=1, padding_mode = 'reflect')
        self.outact = torch.nn.tanh()

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

        out = self.outconv1(hd1)  # d1->320*320*n_classes
        out = self.outact(out)

        n, c, h, w = inputs.shape
        res_flow0 = out[:, :2]
        res_mask = (out[: , 2:3] + 1) / 2
        res_flow1 = out[:, 3:5]

        res_flow0[:, 0, :, :] *= (w - 1) / 2.0  # Horizontal component
        res_flow0[:, 1, :, :] *= (h - 1) / 2.0  # Vertical component
        
        res_flow1[:, 0, :, :] *= (w - 1) / 2.0  # Horizontal component
        res_flow1[:, 1, :, :] *= (h - 1) / 2.0  # Vertical component

        return res_flow0, res_flow1, res_mask

class Model:
    @staticmethod
    def get_name():
        return 'UNet_3Plus_v001'

    def get_model(self):
        return UNet_3Plus

    def get_training_model(self):
        return UNet_3Plus

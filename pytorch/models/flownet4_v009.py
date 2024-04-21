class Model:
	def __init__(self, status = dict(), torch = None):
		if torch is None:
			import torch
		Module = torch.nn.Module

		def channel_shuffle(x, groups=2):
			batch_size, channels, w, h = x.shape
			div_channels = channels - (channels % groups)

			x_main = x[:, :div_channels, :, :]

			mbatch_size, mchannels, mw, mh = x_main.shape
			group_c = mchannels // groups

			if div_channels < channels:
				x_rem = x[:, div_channels:, :, :]

			x_main = x_main.view(mbatch_size, groups, group_c, mw, mh)
			x_main = torch.transpose(x_main, 1, 2).contiguous()
			x_main = x_main.view(mbatch_size, -1, mw, mh)

			if div_channels < channels:
				x = torch.cat((x_main, x_rem), dim=1)
			else:
				x = x_main

			return x

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
				self.act = torch.nn.LeakyReLU(0.1)
				# self.act = torch.nn.LeakyReLU(0.2)

				
			
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

				# self.beta = torch.nn.Parameter(torch.ones((1, num_out_filters, 1, 1)), requires_grad=True)

				# self.act = torch.nn.ELU()
				self.act = torch.nn.LeakyReLU(0.1)
				# self.act = torch.nn.LeakyReLU(0.2)


			def forward(self,x):

				shrtct = self.shortcut(x)
				
				a = self.conv_3x3(x)
				b = self.conv_5x5(a)
				c = self.conv_7x7(b)
				# d = self.conv_9x9(c)

				# x = torch.cat([a,b,c,d],axis=1)
				x = torch.cat([a,b,c],axis=1)

				# x = x * self.beta + shrtct
				x = x + shrtct
				x = self.act(x)
			
				return x

		class MultiresblockFlow(Module):
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
				
				filt_cnt_3x3 = num_in_channels # int(self.W*0.167)
				filt_cnt_5x5 = int(self.W*0.333)
				filt_cnt_7x7 = int(self.W*0.5)
				# filt_cnt_9x9 = int(self.W*0.69)
				num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7 # + filt_cnt_9x9

				self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

				self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (5,5), activation='relu')

				self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (5,5), activation='relu')
				
				self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (5,5), activation='relu')

				# self.conv_9x9 = Conv2d_batchnorm(filt_cnt_7x7, filt_cnt_9x9, kernel_size = (5,5), activation='relu')

				# self.beta = torch.nn.Parameter(torch.ones((1, num_out_filters, 1, 1)), requires_grad=True)

				# self.act = torch.nn.ELU()
				self.act = torch.nn.LeakyReLU(0.1)
				# self.act = torch.nn.LeakyReLU(0.2)


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
			
				return channel_shuffle(x)

		class MultiresblockDeep(Module):
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
				num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7 + filt_cnt_9x9

				self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

				self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

				self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
				
				self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

				self.conv_9x9 = Conv2d_batchnorm(filt_cnt_7x7, filt_cnt_9x9, kernel_size = (3,3), activation='relu')

				# self.beta = torch.nn.Parameter(torch.ones((1, num_out_filters, 1, 1)), requires_grad=True)
				self.beta = Conv2d_batchnorm(num_out_filters ,num_out_filters , kernel_size = (1,1), activation='None')

				# self.act = torch.nn.ELU()
				self.act = torch.nn.LeakyReLU(0.1)
				# self.act = torch.nn.LeakyReLU(0.2)


			def forward(self,x):

				shrtct = self.shortcut(x)
				
				a = self.conv_3x3(x)
				b = self.conv_5x5(a)
				c = self.conv_7x7(b)
				d = self.conv_9x9(c)

				x = torch.cat([a,b,c,d],axis=1)
				# x = torch.cat([a,b,c],axis=1)

				x = x * self.beta(x) + shrtct
				x = self.act(x)
			
				return channel_shuffle(x)


		class Respath(Module):
			'''
			ResPath
			
			Arguments:
				num_in_filters {int} -- Number of filters going in the respath
				num_out_filters {int} -- Number of filters going out the respath
				respath_length {int} -- length of ResPath
				
			'''

			def __init__(self, num_in_filters, num_out_filters, respath_length):
			
				super().__init__()

				self.respath_length = respath_length
				self.shortcuts = torch.nn.ModuleList([])
				self.convs = torch.nn.ModuleList([])
				self.bns = torch.nn.ModuleList([])
				# self.act = torch.nn.ELU(inplace=True)
				self.act = torch.nn.LeakyReLU(0.1)


				for i in range(self.respath_length):
					if(i==0):
						self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation='None'))
						self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation='relu'))

						
					else:
						self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation='None'))
						self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation='relu'))
				
			
			def forward(self,x):

				for i in range(self.respath_length):

					shortcut = self.shortcuts[i](x)

					x = self.convs[i](x)
					x = self.act(x)

					x = x + shortcut
					x = self.act(x)

				return x


		class MultiResUnet(Module):
			'''
			MultiResUNet
			
			Arguments:
				input_channels {int} -- number of channels in image
				num_classes {int} -- number of segmentation classes
				alpha {float} -- alpha hyperparameter (default: 1.67)
			
			Returns:
				[keras model] -- MultiResUNet model
			'''
			def __init__(self, alpha=1.69):
				super().__init__()
			
				self.alpha = alpha
				enc_channels = 3
				enc_classes = 8
				input_channels = 5
				num_classes = 4

				complx_enc = 11 # 27
				complx_flowenc = 24
				complx_chain = 32
				complx_main = 24 # 54

				# Image encoder
				# self.in_filters1_enc = int(complx_enc*self.alpha*0.167)+int(complx_enc*self.alpha*0.333)+int(complx_enc*self.alpha*0.5) #+int(complx_enc*self.alpha*0.69)
				# self.in_filters2_enc = int(complx_enc*2*self.alpha*0.167)+int(complx_enc*2*self.alpha*0.333)+int(complx_enc*2*self.alpha*0.5) #+int(2*complx_enc*self.alpha*0.69)
				# self.in_filters3_enc = int(complx_enc*4*self.alpha*0.167)+int(complx_enc*4*self.alpha*0.333)+int(complx_enc*4*self.alpha*0.5) #+int(2*complx_enc*self.alpha*0.69)

				'''
				self.multiresblock1_enc = torch.nn.Sequential(
					Multiresblock(enc_channels,complx_enc),
					Multiresblock(self.in_filters1_enc,2*complx_enc),
					Multiresblock(self.in_filters2_enc, 4*complx_enc),
					Conv2d_batchnorm(self.in_filters3_enc, enc_classes, kernel_size = (3,3), activation='relu')
				)
				'''
				
				# Encoder Path
				self.multiresblock1 = Multiresblock(input_channels,complx_main)
				self.in_filters1 = int(complx_main*self.alpha*0.167)+int(complx_main*self.alpha*0.333)+int(complx_main*self.alpha*0.5)#+int(complx_main*self.alpha*0.69)
				self.pool1 =  torch.nn.AvgPool2d(4)
				self.multiresblock1_enc = Multiresblock(enc_channels,complx_enc)
				self.enc_filters1 = int(complx_enc*self.alpha*0.167)+int(complx_enc*self.alpha*0.333)+int(complx_enc*self.alpha*0.5)#+int(complx_main*self.alpha*0.69)

				combined_filters = 2*self.enc_filters1 + 6 + 4 + 1 + 1

				self.multiresblock1_flowenc = MultiresblockFlow(combined_filters, complx_flowenc)
				# self.in_filters1_flowenc = int(complx_flowenc*self.alpha*0.167)+int(complx_flowenc*self.alpha*0.333)+int(complx_flowenc*self.alpha*0.5)+int(complx_flowenc*self.alpha*0.69)
				self.in_filters1_flowenc = combined_filters+int(complx_flowenc*self.alpha*0.333)+int(complx_flowenc*self.alpha*0.5) # +int(complx_flowenc*self.alpha*0.69)
				self.respath1 = Respath(self.in_filters1_flowenc,complx_main,respath_length=4)

				self.multiresblock2 = Multiresblock(self.in_filters1_flowenc,complx_main*2)
				self.in_filters2 = int(complx_main*2*self.alpha*0.167)+int(complx_main*2*self.alpha*0.333)+int(complx_main*2*self.alpha* 0.5)#+int(complx_main*2*self.alpha*0.69)
				self.pool2 =  torch.nn.AvgPool2d(2)
				self.multiresblock2_enc = Multiresblock(self.enc_filters1,complx_enc*2)
				self.enc_filters2 = int(complx_enc*2*self.alpha*0.167)+int(complx_enc*2*self.alpha*0.333)+int(complx_enc*2*self.alpha*0.5)#+int(complx_main*self.alpha*0.69)
				
				combined_filters2 = 2*self.enc_filters2 + self.in_filters1_flowenc

				self.multiresblock2_flowenc = MultiresblockFlow(combined_filters2, complx_flowenc*2)
				self.in_filters2_flowenc = combined_filters2+int(complx_flowenc*2*self.alpha*0.333)+int(complx_flowenc*2*self.alpha*0.5)# +int(complx_flowenc*2*self.alpha*0.69)
				self.respath2 = Respath(self.in_filters2_flowenc,complx_main*2,respath_length=3)
			
				self.multiresblock3 =  Multiresblock(self.in_filters2_flowenc,complx_main*4)
				self.in_filters3 = int(complx_main*4*self.alpha*0.167)+int(complx_main*4*self.alpha*0.333)+int(complx_main*4*self.alpha* 0.5)#+int(complx_main*4*self.alpha*0.69)
				self.pool3 =  torch.nn.AvgPool2d(2)
				self.multiresblock3_enc = Multiresblock(self.enc_filters2,complx_enc*4)
				self.enc_filters3 = int(complx_enc*4*self.alpha*0.167)+int(complx_enc*4*self.alpha*0.333)+int(complx_enc*4*self.alpha*0.5)#+int(complx_main*self.alpha*0.69)

				combined_filters3 = 2*self.enc_filters3 + self.in_filters2_flowenc

				self.multiresblock3_flowenc = MultiresblockFlow(combined_filters3, complx_flowenc*4)
				self.in_filters3_flowenc = combined_filters3+int(complx_flowenc*4*self.alpha*0.333)+int(complx_flowenc*4*self.alpha*0.5)# +int(complx_flowenc*4*self.alpha*0.69)			
				self.respath3 = Respath(self.in_filters3_flowenc,complx_main*4,respath_length=2)
			
			
				self.multiresblock4 = Multiresblock(self.in_filters3_flowenc,complx_main*8)
				self.in_filters4 = int(complx_main*8*self.alpha*0.167)+int(complx_main*8*self.alpha*0.333)+int(complx_main*8*self.alpha* 0.5)#+int(complx_main*8*self.alpha*0.69)
				self.pool4 =  torch.nn.AvgPool2d(2)
				self.respath4 = Respath(self.in_filters4,complx_main*8,respath_length=4)
			
			
				self.multiresblock5 = Multiresblock(self.in_filters4,complx_main*16)
				self.in_filters5 = int(complx_main*16*self.alpha*0.167)+int(complx_main*16*self.alpha*0.333)+int(complx_main*16*self.alpha* 0.5)#+int(complx_main*16*self.alpha*0.69)
				self.in_filters5_chain = int(complx_chain*16*self.alpha*0.167)+int(complx_chain*16*self.alpha*0.333)+int(complx_chain*16*self.alpha* 0.5)+int(complx_chain*16*self.alpha*0.69)
				self.multiresblock5_chain = self.convblock = torch.nn.Sequential(
					MultiresblockDeep(self.in_filters5,complx_chain*16),
					MultiresblockDeep(self.in_filters5_chain,complx_chain*16),
					MultiresblockDeep(self.in_filters5_chain,complx_chain*16),
					MultiresblockDeep(self.in_filters5_chain,complx_chain*16),
				)
				self.deepsup5 = Conv2d_batchnorm(self.in_filters5_chain, num_classes, kernel_size = (3,3), activation='relu')
				self.dsupsample5 = torch.nn.Upsample(scale_factor = 32, mode='bilinear')
			
				# Decoder path
				self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5_chain,complx_main*8,kernel_size=(2,2),stride=(2,2))  
				self.concat_filters1 = complx_main*8*2
				self.multiresblock6 = Multiresblock(self.concat_filters1,complx_main*8)
				self.in_filters6 = int(complx_main*8*self.alpha*0.167)+int(complx_main*8*self.alpha*0.333)+int(complx_main*8*self.alpha* 0.5)#+int(complx_main*8*self.alpha*0.69)
				self.deepsup4 = Conv2d_batchnorm(self.in_filters6, num_classes, kernel_size = (3,3), activation='relu')
				self.dsupsample4 = torch.nn.Upsample(scale_factor = 16, mode='bilinear')


				self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,complx_main*4,kernel_size=(2,2),stride=(2,2))  
				self.concat_filters2 = complx_main*4*2
				self.multiresblock7 = Multiresblock(self.concat_filters2,complx_main*4)
				self.in_filters7 = int(complx_main*4*self.alpha*0.167)+int(complx_main*4*self.alpha*0.333)+int(complx_main*4*self.alpha* 0.5)#+int(complx_main*4*self.alpha*0.69)
				self.deepsup3 = Conv2d_batchnorm(self.in_filters7, num_classes, kernel_size = (3,3), activation='relu')
				self.dsupsample3 = torch.nn.Upsample(scale_factor = 8, mode='bilinear')

				self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,complx_main*2,kernel_size=(2,2),stride=(2,2))
				self.concat_filters3 = complx_main*2 *2
				self.multiresblock8 = Multiresblock(self.concat_filters3,complx_main*2)
				self.in_filters8 = int(complx_main*2*self.alpha*0.167)+int(complx_main*2*self.alpha*0.333)+int(complx_main*2*self.alpha* 0.5)#+int(complx_main*2*self.alpha*0.69)
				self.deepsup2 = Conv2d_batchnorm(self.in_filters8, num_classes, kernel_size = (3,3), activation='relu')
				self.dsupsample2 = torch.nn.Upsample(scale_factor = 4, mode='bilinear')

				self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,complx_main,kernel_size=(4,4),stride=(4,4))
				self.concat_filters4 = complx_main *2
				self.multiresblock9 = Multiresblock(self.concat_filters4,complx_main)
				self.in_filters9 = int(complx_main*self.alpha*0.167)+int(complx_main*self.alpha*0.333)+int(complx_main*self.alpha* 0.5)#+int(complx_main*self.alpha*0.69)

				self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes, kernel_size = (5,5), activation='None')

			def tenflow(self, tenFlow):
				tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0) ], 1)
				return tenFlow

			def warp_tenflow(self, tenInput, tenFlow):
				backwarp_tenGrid = {}
				k = (str(tenFlow.device), str(tenFlow.size()))
				if k not in backwarp_tenGrid:
					tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
					tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
					backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)
				g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
				return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

			def id_flow(self, tenInput):
				tenHorizontal = torch.linspace(-1.0, 1.0, tenInput.shape[3]).view(1, 1, 1, tenInput.shape[3]).expand(tenInput.shape[0], -1, tenInput.shape[2], -1)
				tenVertical = torch.linspace(-1.0, 1.0, tenInput.shape[2]).view(1, 1, tenInput.shape[2], 1).expand(tenInput.shape[0], -1, -1, tenInput.shape[3])
				return torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)

			def normalize(self, image_array) :
				def custom_bend(x):
					linear_part = x
					exp_bend = torch.sign(x) * torch.pow(torch.abs(x), 1 / 4 )
					return torch.where(x > 1, exp_bend, torch.where(x < -1, exp_bend, linear_part))

				# transfer (0.0 - 1.0) onto (-1.0 - 1.0) for tanh
				image_array = (image_array * 2) - 1
				# bend values below -1.0 and above 1.0 exponentially so they are not larger then (-4.0 - 4.0)
				image_array = custom_bend(image_array)
				# bend everything to fit -1.0 - 1.0 with hyperbolic tanhent
				image_array = torch.tanh(image_array)
				# move it to 0.0 - 1.0 range
				image_array = (image_array + 1) / 2

				return image_array
			
			def forward(self, img0, img1, flow0, flow1, timestep, mask):

				img0 = img0 * 2 - 1
				img1 = img1 * 2 - 1
				mask = mask * 2 - 1
				# x = torch.cat((flow0, timestep, flow1),dim=1)
				# x_multires1 = self.multiresblock1(x)

				img0_enc = self.multiresblock1_enc(img0)
				img1_enc = self.multiresblock1_enc(img1)
				id_flow = self.id_flow(img0)

				# print (f'x multires1 shape: {x_multires1.shape}')
				x_multires1 = torch.cat((
						img0,
						img1,
						# flow0,
						# flow1,
						flow0 + id_flow,
						flow1 + id_flow,
						self.warp_tenflow(img0_enc, flow0),
						self.warp_tenflow(img1_enc, flow1),
						# img0_renc,
						# id_flow,
						# self.warp_tenflow(img0, flow0),
						mask,
						timestep,
						# self.warp_tenflow(img1, flow1),
						# img1_renc,
						# id_flow,
					), dim=1)
				
				x_multires1 = self.multiresblock1_flowenc(x_multires1)
				x_pool1 = self.pool1(x_multires1)
				x_multires1 = self.respath1(x_multires1)
				
				img0_enc_pool1 = self.pool1(img0_enc)
				img1_enc_pool1 = self.pool1(img1_enc)

				img0_enc2 = self.multiresblock2_enc(img0_enc_pool1)
				img1_enc2 = self.multiresblock2_enc(img1_enc_pool1)

				# flow0 = torch.nn.functional.interpolate(flow0, scale_factor= 1. / 2, mode="bilinear", align_corners=False) * 1. / 2
				# flow1 = torch.nn.functional.interpolate(flow1, scale_factor= 1. / 2, mode="bilinear", align_corners=False) * 1. / 2

				x_pool1 = torch.cat((
					img0_enc2,
					# self.warp_tenflow(img0_enc2, flow0),
					x_pool1,
					# self.warp_tenflow(img1_enc2, flow0),
					img1_enc2
				), dim=1)

				x_multires2 = self.multiresblock2_flowenc(x_pool1)
				x_pool2 = self.pool2(x_multires2)
				x_multires2 = self.respath2(x_multires2)

				img0_enc_pool2 = self.pool2(img0_enc2)
				img1_enc_pool2 = self.pool2(img1_enc2)

				img0_enc3 = self.multiresblock3_enc(img0_enc_pool2)
				img1_enc3 = self.multiresblock3_enc(img1_enc_pool2)

				# flow0 = torch.nn.functional.interpolate(flow0, scale_factor= 1. / 2, mode="bilinear", align_corners=False) * 1. / 2
				# flow1 = torch.nn.functional.interpolate(flow1, scale_factor= 1. / 2, mode="bilinear", align_corners=False) * 1. / 2

				x_pool2 = torch.cat((
					img0_enc3,
					# self.warp_tenflow(img0_enc3, flow0),
					x_pool2,
					# self.warp_tenflow(img1_enc3, flow0),
					img1_enc3
				), dim=1)

				x_multires3 = self.multiresblock3_flowenc(x_pool2)
				x_pool3 = self.pool3(x_multires3)
				x_multires3 = self.respath3(x_multires3)

				x_multires4 = self.multiresblock4(x_pool3)
				x_pool4 = self.pool4(x_multires4)
				# print (f'\nxpool4 shape {x_pool4.shape}')
				x_multires4 = self.respath4(x_multires4)

				x_multires5 = self.multiresblock5(x_pool4)
				# print (f'x_multires5 shape {x_multires5.shape}')

				x_multires5 = self.multiresblock5_chain(x_multires5)

				ds5 = self.deepsup5(x_multires5)
				ds5 = self.dsupsample5(ds5)

				up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
				x_multires6 = self.multiresblock6(up6)

				ds4 = self.deepsup4(x_multires6)
				ds4 = self.dsupsample4(ds4)

				up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
				x_multires7 = self.multiresblock7(up7)

				ds3 = self.deepsup3(x_multires7)
				ds3 = self.dsupsample3(ds3)

				up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
				x_multires8 = self.multiresblock8(up8)

				ds2 = self.deepsup2(x_multires8)
				ds2 = self.dsupsample2(ds2)

				up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
				x_multires9 = self.multiresblock9(up9)
				'''
				del x_multires8
				del x_multires1
				del up9
				'''

				out_fusion =  self.conv_final(x_multires9)

				# res_flow0 = out_fusion[:, :2]
				# res_flow1 = out_fusion[:, 2:4]
				# res_flow0 = torch.tanh(out_fusion[:, :2])
				# res_flow1 = torch.tanh(out_fusion[:, 2:4])
				res = self.normalize(out_fusion[:, :3]) * 2 - 1
				res_mask = torch.sigmoid(out_fusion[:, 3:4])
				# res_flow0 = flow0 + (torch.sigmoid(out_fusion[:, 4:6]) * 2 - 1)
				# res_flow1 = flow1 + (torch.sigmoid(out_fusion[:, 6:8]) * 2 - 1)


				'''
				ds = []

				# ds5_flow0 = ds5[:, :2]
				# ds5_flow1 = ds5[:, 2:4]
				ds5_flow0 = torch.tanh(ds5[:, :2])
				ds5_flow1 = torch.tanh(ds5[:, 2:4])
				ds5_mask = torch.sigmoid((ds5[:, 4:5] + 1) / 2)
				ds.append((ds5_flow0, ds5_flow1, ds5_mask))


				# ds4_flow0 = ds4[:, :2]
				# ds4_flow1 = ds4[:, 2:4]
				ds4_flow0 = torch.tanh(ds4[:, :2])
				ds4_flow1 = torch.tanh(ds4[:, 2:4])
				ds4_mask = torch.sigmoid((ds4[:, 4:5] + 1) / 2)
				ds.append((ds4_flow0, ds4_flow1, ds4_mask))

				# ds3_flow0 = ds3[:, :2]
				# ds3_flow1 = ds3[:, 2:4]
				ds3_flow0 = torch.tanh(ds3[:, :2])
				ds3_flow1 = torch.tanh(ds3[:, 2:4])
				ds3_mask = torch.sigmoid((ds3[:, 4:5] + 1) / 2)
				ds.append((ds3_flow0, ds3_flow1, ds3_mask))


				# ds2_flow0 = ds2[:, :2]
				# ds2_flow1 = ds2[:, 2:4]
				ds2_flow0 = torch.tanh(ds2[:, :2])
				ds2_flow1 = torch.tanh(ds2[:, 2:4])
				ds2_mask = torch.sigmoid((ds2[:, 4:5] + 1) / 2)
				ds.append((ds2_flow0, ds2_flow1, ds2_mask))

				return res_flow0, res_flow1, res_mask, ds

				# print (f'\nmax: {torch.max(out):.4f}')
				# print (f'min: {torch.min(out):.4f}')
				
				return out
				'''

				return res, res_mask #, res_flow0, res_flow1


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
			backwarp_tenGrid = {}

			k = (str(tenFlow.device), str(tenFlow.size()))
			if k not in backwarp_tenGrid:
				tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
				tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
				backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device)
				# end

			tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

			g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
			return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

		# '''
		class Head(Module):
			def __init__(self):
				super(Head, self).__init__()
				alpha=1.69
				num_in_channels = 3
				num_filters = 32
				self.W = num_filters * alpha
				filt_cnt_3x3 = int(self.W*0.167)
				filt_cnt_5x5 = int(self.W*0.333)
				filt_cnt_7x7 = int(self.W*0.5)
				num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

				self.cnn0 = torch.nn.Conv2d(num_in_channels, filt_cnt_3x3, 3, 2, 1)
				self.cnn1 = torch.nn.Conv2d(filt_cnt_3x3, filt_cnt_3x3, 3, 1, 1)
				self.cnn2 = torch.nn.Conv2d(filt_cnt_3x3, filt_cnt_5x5, 3, 1, 1)
				self.cnn3 = torch.nn.Conv2d(filt_cnt_5x5, filt_cnt_7x7, 3, 1, 1)
				self.cnn4 = torch.nn.ConvTranspose2d(num_out_filters, 8, 4, 2, 1)
				self.shortcut = torch.nn.Conv2d(filt_cnt_3x3, num_out_filters, 1, 1)
				self.relu = torch.nn.LeakyReLU(inplace=True)

			def forward(self, x, feat=False):
				x = self.cnn0(x)
				x = self.relu(x)
				shrtct = self.shortcut(x)
				a = self.cnn1(x)
				a = self.relu(a)
				b = self.cnn2(a)
				b = self.relu(b)
				c = self.cnn3(b)
				c = self.relu(c)
				x = (torch.cat([a, b, c], dim=1) + shrtct)
				x = self.relu(x)
				x = self.cnn4(x)
				return x
		# '''

		'''
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
		'''

		class ResConv(Module):
			def __init__(self, c, dilation=1):
				super().__init__()
				self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect')
				self.conv1x = torch.nn.Conv2d(c, c, 1, 1)
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
				# self.encode = Head()

			def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1, encode=None):
				img0 = torch.nn.functional.interpolate(img0, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
				img1 = torch.nn.functional.interpolate(img1, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
				f0_loc = encode(img0)
				f1_loc = encode(img1)
				timestep = (img0[:, :1].clone() * 0 + 1) * timestep
				x = torch.cat((img0, img1, f0_loc, f1_loc, timestep), 1)
				if flow is not None:
					# f0 = torch.nn.functional.interpolate(f0, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
					# f1 = torch.nn.functional.interpolate(f1, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
					mask = torch.nn.functional.interpolate(mask, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
					flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
					x = torch.cat((img0, img1, f0_loc, f1_loc, timestep, mask, flow), 1)
				feat = self.conv0(x)
				feat = self.convblock(feat)
				tmp = self.lastconv(feat)
				tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
				flow = tmp[:, :4] * scale
				mask = tmp[:, 4:5]
				conf = tmp[:, 4:5]
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
				self.fusion = MultiResUnet()

			def forward(self, img0, gt, img1, f0, f1, timestep=0.5, scale=[8, 4, 2, 1]):
				# return self.encode(img0)
				img0 = img0
				img1 = img1
				
				f0 = None
				f1 = None
				
				# f0 = self.encode(img0)
				# f1 = self.encode(img1)

				'''
				if not torch.is_tensor(timestep):
					timestep = (img0[:, :1].clone() * 0 + 1) * timestep
				else:
					timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
				'''

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
						flow_d, mask, conf = stu[i](warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask, flow, scale=scale[i], encode=self.encode)
						flow = flow + flow_d
					else:
						flow, mask, conf = stu[i](img0, img1, f0, f1, timestep, None, None, scale=scale[i], encode=self.encode)

					mask_list.append(mask)
					flow_list.append(flow)
					conf_list.append(conf)
					warped_img0 = warp(img0, flow[:, :2])
					warped_img1 = warp(img1, flow[:, 2:4])
					warped_f0 = None
					warped_f1 = None
					# warped_f0 = warp(f0, flow[:, :2])
					# warped_f1 = warp(f1, flow[:, 2:4])
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

				flow0 = flow_list[3][:, :2]
				flow1 = flow_list[3][:, 2:4]
				timestep_tensor = (img0[:, :1].clone() * 0 + 1) * timestep
				mask_rife = mask_list[3]
				res, res_mask = self.fusion(
					img0,
					img1,
					self.fusion.tenflow(flow0),
					self.fusion.tenflow(flow1),
					timestep_tensor,
					mask_rife,
					)
				
				output_fusion_merged = self.fusion.warp_tenflow(img0, flow0) * res_mask + self.fusion.warp_tenflow(img1, flow1) * (1 - res_mask)
				output = output_fusion_merged + res

				mask_list[3] = res_mask
				merged[3] = output

				return flow_list, mask_list, merged

		self.model = FlownetCas
		self.training_model = FlownetCas

	@staticmethod
	def get_info():
		info = {
			'name': 'Flownet4_v009',
			'file': 'flownet4_v009.py',
			'ratio_support': True
		}
		return info

	@staticmethod
	def get_name():
		return 'TWML_Flownet_v009'
	
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
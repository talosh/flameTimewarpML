class Model:
	def __init__(self, status = dict(), torch = None):
		if torch is None:
			import torch
		Module = torch.nn.Module

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
				self.act = torch.nn.SELU()
			
			def forward(self,x):
				x = self.conv1(x)
				
				if self.activation == 'relu':
					return self.act(x)
				else:
					return x
			
			'''
			def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = 'relu'):
				super().__init__()
				layers = [
					torch.nn.Conv2d(
						in_channels=num_in_filters, 
						out_channels=num_out_filters, 
						kernel_size=kernel_size, 
						stride=stride, 
						padding = 'same',
						padding_mode = 'reflect'
					),
					torch.nn.BatchNorm2d(num_out_filters),
				]

				if activation == 'relu':
					# layers.insert(2, torch.nn.ELU(inplace=True))
					layers.append(torch.nn.SELU(inplace=True))
				
				self.layers = torch.nn.Sequential(*layers)
			
			def forward(self,x):
				return self.layers(x)
			'''

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
				num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7 + filt_cnt_9x9
				
				self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

				self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

				self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
				
				self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

				self.conv_9x9 = Conv2d_batchnorm(filt_cnt_7x7, filt_cnt_9x9, kernel_size = (3,3), activation='relu')

				self.act = torch.nn.SELU()

			def forward(self,x):

				shrtct = self.shortcut(x)
				
				a = self.conv_3x3(x)
				b = self.conv_5x5(a)
				c = self.conv_7x7(b)
				d = self.conv_9x9(c)

				x = torch.cat([a,b,c,d],axis=1)

				x = x + shrtct
				x = self.act(x)
			
				return x

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
				self.act = torch.nn.SELU(inplace=True)

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
				input_channels = 12
				output_channels = 5
				num_classes = 48
				img_classes = 11
				
				# Encoder Path
				self.multiresblock1 = Multiresblock(input_channels,num_classes)
				self.in_filters1 = int(num_classes*self.alpha*0.167)+int(num_classes*self.alpha*0.333)+int(num_classes*self.alpha*0.5)+int(num_classes*self.alpha*0.69)
				self.multiresblock1_img = Multiresblock(3,img_classes)
				self.in_img_filters1 = int(img_classes*self.alpha*0.167)+int(img_classes*self.alpha*0.333)+int(img_classes*self.alpha*0.5)+int(img_classes*self.alpha*0.69)
				self.pool1 =  torch.nn.MaxPool2d(2)
				self.respath1 = Respath(self.in_filters1 + 2 * self.in_img_filters1,num_classes,respath_length=4)

				self.multiresblock2 = Multiresblock(self.in_filters1 + 2 * self.in_img_filters1, num_classes*2)
				self.multiresblock2_img = Multiresblock(self.in_img_filters1,img_classes*2)
				self.in_img_filters2 = int(img_classes*2*self.alpha*0.167)+int(img_classes*2*self.alpha*0.333)+int(img_classes*2*self.alpha*0.5)+int(img_classes*2*self.alpha*0.69)
				self.in_filters2 = int(num_classes*2*self.alpha*0.167)+int(num_classes*2*self.alpha*0.333)+int(num_classes*2*self.alpha* 0.5)+int(num_classes*2*self.alpha*0.69)
				self.pool2 =  torch.nn.MaxPool2d(2)
				self.respath2 = Respath(self.in_filters2 + 2 * self.in_img_filters2,num_classes*2,respath_length=3)
			
				self.multiresblock3 =  Multiresblock(self.in_filters2,num_classes*4)
				self.multiresblock3_img = Multiresblock(self.in_img_filters2,img_classes*4)
				self.in_img_filters3 = int(img_classes*4*self.alpha*0.167)+int(img_classes*4*self.alpha*0.333)+int(img_classes*4*self.alpha*0.5)+int(img_classes*4*self.alpha*0.69)
				self.in_filters3 = int(num_classes*4*self.alpha*0.167)+int(num_classes*4*self.alpha*0.333)+int(num_classes*4*self.alpha* 0.5)+int(num_classes*4*self.alpha*0.69)
				self.pool3 =  torch.nn.MaxPool2d(2)
				self.respath3 = Respath(self.in_filters3 + 2 * self.in_img_filters3,num_classes*4,respath_length=2)
			
				self.multiresblock4 = Multiresblock(self.in_filters3,num_classes*8)
				self.multiresblock4_img = Multiresblock(self.in_img_filters3,img_classes*8)
				self.in_img_filters4 = int(img_classes*8*self.alpha*0.167)+int(img_classes*8*self.alpha*0.333)+int(img_classes*8*self.alpha*0.5)+int(img_classes*8*self.alpha*0.69)
				self.in_filters4 = int(num_classes*8*self.alpha*0.167)+int(num_classes*8*self.alpha*0.333)+int(num_classes*8*self.alpha* 0.5)+int(num_classes*8*self.alpha*0.69)
				self.pool4 =  torch.nn.MaxPool2d(2)
				self.respath4 = Respath(self.in_filters4 + 2 * self.in_img_filters4,num_classes*8,respath_length=1)
			
			
				self.multiresblock5 = Multiresblock(self.in_filters4 + 2 * self.in_img_filters4,num_classes*16)
				self.in_filters5 = int(num_classes*16*self.alpha*0.167)+int(num_classes*16*self.alpha*0.333)+int(num_classes*16*self.alpha* 0.5)+int(num_classes*16*self.alpha*0.69)
			
				# Decoder path
				self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,num_classes*8,kernel_size=(2,2),stride=(2,2))  
				self.concat_filters1 = num_classes*8*2
				self.multiresblock6 = Multiresblock(self.concat_filters1,num_classes*8)
				self.in_filters6 = int(num_classes*8*self.alpha*0.167)+int(num_classes*8*self.alpha*0.333)+int(num_classes*8*self.alpha* 0.5)+int(num_classes*8*self.alpha*0.69)

				self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,num_classes*4,kernel_size=(2,2),stride=(2,2))  
				self.concat_filters2 = num_classes*4*2
				self.multiresblock7 = Multiresblock(self.concat_filters2,num_classes*4)
				self.in_filters7 = int(num_classes*4*self.alpha*0.167)+int(num_classes*4*self.alpha*0.333)+int(num_classes*4*self.alpha* 0.5)+int(num_classes*4*self.alpha*0.69)
			
				self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,num_classes*2,kernel_size=(2,2),stride=(2,2))
				self.concat_filters3 = num_classes*2*2
				self.multiresblock8 = Multiresblock(self.concat_filters3,num_classes*2)
				self.in_filters8 = int(num_classes*2*self.alpha*0.167)+int(num_classes*2*self.alpha*0.333)+int(num_classes*2*self.alpha* 0.5)+int(num_classes*2*self.alpha*0.69)
			
				self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,num_classes,kernel_size=(2,2),stride=(2,2))
				self.concat_filters4 = num_classes*2
				self.multiresblock9 = Multiresblock(self.concat_filters4,num_classes)
				self.in_filters9 = int(num_classes*self.alpha*0.167)+int(num_classes*self.alpha*0.333)+int(num_classes*self.alpha* 0.5)+int(num_classes*self.alpha*0.69)

				self.conv_final = Conv2d_batchnorm(self.in_filters9, output_channels, kernel_size = (1,1), activation='None')

			def warp(self, tenInput, tenFlow):
				backwarp_tenGrid = {}
				k = (str(tenFlow.device), str(tenFlow.size()))
				if k not in backwarp_tenGrid:
					tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
					tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
					backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)
					# end
				tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

				g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
				return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

			def forward(self, img0, img1, flow0, flow1, mask, timestep):

				w_img0 = self.warp(img0, flow0) * 2 - 1
				w_img1 = self.warp(img1, flow0) * 2 - 1

				x = torch.cat((w_img0, flow0, mask, timestep, flow1, w_img1), dim=1)

				x_multires1 = self.multiresblock1(x)
				x_multires1_img0 = self.multiresblock1_img(img0 * 2 - 1)
				x_multires1_img1 = self.multiresblock1_img(img1 * 2 - 1)
				x_multires1 = torch.cat((
						self.warp(x_multires1_img0, flow0),
						x_multires1,
	 					self.warp(x_multires1_img1, flow1)
	  				),
					dim = 1)
				x_pool1 = self.pool1(x_multires1)
				x_pool1_img0 = self.pool1(x_multires1_img0)
				x_pool1_img1 = self.pool1(x_multires1_img1)
				x_multires1 = self.respath1(x_multires1)

				flow0 = torch.nn.functional.interpolate(flow0, scale_factor= 1 / 2, mode='bilinear', align_corners=False) * 1 / 2
				flow1 = torch.nn.functional.interpolate(flow1, scale_factor= 1 / 2, mode='bilinear', align_corners=False) * 1 / 2

				x_multires2 = self.multiresblock2(x_pool1)
				x_multires2_img0 = self.multiresblock2_img(x_pool1_img0)
				x_multires2_img1 = self.multiresblock2_img(x_pool1_img1)
				x_multires2 = torch.cat((
						self.warp(x_multires2_img0, flow0),
						x_multires2,
	 					self.warp(x_multires2_img1, flow1)
	  				),
					dim = 1)
				x_pool2 = self.pool2(x_multires2)
				x_pool2_img0 = self.pool2(x_multires2_img0)
				x_pool2_img1 = self.pool2(x_multires2_img1)
				x_multires2 = self.respath2(x_multires2)

				flow0 = torch.nn.functional.interpolate(flow0, scale_factor= 1 / 2, mode='bilinear', align_corners=False) * 1 / 2
				flow1 = torch.nn.functional.interpolate(flow1, scale_factor= 1 / 2, mode='bilinear', align_corners=False) * 1 / 2

				x_multires3 = self.multiresblock3(x_pool2)
				x_multires3_img0 = self.multiresblock3_img(x_pool2_img0)
				x_multires3_img1 = self.multiresblock3_img(x_pool2_img1)
				x_multires3 = torch.cat((
						self.warp(x_multires3_img0, flow0),
						x_multires3,
	 					self.warp(x_multires3_img1, flow1)
	  				),
					dim = 1)
				x_pool3 = self.pool3(x_multires3)
				x_pool3_img0 = self.pool2(x_multires3_img0)
				x_pool3_img1 = self.pool2(x_multires3_img1)
				x_multires3 = self.respath3(x_multires3)

				flow0 = torch.nn.functional.interpolate(flow0, scale_factor= 1 / 2, mode='bilinear', align_corners=False) * 1 / 2
				flow1 = torch.nn.functional.interpolate(flow1, scale_factor= 1 / 2, mode='bilinear', align_corners=False) * 1 / 2

				x_multires4 = self.multiresblock4(x_pool3)
				x_multires4_img0 = self.multiresblock4_img(x_pool3_img0)
				x_multires4_img1 = self.multiresblock4_img(x_pool3_img1)
				x_multires4 = torch.cat((
						self.warp(x_multires4_img0, flow0),
						x_multires4,
	 					self.warp(x_multires4_img1, flow1)
	  				),
					dim = 1)
				x_pool4 = self.pool4(x_multires4)
				x_multires4 = self.respath4(x_multires4)

				x_multires5 = self.multiresblock5(x_pool4)

				up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
				x_multires6 = self.multiresblock6(up6)
				'''
				del x_multires5
				del x_multires4
				del up6
				'''

				up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
				x_multires7 = self.multiresblock7(up7)
				'''
				del x_multires6
				del x_multires3
				del up7
				'''

				up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
				x_multires8 = self.multiresblock8(up8)
				'''
				del x_multires7
				del x_multires2
				del up8
				'''

				up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
				x_multires9 = self.multiresblock9(up9)
				'''
				del x_multires8
				del x_multires1
				del up9
				'''

				out =  self.conv_final(x_multires9)

				res_flow0 = out[: , :2]
				res_flow1 = out[: , 2:4]
				res_mask = torch.sigmoid(out[: , 4:5])
				
				return res_flow0, res_flow1, res_mask
		
		self.model = MultiResUnet
		self.training_model = MultiResUnet

	@staticmethod
	def get_name():
		return 'MultiRes4_v001'

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
		channels = 3
		try:
			channels = model_state_dict.get('conv_final.conv1.weight').shape[0]
		except Exception as e:
			print (f'Unable to get model dict output channels - setting to 3, {e}')
		return channels

	def get_model(self):
		return self.model
	
	def get_training_model(self):
		return self.training_model


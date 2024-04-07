class Model:
    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
                torch.nn.PReLU(out_planes)
            )

        def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                        kernel_size=4, stride=2, padding=1, bias=True),
                torch.nn.PReLU(out_planes)
            )


    @staticmethod
    def get_name():
        return 'model_v2_004'

    @staticmethod
    def get_file_name():
        import os
        return os.path.basename(__file__)

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

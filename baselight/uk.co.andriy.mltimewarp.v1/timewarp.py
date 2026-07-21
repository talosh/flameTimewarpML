import sys
import flexi
import json
import numpy
import platform
if (platform.system() == 'Darwin'):
    from multiprocessing import shared_memory
elif (platform.system() == 'Linux'):
    import cupy
    from numba import cuda
import torch
from torch.nn import functional as F
import math, time

def get_rife_model(req_device="cuda"):
    from RIFE_HDv3 import Model
    if (req_device == "mps" and not torch.backends.mps.is_available()):
        req_device = "cpu"
    if (req_device == "cuda" and not torch.cuda.is_available()):
        req_device = "cpu"
    
    device = torch.device(req_device)
    
    model = Model()
    model.load_model("RIFE/model/", device, -1)
    torch.set_grad_enabled(False)
    
    if device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model.eval()
    model.device(device)

    return model, req_device

def rife_inference(model, device, src1, src2, scale, ratio, nbPass, width, height):

    img1 = torch.as_tensor(src1, device=device)
    img2 = torch.as_tensor(src2, device=device)

    img1 = torch.reshape(img1, (4, height, width)).unsqueeze(0)
    img2 = torch.reshape(img2, (4, height, width)).unsqueeze(0)

    n, c, h, w = img1.shape

    #determine the padding
    pad = max(int(32 / scale), 128)
    ph = ((h - 1) // pad + 1) * pad
    pw = ((w - 1) // pad + 1) * pad
    padding = (0, pw - w, 0, ph - h)
    img1 = F.pad(img1, padding)
    img2 = F.pad(img2, padding)

    lf = 0.
    rf = 1.

    scale_list = [4, 2, 1]
    for i in range(nbPass):
        if ratio == lf:
            res = img1
            break
        elif ratio == rf:
            res = img2
            break

        f =  (ratio - lf) / (rf - lf) if(i == nbPass - 1) else 0.5

        res = model.inference(img1, img2, scale, f)

        if(i is not nbPass - 1):
            if ratio <= (lf + rf) / 2.:
                img2 = res
                rf = (lf + rf) / 2.
            else:
                img1 = res
                lf = (lf + rf) / 2.
    _, _, h2, w2 = res.shape

    out = res[0]
    if (platform.system() == 'Linux'):
        return out[:, :h, :w].contiguous()
    else:
        return out[:, :h, :w].contiguous().cpu()

######
#
# RIFE Retime
#
class RIFERetime(flexi.Effect):

    def __init__(self):
        self.init([('uk.ltd.filmlight.rife_retime', 1)])
        self.model = None
        self.vram_amount = 0

    def describe_effect(self, id):
        # We don't expose the effect
        return {}

    def query_vram_requirement(self, id, data):
        width = data['width']
        height = data['height']
        params = data['params']
        amount = 3. * math.sqrt(width * height) 
        #As the first instanciation will require more VRAM, make sure we request more
        # if the model hasn't been loaded
        return {'min_mb' : amount, 'max_mb' : 1.1 * amount}

    def set_vram_limit(self, id, data):
        limit = data['limit_mb']
        #If the limit is not enough, we need to desinstanciate the effect 
        #(don't bother if we haven't loaded the model)
        if(self.model is not None and limit < self.vram_amount):
            return 'quit'
        return {'RIFE' : 'Success'}

    def run_generate(self, id, instance, data):
        # No need to verify identifier or version, we only support one
        inputs = data['inputs']
        output = data['output']
        params = data['params']
        
        width = data['width']
        height = data['height']
        timestamp = params['timestamp']
        scale = params['scale']
        nbPass = int(params['pass'])
        
        #Account for VRAM usage
        amount = self.query_vram_requirement(id, data)['min_mb']
        if(self.vram_amount < amount):
            self.vram_amount = amount
       
        #instancidate the model if necessary with the correct backend
        if(self.model is None):
            device = "cpu"
            if (platform.system() == 'Darwin'):
                device = "mps"
            elif (platform.system() == 'Linux'):
                device = "cuda"
        
            self.model, self.device = get_rife_model(device)


        #start_t = time.time()
        
        if (platform.system() == 'Darwin'): #SHM
            out_mem = shared_memory.SharedMemory(name=output['shm'].lstrip('/'))
            src_mem  = shared_memory.SharedMemory(name=inputs['src:0']['shm'].lstrip('/'))
            src2_mem = shared_memory.SharedMemory(name=inputs.get('src:1', '-')['shm'].lstrip('/'))

            #map the shared memory into ndarray objects
            src1_array = numpy.ndarray((4, height, width), numpy.float32, src_mem.buf)
            src2_array = numpy.ndarray((4, height, width), numpy.float32, src2_mem.buf)

            out_array = numpy.ndarray((4, height, width), numpy.float32, out_mem.buf)
            out_array[:,:,:] = rife_inference(self.model, self.device, src1_array, src2_array, scale, timestamp, nbPass, width, height)
            #close the shared memory
            src_mem.close()
            src2_mem.close()
            out_mem.close()

        else: #CUDA_IPC x Linux
            shape = height * width * 16
            dtype = numpy.dtype(numpy.float32)

            #standard_b64decode
            ihandle1 = int(inputs['src:0']['cuda_ipc'], base=16).to_bytes(64, byteorder='big')
            ihandle2 = int(inputs.get('src:1','-')['cuda_ipc'], base=16).to_bytes(64, byteorder='big')
            ohandle  = int(output['cuda_ipc'], base=16).to_bytes(64, byteorder='big')
            
            with cuda.open_ipc_array(ihandle1, shape=shape // dtype.itemsize, dtype=dtype) as src1_array:
                with cuda.open_ipc_array(ihandle2, shape=shape // dtype.itemsize, dtype=dtype) as src2_array:
                    with cuda.open_ipc_array(ohandle, shape=shape // dtype.itemsize, dtype=dtype) as out_array:
                        
                        res = rife_inference(self.model, self.device, src1_array, src2_array, scale, timestamp, nbPass, width, height)

                        cupy.cuda.runtime.memcpy(out_array.gpu_data._mem.handle.value, res.__cuda_array_interface__['data'][0], height * width * 4 * 4, 4 )
            cuda.synchronize()
            torch.cuda.empty_cache()

        return {'metadata' : {'RIFE' : 'Success'}}

if __name__ == '__main__':

    # Register each effect
    RIFERetime()
    
    # and run
    flexi.run()

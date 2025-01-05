import torch
import torch.nn as nn

import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization.observer import MovingAverageMinMaxObserver


from MODNet.src.models.modnet import MODNet

pretrained_ckpt = './MODNet/pretrained/q_model.ckpt'


class Qmodel(nn.Module):
    def __init__(self, base, q = False):
        # By turning on Q we can turn on/off the quantization
        super(Qmodel, self).__init__()
        self.q = q
        self.base = base
        if q:
          self.quant = QuantStub()
          self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor, inference) -> torch.Tensor:
        if self.q:
          x = self.quant(x)
          
          o1, o2, matte = self.base(x, inference)
        
        if self.q:
          matte = self.dequant(matte)
        return o1,o2, matte

class MyQuantizedModel:
    def __init__(self, model_class=MODNet, model_weighs_path=pretrained_ckpt):
        """Note, we need model weights path to evaluate model weighs size
        
            In this example we do not calibrate model performance, because the weighs were saved previously.
        """
        
        self.model_weighs_path = model_weighs_path
        weights = torch.load(model_weighs_path,  map_location=torch.device('cpu'))
        base = model_class(backbone_pretrained=False)
        qmodel = Qmodel(base, q=True)
        qmodel.eval()
        qmodel.qconfig =  torch.ao.quantization.get_default_qconfig('fbgemm')
        prepared_model = torch.ao.quantization.prepare(qmodel, inplace=False)
        qmodel_int8 = torch.ao.quantization.convert(prepared_model)
        qmodel_int8.load_state_dict(weights)
        
        self.model = qmodel_int8
        self.model_path = model_weighs_path
    
    def __call__(self,image, flag=True):
        
        """Note, the model returns three variables 
           This code will be used for model performance validation

       def transform(self, model, image):
            
            im, im_h, im_w = self.process_image(image)
            im = im.to(self.device)
            _, _, matte = model(im.to(self.device), True)
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
        
            return Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        """
        return self.model(image, flag)
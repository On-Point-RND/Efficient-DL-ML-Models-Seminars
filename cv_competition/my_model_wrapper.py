import torch
from MODNet.src.models.modnet import MODNet


pretrained_ckpt = './MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'

class MyEfficientModel:
    def __init__(self, model_class=MODNet, model_weighs_path=pretrained_ckpt):
        """Note, we need model weights path to evaluate model weighs size"""
        self.model_weighs_path = model_weighs_path
        weights = torch.load(model_weighs_path,  map_location=torch.device('cpu'))
        model = model_class(backbone_pretrained=False)
        model.load_state_dict(weights)
        
        
        self.model = model
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
        
    



    

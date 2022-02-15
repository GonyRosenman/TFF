import torch
import imgaug.augmenters as iaa

class brain_gaussian(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.p = kwargs.get('augment_prob')
        self.blur = {'sigma':(0.0,1.1)}
        self.noise = {'scale':(0,0.1)}
        if self.p > 0:
            self.blur['object'] = iaa.GaussianBlur(sigma = self.blur['sigma'])
            self.noise['object'] = iaa.AdditiveGaussianNoise(scale = self.noise['scale'])

    def forward(self, img):
        C,H,W,D,T = img.shape
        if torch.rand((1,)) < self.p:
            aug = ['blur','noise'][torch.randint(2,(1,))]
            aug_dict = getattr(self,aug)
            aug_dict['object'] = aug_dict['object'].to_deterministic()
            for slic in range(img.shape[3]):
                to_augment = img[:,:,:,slic,:].permute(1,2,0,3).reshape(H,W,C * T).numpy()
                img[:,:,:,slic,:] = torch.from_numpy(aug_dict['object'].augment_image(image=to_augment)).reshape(H,W,C,T).permute(2,0,1,3)
        return img

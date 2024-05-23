import torch
from torchvision import transforms


class TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes(torch.nn.Module):
    def __init__(
        self, 
        resize=(128,128), 
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        super(TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes, self).__init__()
        #permuter_fn= transforms.Lambda(lambda x: x.permute(2, 0, 1))
        self.transform = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Resize(
            resize,
            antialias=True,
            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=mean,
            std=std)
        ])
        
    
    def forward(self, x):

        x=self.transform(x)
        return x

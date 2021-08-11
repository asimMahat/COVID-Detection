import io
import torch
from PIL import Image
import torchvision
from torchvision import models
import torchvision.transforms as transforms

def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features = 512, out_features = 4)
    checkpoint = torch.load('my_model.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

    # my_model = torch.load('my_model.pth')
    # return my_model
    
def transform_image(image_bytes):
    my_transforms =  torchvision.transforms.Compose([ 
                    torchvision.transforms.Resize(size=(224,224)),
                    torchvision.transforms.ToTensor(),  
                    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])  
                            ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    return my_transforms(image).unsqueeze(0)



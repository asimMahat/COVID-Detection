import torch
import io
import json
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from pathlib import Path

#Load model
model = models.resnet18()
model.fc = torch.nn.Linear(in_features = 512, out_features = 4)
checkpoint = torch.load('my_model.pth')
model.load_state_dict(checkpoint['model'])

#transforming image
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

resnet_class_index = json.load(open('resnet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.view(1, 3, 224,224)
    model.eval()
    outputs=model.forward(tensor)
    _,y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return resnet_class_index[predicted_idx]

with open("COVID-10.png",'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

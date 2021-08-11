 
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
import torchvision.models as models


model = models.resnet18()
model.fc = torch.nn.Linear(in_features = 512, out_features = 4)
checkpoint = torch.load('my_model.pth')
model.load_state_dict(checkpoint['model'])
model.eval()
 


transform = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.ToTensor(),  
    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])  
])


image = Image.open(Path('COVID-10.png')).convert('RGB') 
input = transform(image)
input = input.view(1, 3, 224,224)
output = model(input)
prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)
if (prediction == 0):
    print ('Normal')
if (prediction == 1):
    print ('Viral Pneumonia')
if (prediction == 2):
    print ('COVID')
if (prediction == 3):
    print ('Lung Opacity')

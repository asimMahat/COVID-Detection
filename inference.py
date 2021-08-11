import json

from commons import get_model,transform_image

model = get_model()

resnet_class_index = json.load(open('resnet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.view(1, 3, 224,224)
    # model.eval()
    outputs=model.forward(tensor)
    _,y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return resnet_class_index[predicted_idx]


    
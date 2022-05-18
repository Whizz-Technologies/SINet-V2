import onnx
import onnxruntime as nxrun
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2

onnx_model = onnx.load('Gaurav.onnx')
sess = nxrun.InferenceSession("./Gaurav.onnx")

print("The model expects input shape: ", sess.get_inputs()[0].shape)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[3].name

print("The model expects input name: ", input_name)
print("The model expects label name: ", label_name)

path_image = "./COD10K-CAM-1-Aquatic-3-Crab-71.jpg"
orig_img = cv2.imread(path_image)
input_image = Image.open(path_image)
img_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

input_tensor = img_transform(input_image).unsqueeze(0)
#input_batch = input_tensor.unsqueeze(0)
#print(input_tensor.shape)
result = sess.run([label_name], {input_name: input_tensor.numpy()})
#convert list to tensor
result = torch.tensor(result)
#print(result)
res = result.sigmoid().data.cpu().numpy().squeeze()
#convert image to unint8
res = (res * 255).astype(np.uint8)
res = cv2.resize(res, (orig_img.shape[1], orig_img.shape[0]))



#Save mask
cv2.imwrite('./onnx.jpg', res)

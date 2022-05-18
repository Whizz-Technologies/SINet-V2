import onnx
from onnx_tf.backend import prepare
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2


model = onnx.load('./Gaurav.onnx')
tf_rep = prepare(model)

path_image = "./COD10K-CAM-1-Aquatic-3-Crab-71.jpg"
orig_img = cv2.imread(path_image)
input_image = Image.open(path_image)
img_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

input_tensor = img_transform(input_image).unsqueeze(0)
output = tf_rep.run(input_tensor.numpy())

print(type(output))
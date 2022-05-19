from PIL import Image
from torchvision import transforms
from lib.Network_Res2Net_GRA_NCD import Network
import torch.nn.functional as F
import numpy as np
import cv2
import torch

path_image = "./COD10K-CAM-1-Aquatic-3-Crab-71.jpg"
input_image = Image.open(path_image)
#load .pt model
import onnxruntime as nxrun    #New Line Added
session_quant = nxrun.InferenceSession("model_uint8.onnx") #New Line Added
input_name = session_quant.get_inputs()[0].name  #New Line Added
label_name = session_quant.get_outputs()[3].name  #New Line Added
orig_img = cv2.imread(path_image)

#Image Transform and Inference
img_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

input_tensor = img_transform(input_image)
print("Input Tensor type",input_tensor.dtype)
input_batch = input_tensor.unsqueeze(0)
result = session_quant.run([label_name], {input_name: input_batch.numpy()}) #New Line Added





#Output Post-Processing
res = torch.tensor(result) #New Line Added
#res = F.upsample(res, size=352, mode='bilinear', align_corners=False)
res = res.sigmoid().data.cpu().numpy().squeeze()
#res = (res - res.min()) / (res.max() - res.min() + 1e-8)
print('> {}'.format(path_image))
#convert image to unint8
res = (res * 255).astype(np.uint8)
res = cv2.resize(res, (orig_img.shape[1], orig_img.shape[0]))



#Save mask
cv2.imwrite('./rr.jpg', res)
#find contour in res
ret, thresh = cv2.threshold(res, 100, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#plot contours
overlayed = cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 3)
overlayed_saved_path = './rr_overlayed.jpg'


#Save Overlayed Image
cv2.imwrite(overlayed_saved_path, overlayed)
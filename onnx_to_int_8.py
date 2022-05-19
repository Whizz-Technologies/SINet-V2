import onnx
import onnxruntime as nxrun
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import os
import time

onnx_model = onnx.load('Gaurav.onnx')


'''print("The model expects input shape: ", sess.get_inputs()[0].shape)

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
cv2.imshow('mask', res)
cv2.waitKey(0)
#cv2.imwrite('./onnx.jpg', res)'''

def preprocess_image(img_path, height, width, channel=3):
    input_image = Image.open(img_path)
    img_transform = transforms.Compose([
                    transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    input_tensor = img_transform(input_image).unsqueeze(0)
    #input_tensor = img_transform(input_image)
    #input_tensor = input_tensor.numpy()
    return input_tensor


def run_sample(session, image_tensor, label_name):
    current_time = time.time()
    result = sess.run([label_name], {input_name: input_tensor.numpy()})
    new_time = time.time()

    print("Time Taken to Run the Model is: ", new_time - current_time)
    result = torch.tensor(result)
    #print(result)
    res = result.sigmoid().data.cpu().numpy().squeeze()
    #convert image to unint8
    res = (res * 255).astype(np.uint8)
    #res = cv2.resize(res, (orig_img.shape[1], orig_img.shape[0]))
    cv2.imshow('mask', res)
    cv2.waitKey(0)

def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        image_data = image_data.numpy()
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    #batch_data = image_data
    #convert tensor to numpy array
    #batch_data = batch_data.numpy()
    print(batch_data.shape)
    return batch_data


class datareader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            image_height = 352
            image_width = 352
            nhwc_data_list = preprocess_func(self.image_folder, image_height, image_width, size_limit=0)
            #nhwc_data_list = preprocess_image("./COD10K-CAM-1-Aquatic-3-Crab-71.jpg", image_height, image_width)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

if __name__ == '__main__':
    img_path = "./COD10K-CAM-1-Aquatic-3-Crab-71.jpg"
    image_height = 352
    image_width = 352
    input_tensor = preprocess_image(img_path, image_height, image_width, channel=3)
    sess = nxrun.InferenceSession("./Gaurav.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[3].name
    run_sample(sess, input_tensor, label_name)
    
    # change it to your real calibration data set
    calibration_data_folder = "calibration_folder"

    #Here we change the model to Int8
    #dr = datareader(calibration_data_folder)

    #quantize_static('Gaurav.onnx', 'Gaurav_uint8.onnx', dr)

    print('ONNX full precision model size (MB):', os.path.getsize("Gaurav.onnx")/(352*352))
    print('ONNX quantized model size (MB):', os.path.getsize("Gaurav_uint8.onnx")/(352*352))

    session_quant = nxrun.InferenceSession("Gaurav_uint8.onnx")
    run_sample(session_quant, input_tensor, label_name)


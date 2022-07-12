import torch 
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
import imageio
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import shutil
import tensorflow as tf
from keras import backend as K
tf.config.experimental.set_visible_devices([], 'GPU')
def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred))
#   union = K.sum(y_true)+K.sum(y_pred)-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth))
#   return iou
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def predict(folder_path):
    path_ckpt = './Net_epoch_best.pth'
    model = Network(imagenet_pretrained=False)
    model.load_state_dict(torch.load(path_ckpt))
    model.cuda()
    model.eval()
    try:
        shutil.rmtree(folder_path + '/batch_output')

    except:
        pass
    os.mkdir(folder_path + '/batch_output')
    input_folder = folder_path + '/' + 'batch_input'
    output_folder = folder_path + '/' + 'batch_output'
    for file in os.listdir(input_folder):
        if 'mask' in file:
            continue 
        path = input_folder + '/' + file
        mask_save = output_folder + '/' + file.replace('.jpg', '_mask.jpg')
        overlay_save = output_folder + '/' + file.replace('.jpg', '_overlay.jpg')
        img = Image.open(path).convert('RGB')
        img_transform = transforms.Compose([
                    transforms.Resize((352, 352)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = img_transform(img).unsqueeze(0)
        img = img.cuda()

        orig_img = cv2.imread(path)
        #resize original image to 640 480
        #orig_img = cv2.resize(orig_img, (640, 480))
        res5, res4, res3, res2 = model(img)
        res = res2
        res = F.upsample(res, size=352, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #convert image to unint8
        res = (res * 255).astype(np.uint8)
        res = cv2.resize(res, (orig_img.shape[1], orig_img.shape[0]))
        cv2.imwrite(mask_save , res)
        #find contour in res
        ret, thresh = cv2.threshold(res, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #plot contours
        overlayed = cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(overlay_save, overlayed)


def metric(folder_path):
    input_folder = folder_path + '/' + 'batch_input'
    output_folder = folder_path + '/' + 'batch_output'
    mae_sum = 0
    no_files = 0
    iou_dict ={}
    iou_list = []
    for file in os.listdir(input_folder):
        if 'mask' in file: 
            no_files = no_files + 1
            input_mask_path = input_folder + '/' + file
            output_mask_path = output_folder + '/' + file.replace('mask.jpg', '_mask.jpg')
            print(input_mask_path)
            print(output_mask_path)
            input_mask = cv2.imread(input_mask_path)
            output_mask = cv2.imread(output_mask_path)
            #shape of input_mask 
            print(input_mask.shape)
            #shape of output_mask
            print(output_mask.shape)
            #convert input mask to int8
            input_mask = input_mask.astype(np.int8)
            #convert output mask to int8
            output_mask = output_mask.astype(np.int8)
            iou = iou_coef(input_mask, output_mask)
            print(iou)
            iou_dict = {file: iou}
            iou_list.append(iou_dict.copy())
    return iou_list



if __name__ == '__main__':
    predict(folder_path)
    metric(folder_path)

    
        

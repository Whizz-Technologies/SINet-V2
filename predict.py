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
#from matplotlib import pyplot as plt

path_image = './animal-1.jpg'



def copy(src, dst):
    for file in os.listdir(src):
        shutil.copyfile(os.path.join(src, file), dst)



def predict(folder_path):
    path_ckpt = './Net_epoch_best.pth'
    model = Network(imagenet_pretrained=False)
    model.load_state_dict(torch.load(path_ckpt))
    model.cuda()
    model.eval()
    input_folder = folder_path + '/' + 'input'
    output_folder = folder_path + '/' + 'output'
    currentDateAndTime = datetime.now()
    final_output_folder_name = './output/' + currentDateAndTime.strftime('%Y-%m-%d_%H-%M-%S') +'/'
    final_image_save_path_input = final_output_folder_name + 'input'
    final_image_save_path_output = final_output_folder_name + 'output'
    os.mkdir(final_output_folder_name)
    os.mkdir(final_image_save_path_input)
    os.mkdir(final_image_save_path_output)
    first_one = True
    final_image = None
    for file in os.listdir(input_folder):
        path = input_folder + '/' + file
        output_save_path = output_folder + '/' + file
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
        cv2.imwrite(final_image_save_path_input + '/' + file, orig_img)
        res5, res4, res3, res2 = model(img)
        # print(type(res5))
        #save res5 res4 res3 res2 into a single image
        # res5 = F.upsample(res5, size=352, mode='bilinear', align_corners=False)
        # res4 = F.upsample(res4, size=352, mode='bilinear', align_corners=False)
        # res3 = F.upsample(res3, size=352, mode='bilinear', align_corners=False)
        # res2 = F.upsample(res2, size=352, mode='bilinear', align_corners=False)
        # res5 = res5.sigmoid().data.cpu().numpy().squeeze()
        # res4 = res4.sigmoid().data.cpu().numpy().squeeze()
        # res3 = res3.sigmoid().data.cpu().numpy().squeeze()
        # res2 = res2.sigmoid().data.cpu().numpy().squeeze()
        # # res5 = (res5 - res5.min()) / (res5.max() - res5.min() + 1e-8)
        # # res4 = (res4 - res4.min()) / (res4.max() - res4.min() + 1e-8)
        # # res3 = (res3 - res3.min()) / (res3.max() - res3.min() + 1e-8)
        # # res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
        # res5 = np.uint8(res5 * 255)
        # res4 = np.uint8(res4 * 255)
        # res3 = np.uint8(res3 * 255)
        # res2 = np.uint8(res2 * 255)
        # #stack image vertically using opencv
        # kk = cv2.vconcat([res5, res4, res3, res2])
        # #write image to disk
        # cv2.imwrite("./stack.jpg", kk)
        res = res2
        res = F.upsample(res, size=352, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {}'.format(path_image))
        #convert image to unint8
        res = (res * 255).astype(np.uint8)
        res = cv2.resize(res, (orig_img.shape[1], orig_img.shape[0]))
        cv2.imwrite(output_save_path, res)
        cv2.imwrite(final_image_save_path_output + '/' + file, res)
        #find contour in res
        ret, thresh = cv2.threshold(res, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #plot contours
        overlayed = cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 3)
        contour_save_path = output_folder + '/' + 'contour_' + file
        cv2.imwrite(contour_save_path, overlayed)
        cv2.imwrite(final_image_save_path_output + '/' + 'contour_' + file, overlayed)
        #plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        #plt.imshow(res, alpha=0.5) 
        #plt.show()
        #make res three channel
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        #concat image column wise
        img_concat = np.concatenate((res, overlayed), axis=1)
        #write filename in img_concat
        cv2.putText(img_concat, file, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if(first_one):
            first_one = False
            final_image = img_concat

        else:
            final_image = np.concatenate((final_image, img_concat), axis=0)
    
    #write the final image
    final_image_save_path = folder_path + '/output/' + 'final_image.jpg'
    final_image_save_path_final = folder_path + '/output/' + 'final_image_final.jpg'
    cv2.imwrite(final_image_save_path, final_image)
    cv2.imwrite(final_image_save_path_final, final_image)
    #copy(output_folder, final_output_folder_name)
    #copy(input_folder, final_output_folder_name)
        


if __name__ == '__main__':
    path_image = './static/'
    predict(path_image)
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

path_image = './animal-1.jpg'

def predict(path_image):
    path_ckpt = './Net_epoch_best.pth'
    model = Network(imagenet_pretrained=False)
    model.load_state_dict(torch.load(path_ckpt))
    model.cuda()
    model.eval()
    img = Image.open(path_image)
    img.convert('RGB')
    img_transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = img_transform(img).unsqueeze(0)
    img = img.cuda()

    res5, res4, res3, res2 = model(img)
    res = res2
    res = F.upsample(res, size=352, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print('> {}'.format(path_image))
    #convert image to unint8
    res = (res * 255).astype(np.uint8)
    imageio.imwrite('./static/output.jpg', res)


if __name__ == '__main__':
    path_image = './animal-1.jpg'
    predict(path_image)
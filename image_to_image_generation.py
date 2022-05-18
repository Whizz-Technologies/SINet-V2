import cv2
import numpy as np
from PIL import Image
import os


def scale_camo_shape(image_path, scaling_factor):
    #read image
    img = cv2.imread(image_path)
    #resize image
    #print("Orginal image shape: ", img.shape)
    img = cv2.resize(img, (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))
    #print("Scaled image shape: ", img.shape)
    #cv2.imshow('overlayed', img)
    #cv2.waitKey(0)
    return img


def fill_contour(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #otsu thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey(0)
    return thresh


def overlay(background, img_path, contour, x, y):


    #background shape
    background_shape = background.size
    #read foreground Image
    img2 = Image.open(img_path).convert("RGBA").resize(background_shape)


    #read filled contour
    mask = contour

    #center cordinates of mask

    mask_shape = mask.shape

    mask_center = (int(mask_shape[1]/2), int(mask_shape[0]/2))

    #print("Mask Shape: ", mask_shape)

    #print("Mask Center Coordinates: ", mask_center)


    #difference between nex and previous center

    diff_x = max(0, x - mask_center[0])
    diff_y = max(0, y - mask_center[1])

    extend_y = max(0,  background_shape[1] - mask.shape[0] -diff_y)
    extend_x = max(0, background_shape[0] - mask.shape[1] -diff_x)

    #print(diff_x, diff_y, extend_x, extend_y)
    mask = cv2.copyMakeBorder(mask, diff_y, extend_y, diff_x, extend_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])


    cv2.imwrite('./contour_mask_resized.jpg', mask)

    #Open Image 
    mask = Image.open('./contour_mask_resized.jpg').convert('L').resize(background_shape)

    #print("Background Shape: ", background.size)
    #print("Filled Contour Shape: ", mask.size )
    #print("Foreground Shape: ", img2.size)
    #paste foreground on background
    im = Image.composite(background, img2, mask)
    #img2.show()
    #mask.show()
    #PIL image show
    #im.show()
    return im






if __name__ == '__main__':

    '''#check if camo_shape folder exist in ./static
    if not os.path.exists('./static/camo_shape'):
        os.makedirs('./static/camo_shape')

    else:
        #remove camo_shape folder
        shutil.rmtree('./static/camo_shape')
        os.makedirs('./static/camo_shape')


    #check if pattern folder exist in ./static
    if not os.path.exists('./static/pattern'):
        os.makedirs('./static/pattern')
    
    else:
        #remove pattern folder
        shutil.rmtree('./static/pattern')
        os.makedirs('./static/pattern')


    #check if background folder exist in ./static
    if not os.path.exists('./static/background'):
        os.makedirs('./static/background')

    else:
        #remove background folder
        shutil.rmtree('./static/background')
        os.makedirs('./static/background')


    #check if overlayed folder exist in ./static
    if not os.path.exists('./static/overlayed'):
        os.makedirs('./static/overlayed')

    else:
        #remove overlayed folder
        shutil.rmtree('./static/overlayed')
        os.makedirs('./static/overlayed')'''





    #camo_path = input("Enter Camo Image Path: ")
    #pattern_path = input("Enter Pattern Image Path: ")
    #background_path = input("Enter Background Image Path: ")

    camo_path = './static/camo_shape/'
    pattern_path = './static/pattern/'
    background_path = './static/background/'
    overlayed_path = './static/overlayed/'


    #check the file in camo_path
    if os.listdir(camo_path):
        print("Camo Image Found")
        camo_path = camo_path + os.listdir(camo_path)[0]

    if os.listdir(pattern_path):
        print("Pattern Image Found")
        pattern_path = pattern_path + os.listdir(pattern_path)[0]

    if os.listdir(background_path):
        print("Background Image Found")
        background_path = background_path + os.listdir(background_path)[0]



    scaling_factor = input("Scaling Factor: ")
    background = Image.open(background_path).convert("RGBA")
    scaled_camo = scale_camo_shape(camo_path, float(scaling_factor))
    #contour = border_extraction(img)
    img = fill_contour(scaled_camo)
    #center of img
    center = (int(img.shape[1]/2), int(img.shape[0]/2))

    print("Center of mask: ", center)
    print("X Cordinate must be less than ", background.size[0], " Y Cordinate must be less than ", background.size[1])


    
    #coordinates for mask
    x = ''
    y = ''
   
    x = input("Enter x coordinate: ")

    y = input("Enter y coordinate: ")

    if(x == '' or y == ''):
        x = center[0]
        y = center[1]

    overlayed_image = overlay(background, pattern_path, img, int(x), int(y))
    #save overlayed image to overlayed folder
    overlayed_image = overlayed_image.convert('RGB')
    overlayed_image.save(overlayed_path + 'overlayed.jpg')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
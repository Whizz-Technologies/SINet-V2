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
    # cv2.imshow('thresh', thresh)    
    # cv2.waitKey(0)
    return thresh


def overlay(background, scaling_factor, img_path, contour, x, y):
    #background shape
    background_shape_orig = background.size
    #read foreground Image
    #img2 = Image.open(img_path).convert("RGB").resize(background_shape)
    img2 = Image.open(img_path).convert("RGB")
    #print img2 shape
    print(img2.size)
    #resize img2 to scaling_factor
    #img2 = img2.resize((int(img2.size[0] * float(scaling_factor)), int(img2.size[1] * float(scaling_factor))))
    print(img2.size)
    width_ratio = background_shape_orig[0] / img2.size[0]
    height_ratio = background_shape_orig[1] / img2.size[1]
    print("width_ratio: ", width_ratio)
    print("height_ratio: ", height_ratio)

    #round the ratio to nearest integer
    width_ratio = int(round(width_ratio))
    height_ratio = int(round(height_ratio))

    print("width_ratio: ", width_ratio)
    print("height_ratio: ", height_ratio)

    #height and width
    height = int(img2.size[1] * height_ratio)
    width = int(img2.size[0] * width_ratio)

    #resize background to height and width
    background = background.resize((width, height))
    new_im = Image.new('RGB', (width, img2.size[1]))
    new_im2 = Image.new('RGB', (width, height))
    x_offset = 0
    for i in range(0, width_ratio):
        new_im.paste(img2, (x_offset,0))
        x_offset += img2.size[0]

    y_offset = 0
    for i in range(0, height_ratio):
        new_im2.paste(new_im, (0,y_offset))
        y_offset += new_im.size[1]

    #show the image
    #new_im2.show()
    #print(img2.size)
    img2 = new_im2
    #print shape of img2
    #print(img2.size)

    #print background shape
    #print(background.size)
    #read filled contour
    background_shape = background.size
    # #save img2
    # img2.save("img2.png")

    # #save background
    # background.save("background.png")


    # #read img
    # img = Image.open("img2.png").convert("RGBA")
    # #read background
    # bg = Image.open("background.png").convert("RGBA")
    mask = contour


    #normalize x and y
    widht_ratio_of_bg = background_shape[0] / background_shape_orig[0]
    height_ratio_of_bg = background_shape[1] / background_shape_orig[1]

    x = int(x * widht_ratio_of_bg)
    y = int(y * height_ratio_of_bg)

    print("x: ", x)
    print("y: ", y)
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
    #resize im to background original shape
    im = im.resize(background_shape_orig)
    #img2.show()
    #mask.show()
    #PIL image show
    #im.show()
    return im


def predict_ioig(camo_path, pattern_path, background_path, scaling_factor):

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



    
    background = Image.open(background_path).convert("RGBA")
    scaled_camo = scale_camo_shape(camo_path, float(scaling_factor))
    #contour = border_extraction(img)
    img = fill_contour(scaled_camo)
    #center of img
    center = (int(img.shape[1]/2), int(img.shape[0]/2))

    return center, background.size, background, img, pattern_path
    





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

    camo_path = './static/uploads/camo_shape/'
    pattern_path = './static/uploads/pattern/'
    background_path = './static/uploads/background/'
    overlayed_path = './static/uploads/overlayed/'
    scaling_factor = input("Scaling Factor: ")

    center, background_size , background, img, pattern_path = predict_ioig(camo_path, pattern_path, background_path, scaling_factor)
    #save img
    cv2.imwrite('./static/uploads/filled_contour.jpg', img)
    print(background_size)
    print(background)
    print(img)
    print(pattern_path)

    print("Center of mask: ", center)
    back=background_size
    if back:
        print("X Cordinate must be less than ", background_size[0])
        print("Y Cordinate must be less than ", background_size[1])
    # print("X Cordinate must be less than ", background_size[0], " Y Cordinate must be less than ", background_size[1])

    #coordinates for mask
    x = ''
    y = ''
   
    x = input("Enter x coordinate: ")

    y = input("Enter y coordinate: ")

    if(x == '' or y == ''):
        x = center[0]
        y = center[1]

    overlayed_image = overlay(background, scaling_factor, pattern_path, img, int(x), int(y))
    #save overlayed image to overlayed folder
    overlayed_image = overlayed_image.convert('RGB')
    overlayed_image.save(overlayed_path + 'overlayed.jpg')




    cv2.waitKey(0)
    cv2.destroyAllWindows()

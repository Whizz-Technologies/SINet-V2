from generate_i2g_batch import *
from run_batch_infer import *
import cv2
def run(number_of_images, folder_name, camo_path, pattern_path, background_path):
    img = batch_img(number_of_images, folder_name, camo_path, pattern_path, background_path)
    folder_path = './static/uploads/overlayed/' + folder_name
    predict(folder_path)
    y = metric(folder_path)
    return y,img


if __name__ == '__main__':
    camo_path = './static/uploads/camo_shape/'
    pattern_path = './static/uploads/pattern/'
    background_path = './static/uploads/background/'
    number_of_images = input("Enter number of images to be generated: ")
    number_of_images = int(number_of_images)
    folder_name = input("Enter folder name: ")
    value,img = run(number_of_images, folder_name, camo_path, pattern_path, background_path)
    print(value)
    cv2.imshow('image',img)
    cv2.waitKey(0)




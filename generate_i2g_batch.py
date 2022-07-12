import random
from image_to_image_generation import *
import shutil
import os


def sample_imgs(folder_path):
    first = True
    first_in_column = True
    sample_img = []
    number_of_rows = 4
    number_of_cols = 2
    for file in os.listdir(folder_path):
        print(file)
        print("Number of rows: ", number_of_rows, " Number of columns: ", number_of_cols)
        if number_of_rows == 0:
            break
        if 'mask' in file:
            continue
        else:
            img_path = folder_path + file
            img = cv2.imread(img_path)
            mask_path = folder_path + file.replace('.jpg', 'mask.jpg')
            mask = cv2.imread(mask_path)
            #resize mask to 160*120
            mask = cv2.resize(mask, (160, 120))
            #resize img to 160*120
            img = cv2.resize(img, (160, 120))
            #concat horizontally
            
            if((number_of_cols == 0) and first == True):
                print("Row Added")
                first = False
                sample_img = temp_img
                number_of_rows-=1
                number_of_cols = 2
                first_in_column = True
                temp_img = []


            
            if number_of_cols > 0:
                print("Column Added")
                number_of_cols-=1
                kk = np.concatenate((img, mask), axis=1)
                if (first_in_column == True):
                    temp_img = kk
                    first_in_column = False

                else:
                    temp_img = np.concatenate((temp_img, kk), axis=1)


            if(number_of_cols == 0 and first == False):
                print("Row Added")
                # cv2.imshow('sample', temp_img)
                # cv2.waitKey(0)
                # cv2.imshow('sample', sample_img)
                # cv2.waitKey(0)

                sample_img = np.concatenate((sample_img, temp_img), axis=0)
                number_of_rows-=1
                number_of_cols = 2
                first_in_column = True
                temp_img = []





    return sample_img



def batch_img(number, folder_name, camo_path, pattern_path, background_path):

    #remote everything in batch folder
    try:
        shutil.rmtree('./static/uploads/overlayed/' + folder_name + '/batch_input')

    except:
        pass
    #make batch folder
    os.makedirs('./static/uploads/overlayed/' + folder_name + '/batch_input')

    while(number != 0):
        # camo_path = './static/uploads/camo_shape/'
        # pattern_path = './static/uploads/pattern/'
        # background_path = './static/uploads/background/'
        overlayed_path = './static/uploads/overlayed/' + folder_name + '/batch_input/'
        scaling_factor = round(random.uniform(0.01, 0.50), 2)
        center, background_size , background, img, pattern_path_2 = predict_ioig(camo_path, pattern_path, background_path, scaling_factor)
        #save img
        cv2.imwrite('./static/uploads/filled_contour.jpg', img)
        # print(background_size)
        # print(background)
        # print(img)
        # print(pattern_path)

        # print("Center of mask: ", center)
        # back=background_size
        # print("X Cordinate must be less than ", background_size[0], " Y Cordinate must be less than ", background_size[1])
        x = random.uniform(1, background_size[0])
        y = random.uniform(1, background_size[1])
        
        overlayed_image, mask_img = overlay(background, scaling_factor, pattern_path_2, img, int(x), int(y), True)
        #save overlayed image to overlayed folder
        overlayed_image = overlayed_image.convert('RGB')
        overlayed_image.save(overlayed_path + 'overlayed' + str(number) + '.jpg')
        mask_img = mask_img.convert('RGB')
        mask_img.save(overlayed_path + 'overlayed' + str(number) + 'mask' + '.jpg')
        number = number - 1
    folder_path = './static/uploads/overlayed/' + folder_name + '/batch_input/'
    sample = sample_imgs(folder_path)
    return sample



if __name__ == '__main__':
    number = input("Enter number of images to be generated: ")
    number = int(number)
    folder_name = input("Enter folder name: ")
    batch_img(number, folder_name)

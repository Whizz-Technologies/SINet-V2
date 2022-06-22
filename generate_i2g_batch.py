import random
from image_to_image_generation import *








number = input("Enter number of images to be generated: ")

number = int(number)
while(number != 0):
    camo_path = './static/uploads/camo_shape/'
    pattern_path = './static/uploads/pattern/'
    background_path = './static/uploads/background/'
    overlayed_path = './static/uploads/overlayed/batch/'
    scaling_factor = round(random.uniform(0.01, 0.50), 2)
    center, background_size , background, img, pattern_path = predict_ioig(camo_path, pattern_path, background_path, scaling_factor)
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
    
    overlayed_image = overlay(background, scaling_factor, pattern_path, img, int(x), int(y))
    #save overlayed image to overlayed folder
    overlayed_image = overlayed_image.convert('RGB')
    overlayed_image.save(overlayed_path + 'overlayed' + str(number) + '.jpg')
    number = number - 1





cv2.waitKey(0)
cv2.destroyAllWindows()

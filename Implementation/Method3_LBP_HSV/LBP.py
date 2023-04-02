import cv2
import numpy as np
from tqdm import tqdm
import os

def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def LBP(query_path, save_path):
    for img_file in tqdm(os.listdir(query_path)):
        query_file = img_file.split('.')[0]

        # Read image file
        img_bgr = cv2.imread(query_path + query_file + '.jpg')

        #img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)

        height, width, _ = img_bgr.shape

        # We need to convert RGB image
        # into gray one because gray
        # image has one channel only.
        img_gray = cv2.cvtColor(img_bgr,
                                cv2.COLOR_BGR2GRAY)

        # Create a numpy array as
        # the same height and width
        # of RGB image
        img_lbp = np.zeros((height, width),
                           np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

        cv2.imwrite(save_path + query_file + '.jpg', img_lbp)



query_path = '../../Image Data/Testing Image/query_croped_4186/'
query_save_path = '../../Image Data/Testing Image/query_croped_LBP/'

gallery_path = '../../Image Data/Dataset Image/gallery_croped_4186/'
gallery_save_path = '../../Image Data/Dataset Image/gallery_croped_LBP/'


LBP(query_path, query_save_path)
LBP(gallery_path, gallery_save_path)




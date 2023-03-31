import cv2
import numpy as np
import os
from tqdm import tqdm


def img_preprocess(img_path, crop_txt_path, crop_img_save_path):
    query_img = cv2.imread(img_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    crop_info = np.loadtxt(crop_txt_path)     #load the coordinates of the bounding box
    croped_img = query_img[int(crop_info[1]):int(crop_info[1] + crop_info[3]), int(crop_info[0]):int(crop_info[0] + crop_info[2]), :] #crop the instance region
    cv2.imwrite(crop_img_save_path, croped_img[:,:,::-1])  #save the cropped region


# crop gallery image
img_path = './Dataset Image/gallery_4186/'
txt_crop_path = './Dataset Image/gallery_txt_4186/'
crop_img_save_path = './Dataset Image/gallery_croped_4186/'

for file in tqdm(os.listdir(img_path)):
    file_name = file.split('.')[0]
    print(file_name)
    img_preprocess(img_path + file_name + '.jpg', txt_crop_path + file_name + '.txt', crop_img_save_path + file_name + '.jpg')


# crop query image
img_path = './Testing Image/query_4186/'
txt_crop_path = './Testing Image/query_txt_4186/'
crop_img_save_path = './Testing Image/query_croped_4186/'

for file in tqdm(os.listdir(img_path)):
    file_name = file.split('.')[0]
    print(file_name)
    img_preprocess(img_path + file_name + '.jpg', txt_crop_path + file_name + '.txt', crop_img_save_path + file_name + '.jpg')
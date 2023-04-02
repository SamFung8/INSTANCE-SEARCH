import cv2
from tqdm import tqdm
import os
import numpy as np


def output(data, path, file):
    np.array(data)
    np.save(os.path.join(path, file.split('.')[0] + ".npy"), data)


def color_his(query_path, save_path_hsv):
    for img_file in tqdm(os.listdir(query_path)):
        query_file = img_file.split('.')[0]

        # Read image file
        img = cv2.imread(query_path + query_file + '.jpg')

        # Convert BGR to HSV format
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create empty lists for histogram values
        h_hist = [0] * 180
        s_hist = [0] * 256
        v_hist = [0] * 256

        # Loop through each pixel and increment histogram bins
        for i in range(img_hsv.shape[0]):
            for j in range(img_hsv.shape[1]):
                h_hist[img_hsv[i, j, 0]] += 1
                s_hist[img_hsv[i, j, 1]] += 1
                v_hist[img_hsv[i, j, 2]] += 1

        # Normalize histogram values by dividing by total number of pixels
        total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
        h_hist = [x / total_pixels for x in h_hist]
        s_hist = [x / total_pixels for x in s_hist]
        v_hist = [x / total_pixels for x in v_hist]
        data = []
        data.append(h_hist)
        data.append(s_hist)
        data.append(v_hist)
        output(data, save_path_hsv, query_file)


query_path = '../../Image Data/Testing Image/query_croped_LBP/'
query_save_path_hsv = './feature/query/HSV_color/'

gallery_path = '../../Image Data/Dataset Image/gallery_croped_LBP/'
gallery_save_path_hsv = './feature/gallery/HSV_color/'

color_his(query_path, query_save_path_hsv)
color_his(gallery_path, gallery_save_path_hsv)
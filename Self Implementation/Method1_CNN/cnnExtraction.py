import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

cnnModel_avg_layer = Xception(weights='imagenet', include_top=False, pooling='avg')
cnnModel_avg_layer.summary()

cnnModel_max_layer = Xception(weights='imagenet', include_top=False, pooling='max')
cnnModel_max_layer.summary()


def img_preprocess(img_path, crop_txt_path, crop_img_save_path, query_process):
    if query_process:
        query_img = cv2.imread(img_path)
        query_img = query_img[:,:,::-1] #bgr2rgb
        crop_info = np.loadtxt(crop_txt_path)     #load the coordinates of the bounding box
        croped_img = query_img[int(crop_info[1]):int(crop_info[1] + crop_info[3]), int(crop_info[0]):int(crop_info[0] + crop_info[2]), :] #crop the instance region
        cv2.imwrite(crop_img_save_path, croped_img[:,:,::-1])  #save the cropped region
        
        query_img = cv2.resize(query_img, (299, 299), interpolation=cv2.INTER_CUBIC)
        croped_img = cv2.resize(croped_img, (299, 299), interpolation=cv2.INTER_CUBIC)
        return croped_img, query_img
    else:
        img = cv2.imread(img_path)
        img = img[:,:,::-1] #bgr2rgb
        img_resize = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC) # resize the image
        return img

def extract_avg_layer(croped_img, img, featsave_path, img_name, query_process):
    save_path = os.path.join(featsave_path, img_name + "_uncropImg.npy")
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = cnnModel_avg_layer.predict(img_data)
    np.save(save_path, feature)
    
    if query_process:
        save_path = os.path.join(featsave_path, img_name + "_cropedImg.npy")
        img_data = image.img_to_array(croped_img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = cnnModel_avg_layer.predict(img_data)
        np.save(save_path, feature)
    
def extract_max_layer(croped_img, img, featsave_path, img_name, query_process):
    save_path = os.path.join(featsave_path, img_name + "_uncropImg.npy")
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = cnnModel_max_layer.predict(img_data)
    np.save(save_path, feature)
    
    if query_process:
        save_path = os.path.join(featsave_path, img_name + "_cropedImg.npy")
        img_data = image.img_to_array(croped_img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = cnnModel_max_layer.predict(img_data)
        np.save(save_path, feature)


def main():

    # Extract query image
    query_path = '../../Image Data/Testing Image/query_4186'
    txt_crop_path = '../../Image Data/Testing Image/query_txt_4186'
    crop_img_save_path = '../../Image Data/Testing Image/query_croped_4186'
    avg_feature_save_path = './feature/query_img/avg_layer'
    max_feature_save_path = './feature/query_img/max_layer'
    
    for img_file in tqdm(os.listdir(query_path)):
        img_name = img_file.split('.')[0]
        croped_img, uncrop_img = img_preprocess(os.path.join(query_path, img_file), os.path.join(txt_crop_path, img_name + ".txt"), os.path.join(crop_img_save_path, img_file), True)           
        extract_avg_layer(croped_img, uncrop_img, avg_feature_save_path, img_name, True)
        extract_max_layer(croped_img, uncrop_img, max_feature_save_path, img_name, True)


    # Extract gallery image
    gallery_path = '../../Image Data/Dataset Image/gallery_4186'
    avg_feature_save_path = './feature/gallery_img/avg_layer'
    max_feature_save_path = './feature/gallery_img/max_layer'
    
    for img_file in tqdm(os.listdir(gallery_path)):
        img_name = img_file.split('.')[0]
        img = img_preprocess(os.path.join(gallery_path, img_file), "", "", False)
        extract_avg_layer("", img, avg_feature_save_path, img_name, False)
        extract_max_layer("", img, max_feature_save_path, img_name, False)


if __name__=='__main__':
    main()
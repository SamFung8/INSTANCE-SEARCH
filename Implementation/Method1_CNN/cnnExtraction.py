import cv2
import os
import numpy as np
from tqdm import tqdm

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

cnnModel_avg_layer = Xception(weights='imagenet', include_top=False, pooling='avg')
cnnModel_avg_layer.summary()

cnnModel_max_layer = Xception(weights='imagenet', include_top=False, pooling='max')
cnnModel_max_layer.summary()


def img_preprocess(img_path):
    img = cv2.imread(img_path)
    img = img[:,:,::-1] #bgr2rgb
    img_resize = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC) # resize the image
    return img_resize

def extract_avg_layer(croped_img, featsave_path, img_name):
    save_path = os.path.join(featsave_path, img_name + "_cropedImg.npy")
    img_data = image.img_to_array(croped_img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = cnnModel_avg_layer.predict(img_data)
    np.save(save_path, feature)
    
def extract_max_layer(croped_img, featsave_path, img_name):
    save_path = os.path.join(featsave_path, img_name + "_cropedImg.npy")
    img_data = image.img_to_array(croped_img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = cnnModel_max_layer.predict(img_data)
    np.save(save_path, feature)


def main():

    # Extract query image
    query_path = '../../Image Data/Testing Image/query_croped_4186'
    avg_feature_save_path = './feature/query_img/avg_layer'
    max_feature_save_path = './feature/query_img/max_layer'
    
    for img_file in tqdm(os.listdir(query_path)):
        img_name = img_file.split('.')[0]
        croped_img = img_preprocess(os.path.join(query_path, img_file))
        extract_avg_layer(croped_img, avg_feature_save_path, img_name)
        extract_max_layer(croped_img, max_feature_save_path, img_name)


    # Extract gallery image
    gallery_path = '../../Image Data/Dataset Image/gallery_croped_4186'
    avg_feature_save_path = './feature/gallery_img/avg_layer'
    max_feature_save_path = './feature/gallery_img/max_layer'
    
    for img_file in tqdm(os.listdir(gallery_path)):
        img_name = img_file.split('.')[0]
        croped_img = img_preprocess(os.path.join(gallery_path, img_file))
        extract_avg_layer(croped_img, avg_feature_save_path, img_name)
        extract_max_layer(croped_img, max_feature_save_path, img_name)

if __name__=='__main__':
    main()
import cv2
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt

def similarity(query_feat, gallery_feat):
    query_feat = cv2.resize(query_feat, (224, 224), interpolation=cv2.INTER_CUBIC)
    gallery_feat = cv2.resize(gallery_feat, (224, 224), interpolation=cv2.INTER_CUBIC)
    # query_feat = cv2.cvtColor(query_feat, cv2.COLOR_BGR2GRAY)
    # gallery_feat = cv2.cvtColor(gallery_feat, cv2.COLOR_BGR2GRAY)
    query_feat = cv2.calcHist([query_feat], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # cv2.normalize(query_feat, query_feat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    gallery_feat = cv2.calcHist([gallery_feat], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # cv2.normalize(gallery_feat, gallery_feat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # query_feat = query_feat.reshape(1,query_feat.shape[0] * query_feat.shape[1])
    # gallery_feat = gallery_feat.reshape(1,gallery_feat.shape[0] * gallery_feat.shape[1])
    sim = cv2.compareHist(query_feat, gallery_feat, cv2.HISTCMP_BHATTACHARYYA)
    # print(sim)
    sim = np.squeeze(sim)
    return sim


def retrival_idx(gallery_dir, query_dir, query_file):
    query_feat = cv2.imread(os.path.join(query_dir + query_file) + '.jpg')
    # print(os.path.join(query_dir + query_file) + '.npy')
    dict = {}
    for gallery_file in tqdm(os.listdir(gallery_dir)):
        gallery_feat = cv2.imread(os.path.join(gallery_dir, gallery_file))
        # print(os.path.join(gallery_dir, gallery_file))
        gallery_idx = gallery_file.split('.')[0] + '.jpg'
        sim = similarity(query_feat, gallery_feat)
        res = sim
        dict[gallery_idx] = res
    sorted_dict = sorted(dict.items(), key=lambda item: item[1])  # Sort the similarity score
    # print(sorted_dict)
    return sorted_dict


def visulization(retrived, query):
    plt.subplot(4, 3, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:, :, ::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(9):
        img_path = '../../Image Data/Dataset Image/gallery_croped_4186/' + retrived[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:, :, ::-1]
        plt.subplot(4, 3, i + 1)
        plt.title(retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()


def getRank(query_path, gallery_dir):
    query_list = []
    for img_file in tqdm(os.listdir(query_path)):
        query_list.append(int(img_file.split('.')[0]))

    query_list.sort()
    # print(query_list)

    for img in query_list:
        query_file = str(img)
        result = retrival_idx(gallery_dir, query_path, query_file)
        # result.reverse()
        visulization(result[0:9], os.path.join('../../Image Data/Testing Image/query_croped_4186/', query_file) + '.jpg')


query_path = '../../Image Data/Testing Image/query_croped_4186/'
query_save_path = './feature/query/HSV_color/'

gallery_path = '../../Image Data/Dataset Image/gallery_croped_4186/'
gallery_save_path = './feature/gallery/HSV_color/'


getRank(query_path, gallery_path)
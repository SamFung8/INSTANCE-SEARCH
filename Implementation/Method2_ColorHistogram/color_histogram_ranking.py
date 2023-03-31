import cv2
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

def similarity(query_feat, gallery_feat, query_feat2, gallery_feat2):
    h_sim = euclidean(query_feat2[0], gallery_feat2[0])
    s_sim = euclidean(query_feat2[1], gallery_feat2[1])
    v_sim = euclidean(query_feat2[2], gallery_feat2[2])
    sim2 = h_sim + s_sim + v_sim

    h_sim = euclidean(query_feat[0], gallery_feat[0])
    s_sim = euclidean(query_feat[1], gallery_feat[1])
    v_sim = euclidean(query_feat[2], gallery_feat[2])
    sim = h_sim + s_sim + v_sim

    sim = (sim + sim2) /2

    sim = np.squeeze(sim)
    return sim


def retrival_idx(gallery_dir, query_dir, query_file):
    query_feat = np.load(os.path.join(query_dir + query_file) + '.npy', allow_pickle=True)
    query_feat2 = np.load(os.path.join('./feature/query/RGB_color/' + query_file) + '.npy', allow_pickle=True)
    # print(os.path.join(query_dir + query_file) + '.npy')
    dict = {}
    for gallery_file in tqdm(os.listdir(gallery_dir)):
        gallery_feat = np.load(os.path.join(gallery_dir, gallery_file), allow_pickle=True)
        gallery_feat2 = np.load(os.path.join('./feature/gallery/RGB_color/', gallery_file), allow_pickle=True)
        # print(os.path.join(gallery_dir, gallery_file))
        gallery_idx = gallery_file.split('.')[0] + '.jpg'
        sim = similarity(query_feat, gallery_feat, query_feat2, gallery_feat2)
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
        img_path = '../../Image Data/Dataset Image/gallery_4186/' + retrived[i][0]
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
        visulization(result[0:9], os.path.join('../../Image Data/Testing Image/query_4186/', query_file) + '.jpg')



query_save_path = './feature/query/HSV_color/'
gallery_save_path = './feature/gallery/HSV_color/'


getRank(query_save_path, gallery_save_path)
import cv2
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

def similarity(query_feat_hsv, gallery_feat_hsv, query_feat_rgb, gallery_feat_rgb):
    r_sim = euclidean(query_feat_rgb[0], gallery_feat_rgb[0])
    g_sim = euclidean(query_feat_rgb[1], gallery_feat_rgb[1])
    b_sim = euclidean(query_feat_rgb[2], gallery_feat_rgb[2])
    sim_rgb = r_sim + g_sim + b_sim

    h_sim = euclidean(query_feat_hsv[0], gallery_feat_hsv[0])
    s_sim = euclidean(query_feat_hsv[1], gallery_feat_hsv[1])
    v_sim = euclidean(query_feat_hsv[2], gallery_feat_hsv[2])
    sim_hsv = h_sim + s_sim + v_sim

    sim = (sim_hsv + sim_rgb) /2

    sim = np.squeeze(sim)
    return sim


def retrival_idx(gallery_dir_hsv, query_dir_hsv, query_file, gallery_dir_rgb, query_dir_rgb):
    query_feat_hsv = np.load(os.path.join(query_dir_hsv + query_file) + '.npy', allow_pickle=True)
    query_feat_rgb = np.load(os.path.join(query_dir_rgb + query_file) + '.npy', allow_pickle=True)
    # print(os.path.join(query_dir + query_file) + '.npy')
    dict = {}
    for gallery_file in tqdm(os.listdir(gallery_dir_hsv)):
        gallery_feat_hsv = np.load(os.path.join(gallery_dir_hsv, gallery_file), allow_pickle=True)
        gallery_feat_rgb = np.load(os.path.join(gallery_dir_rgb, gallery_file), allow_pickle=True)
        # print(os.path.join(gallery_dir, gallery_file))
        gallery_idx = gallery_file.split('.')[0] + '.jpg'
        sim = similarity(query_feat_hsv, gallery_feat_hsv, query_feat_rgb, gallery_feat_rgb)
        res = sim
        dict[gallery_idx] = res
    sorted_dict = sorted(dict.items(), key=lambda item: item[1])  # Sort the similarity score
    # print(sorted_dict)
    return sorted_dict


def visulization(retrived, query):
    plt.subplot(4, 4, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:, :, ::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(9):
        img_path = '../../Image Data/Dataset Image/gallery_4186/' + retrived[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:, :, ::-1]
        plt.subplot(4, 4, i + 2)
        plt.title('Value = ' + "%.2f" % retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()


def output(result, count):
    print('Outputting rank list of Q' + str(count))
    f = open(r'./rank_list.txt', 'a')
    f.write('Q'+str(count)+': ')
    for j in result:
        f.write(str(j[0].split('.')[0])+' ')
    f.write('\n')
    f.close()


def getRank(query_path_hsv, gallery_dir_hsv, query_path_rgb, gallery_dir_rgb):
    query_list = []
    count = 1
    for img_file in tqdm(os.listdir(query_path_hsv)):
        query_list.append(int(img_file.split('.')[0]))

    query_list.sort()
    # print(query_list)

    for img in query_list:
        query_file = str(img)

        result = retrival_idx(gallery_dir_hsv, query_path_hsv, query_file, gallery_dir_rgb, query_path_rgb)
        # result.reverse()

        output(result, count)
        count = count + 1

        visulization(result, os.path.join('../../Image Data/Testing Image/query_4186/', query_file) + '.jpg')



query_save_path_hsv = './feature/query/HSV_color/'
gallery_save_path_hsv = './feature/gallery/HSV_color/'

query_save_path_rgb = './feature/query/RGB_color/'
gallery_save_path_rgb = './feature/gallery/RGB_color/'

os.remove('./rank_list.txt')
getRank(query_save_path_hsv, gallery_save_path_hsv, query_save_path_rgb, gallery_save_path_rgb)
# Retrieve the most similar images by measuring the similarity between features.
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from tqdm import tqdm


def output(result, count):
    print('Outputting rank list of Q' + str(count))
    f = open(r'./rank_list.txt', 'a')
    f.write('Q'+str(count)+': ')
    for j in result:
        f.write(str(j[0].split('.')[0])+' ')
    f.write('\n')
    f.close()


def similarity(query_feat, gallery_feat):
    #print(query_feat.shape)
    if len(query_feat.shape) == 3:
      nd, nx, ny = query_feat.shape
      query_feat = query_feat.reshape(1,nd* nx*ny)
      nd, nx, ny = gallery_feat.shape
      gallery_feat = gallery_feat.reshape(1,nd* nx*ny)
    else:
      query_feat = query_feat.reshape(1,query_feat.shape[0] * query_feat.shape[1])
      gallery_feat = gallery_feat.reshape(1,gallery_feat.shape[0] * gallery_feat.shape[1])
    sim = cosine_similarity(query_feat, gallery_feat)
    #print(sim)
    sim = np.squeeze(sim)
    return sim

def retrival_idx(gallery_dir):
    query_cropped_feat_max = np.load('./feature/query_img/max_layer/' + query_file + '_cropedImg.npy')
    query_cropped_feat_avg = np.load('./feature/query_img/avg_layer/' + query_file + '_cropedImg.npy')
    dict = {}
    for gallery_file in os.listdir(gallery_dir):
        gallery_feat_max = np.load(os.path.join('./feature/gallery_img/max_layer/', gallery_file))
        gallery_feat_avg = np.load(os.path.join('./feature/gallery_img/avg_layer/', gallery_file))
        gallery_idx = gallery_file.split('_')[0] + '.jpg'
        sim_max_cropped = similarity(query_cropped_feat_max, gallery_feat_max)
        sim_avg_cropped = similarity(query_cropped_feat_avg, gallery_feat_avg)
        res = (sim_avg_cropped + sim_max_cropped)/ 2
        dict[gallery_idx] = res
    #print(dict)
    sorted_dict = sorted(dict.items(), key=lambda item: item[1]) # Sort the similarity score
    return sorted_dict

def visulization(retrived, query):
    plt.subplot(4, 4, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(9):
        img_path = '../../Image Data/Dataset Image/gallery_4186/' + retrived[i][0]
        print(img_path)
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(4, 4, i+2)
        plt.title('Value = ' + "%.2f" % retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()

if __name__ == '__main__':
    os.remove('./rank_list.txt')
    query_list = []
    query_path = '../../Image Data/Testing Image/query_4186/'
    count = 1

    for img_file in tqdm(os.listdir(query_path)):
        query_list.append(int(img_file.split('.')[0]))
    
    query_list.sort()
    print(query_list)
    
    for img in query_list:
        query_file = str(img)
        gallery_dir = './feature/gallery_img/max_layer/'
        
        result = retrival_idx(gallery_dir)
        result.reverse()
    
        output(result, count)
        count = count + 1
        
        #best_five = retrival_idx(gallery_dir) # retrieve top 5 matching images in the gallery.
        #print(best_five)
        #best_five.reverse()
        #cv2.imread("./data/gallery/" + str(18841.jpg))
        query_path = '../../Image Data/Testing Image/query_4186/' + query_file + '.jpg'
        visulization(result, query_path) # Visualize the retrieval results


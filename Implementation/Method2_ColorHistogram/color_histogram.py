import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def output(data, path, file):
    np.array(data)
    np.save(os.path.join(path, file.split('.')[0] + ".npy"), data)

def color_his(query_path, save_path):
    for img_file in tqdm(os.listdir(query_path)):
        query_file = img_file.split('.')[0]

        # Read image file
        img = cv2.imread(query_path + query_file + '.jpg')
        
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Convert BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create empty lists for histogram values
        red_hist = [0] * 256
        green_hist = [0] * 256
        blue_hist = [0] * 256

        # Loop through each pixel and increment histogram bins
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                red_hist[img[i,j,0]] += 1
                green_hist[img[i,j,1]] += 1
                blue_hist[img[i,j,2]] += 1

        # Normalize histogram values by dividing by total number of pixels
        total_pixels = img.shape[0] * img.shape[1]
        red_hist = [x / total_pixels for x in red_hist]
        green_hist = [x / total_pixels for x in green_hist]
        blue_hist = [x / total_pixels for x in blue_hist]
        data = []
        data.append(red_hist)
        data.append(green_hist)
        data.append(blue_hist)
        output(data, save_path, query_file)


def similarity(query_feat, gallery_feat):
    query_feat = cv2.resize(query_feat, (224, 224), interpolation=cv2.INTER_CUBIC)
    gallery_feat = cv2.resize(gallery_feat, (224, 224), interpolation=cv2.INTER_CUBIC)
    #query_feat = cv2.cvtColor(query_feat, cv2.COLOR_BGR2GRAY)
    #gallery_feat = cv2.cvtColor(gallery_feat, cv2.COLOR_BGR2GRAY)
    query_feat = cv2.calcHist([query_feat], [0,1,2], None, [256,256,256], [0, 256,0, 256,0, 256])
    #cv2.normalize(query_feat, query_feat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    gallery_feat = cv2.calcHist([gallery_feat], [0,1,2], None, [256,256,256], [0, 256,0, 256,0, 256])
    #cv2.normalize(gallery_feat, gallery_feat, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #query_feat = query_feat.reshape(1,query_feat.shape[0] * query_feat.shape[1])
    #gallery_feat = gallery_feat.reshape(1,gallery_feat.shape[0] * gallery_feat.shape[1])
    sim = cv2.compareHist(query_feat, gallery_feat, cv2.HISTCMP_BHATTACHARYYA )
    #print(sim)
    sim = np.squeeze(sim)
    return sim
    
def retrival_idx(gallery_dir, query_dir, query_file):


    query_feat = cv2.imread(os.path.join(query_dir + query_file) + '.jpg')
    #print(os.path.join(query_dir + query_file) + '.npy')
    dict = {}
    for gallery_file in tqdm(os.listdir(gallery_dir)):
        gallery_feat = cv2.imread(os.path.join(gallery_dir, gallery_file))
        #print(os.path.join(gallery_dir, gallery_file))
        gallery_idx = gallery_file.split('.')[0] + '.jpg'
        sim = similarity(query_feat, gallery_feat)
        res = sim
        dict[gallery_idx] = res
    sorted_dict = sorted(dict.items(), key=lambda item: item[1]) # Sort the similarity score
    #print(sorted_dict)
    return sorted_dict    
    
        
def visulization(retrived, query):
    plt.subplot(4, 3, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(9):
        img_path = './own_data/gallery_4186/' + retrived[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(4, 3, i+1)
        plt.title(retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()
    
def getRank(query_path, gallery_dir):
    query_list = []
    for img_file in tqdm(os.listdir(query_path)):
        query_list.append(int(img_file.split('.')[0]))

    query_list.sort()
    #print(query_list)

    for img in query_list:
        query_file = str(img)        
        result = retrival_idx(gallery_dir, query_path, query_file)
        #result.reverse()    
        visulization(result[0:9] , os.path.join('./own_data/query_4186/', query_file) + '.jpg')
    

query_path = './own_data/query_4186/'
query_save_path = './color_his/query_LBP/'

gallery_path = './own_data/gallery_4186/'
gallery_save_path = './color_his/gallery_LBP/'

#color_his(query_path, query_save_path)
#color_his(gallery_path, gallery_save_path)

getRank(query_path, gallery_path)

#file = '35.jpg'
#f = open(os.path.join(query_save_path, file.split('.')[0] + ".txt"), 'r')
#print(f.shape)
#f.close()


#print(np.load(os.path.join(query_save_path, "35.npy")).shape)
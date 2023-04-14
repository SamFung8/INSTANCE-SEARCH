# INSTANCE-SEARCH- ASSIGNMENT 1 CS4186 VISION AND IMAGE 

You are given a collection of 5,000 images and 20 testing query instances (you can download from this link: [onedrive link](https://portland-my.sharepoint.com/personal/srwang3-c_my_cityu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsrwang3%2Dc%5Fmy%5Fcityu%5Fedu%5Fhk%2FDocuments%2Fcityu%5Fclass%2FCS4186%5F5187%5FCV%2Fasg1%5Fdata%2Fdatasets%5F4186%2Ezip&parent=%2Fpersonal%2Fsrwang3%2Dc%5Fmy%5Fcityu%5Fedu%5Fhk%2FDocuments%2Fcityu%5Fclass%2FCS4186%5F5187%5FCV%2Fasg1%5Fdata&ga=1)). Each image contains one instance (object). Your task is to implement two methods for instance search. Specifically, given a query instance with the instance bounding box location in the query instance image (stored in the query_txt directory with the same name of query instance image), a method needs to find the images that contain the same query instance from the image collection （5000 images） and then ranks them according to similarity or confidence. The 20 testing query images are used for evaluating the performance of your implementation.  

# Image Preprocessing
I had used a tool called labelImg (https://github.com/heartexlabs/labelImg) label the object location of gallery dataset. And it was saved at ‘\Image Data\Dataset Image\gallery_crop_info’ as .xml file.
To get the cropped image of query and gallery dataset you need to first: 1. .xml file -> .txt formate; 2. load .txt -> crop the image
1. run '\Image Data\Dataset Image\get_crop_txt.py'
2. run '\Image Data\crop_img.py'


# How To Run
Below the the step of each method running process.

######## Method 1: CNN ########
1. run ‘\Implementation\Method1_CNN\cnnExtraction.py’ 
2. run ‘\Implementation\Method1_CNN\cnnRankList.py’

######## Method 2: Color Histogram ########
1. run ‘\Implementation\Method2_ColorHistogram\color_histogram.py’ 
2. run ‘\Implementation\Method2_ColorHistogram\color_histogram_ranking.py’

######## Method 3: LBP + Color Histogram ########
1. run ‘\Implementation\Method3_LBP_HSV\LBP.py’
2. run ‘\Implementation\Method3_LBP_HSV\color_histogram_HSV.py’ 
3. run ‘\Implementation\Method3_LBP_HSV\LBP_color_his_ranking.py’
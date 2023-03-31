import os
from tqdm import tqdm
from bs4 import BeautifulSoup

def create_txt(input_path, file_name, output_path):
    with open(input_path + file_name + '.xml', 'r') as f:
        data = f.read()

    # Passing the stored data inside the beautifulsoup parser
    bs_data = BeautifulSoup(data, 'xml')

    # Finding all instances of tag
    xmin = bs_data.find('xmin')
    ymin = bs_data.find('ymin')
    xmax = bs_data.find('xmax')
    ymax = bs_data.find('ymax')

    text = open(output_path + file_name + '.txt', "w")
    text.write(xmin.text + " " + ymin.text + " " + str(int(xmax.text) - int(xmin.text)) + " " + str(int(ymax.text) - int(ymin.text)) + "\n")
    text.close()


def main():
    info_path = './gallery_crop_info/'
    txt_path = './gallery_txt_4186/'

    for file in tqdm(os.listdir(info_path)):
        file_name = file.split('.')[0]
        print(file_name)
        create_txt(info_path, file_name, txt_path)


if __name__=='__main__':
    main()
import numpy as np
import os
from PIL import Image

#array()方法将图像转换成NumPy的数组对象

img_dir = 'F:/jinhao/PSPNet/output/market/'
img_npy_path = 'F:/jinhao/PSPNet/output/market_npy/'
def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(file)
    return L
def img2npy(img_path):
	img = np.array(Image.open(img_path))
	np.save(img_path+".npy", img)

L = file_name(img_dir)
length = len(L)

for i in range(length):
	img2npy(img_dir+L[i])

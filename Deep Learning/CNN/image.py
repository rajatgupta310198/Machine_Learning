import os, cv2, glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
#data_dir = '/data'
#
#image_dir = '/img'

#os.mkdir(os.getcwd() + '/' + data_dir + '/' + image_dir)
#Image  in BGR so train network in BGR and while serving read convert images from RGB to BGR
# os.chdir(os.getcwd() + data_dir)

def extract_images_labels(file_name):
    f = open(file_name, 'rb')

    dict = pickle.load(f, encoding='bytes')
    f.close()
    images = dict[b'data']
    labels = dict[b'labels']
    n_values = np.max(labels) + 1
    labels = np.eye(n_values)[labels]
    images = images.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    return images, labels
#dataset = unpickle()
# images, la = extract_images_labels('data_batch_2')
# image = images[10]
#img = cv2.imread('0.jpg')
#cv2.imshow('Ship original 320x320', img)
#img_1 = cv2.resize(img, (100, 100))
#print(img_1.shape)
## image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
## image = cv2.imread('ship.png')
## image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imshow('Ship 100x100', img_1)
#cv2.waitKey(0)
#cv2.destroyAllwindows()

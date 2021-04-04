# Read data from MNIST dataset
# number of train data: 60,000
# number of test data : 10,000
# Software: Python3
# Designer: Black Chocolate

import os
import struct
import numpy as np
# import time
# import cv2
# import matplotlib.pyplot as plt
"""
parameter:
path -- minist dataset path
num  -- the number of train or test image
label-- the data for train or test
"""
def mnist_data(path, kind='train'):

  if(kind=='train'):
    kind = 'train'
    # print('Read MNIST data for train')
  else:
    kind = 't10k'
    # print('Read MNIST data for test')
  
  """path join"""
  label_path = os.path.join(path,'%s-labels-idx1-ubyte'%kind)
  image_path = os.path.join(path,'%s-images-idx3-ubyte'%kind)
  
  """read lable"""
  with open(label_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    label    = np.fromfile(lbpath, dtype=np.uint8)

    # if(magic==2049 and label.shape==(n,) and n==60000):
    #   print("Read train label done!")
    # elif(magic==2049 and label.shape==(n,) and n==10000):
    #   print("Read test label done!")
    # else:
    #   while(1):
    #     print("Read label error!")
    #     time.sleep(1)
    # print('%s label num is %d' %(kind, n))

  """read image"""  
  with open(image_path, 'rb') as imgpath:
    magic, n, row, column = struct.unpack('>IIII', imgpath.read(16))
    image    = np.fromfile(imgpath, dtype=np.uint8)
    
    # if(magic==2051 and label.shape==(n,) and n==60000):
    #   print("Read train label done!")
    # elif(magic==2051 and label.shape==(n,) and n==10000):
    #   print("Read test label done!")
    # else:
    #   while(1):
    #     print("Read label error!")
    #     time.sleep(1)
    # print('%s image num is %d' %(kind, n))
  return label,image

def mnist_data_batch(batch=200, kind='train', one_hot=True):
    label ,image = mnist_data('MNIST', kind)
    image = image.reshape([-1, 784])
    label = label.reshape([-1,1])
    img_lb = np.concatenate([image, label], axis=1)
    
    np.random.shuffle(img_lb)
    row_rand = img_lb[0:batch,:]
    image = row_rand[:,0:784]
    label = row_rand[:,784]
    
    if(one_hot==True):
        label_one_hots = np.zeros([batch, 10])
        for i in range(batch):
            label_one_hots[i,label[i]] = 1
    return image, label_one_hots

class mnist_img():
    def __init__(self, path='MNIST', batch=200, one_hot=True):
        self.one_hot = one_hot
        self.path    = path
        self.batch   = batch
        self.train_image = np.zeros((60000,784),dtype='uint8')
        self.train_label = np.zeros((60000,1),dtype='uint8')
        self.train_label_batch = np.zeros((batch, 10),dtype='uint8')
        self.train_image_batch = np.zeros((batch, 784),dtype='uint8')
        self.test_image = np.zeros((10000,784),dtype='uint8')
        self.test_label = np.zeros((10000,1),dtype='uint8')
        self.test_label_batch = np.zeros((batch, 10),dtype='uint8')
        self.test_image_batch = np.zeros((batch, 784),dtype='uint8')

    def update_train_img(self):
      print("Read image and labels")
      """path join"""
      label_path = os.path.join(self.path,'train-labels-idx1-ubyte')
      image_path = os.path.join(self.path,'train-images-idx3-ubyte')
      
      """read lable"""
      with open(label_path, 'rb') as lbpath:
        magic, n    = struct.unpack('>II', lbpath.read(8))
        self.train_label = np.fromfile(lbpath, dtype=np.uint8)
        self.train_label = self.train_label.reshape([-1,1])
    
      """read image"""  
      with open(image_path, 'rb') as imgpath:
        magic, n, row, column = struct.unpack('>IIII', imgpath.read(16))
        self.train_image      = np.fromfile(imgpath, dtype=np.uint8)
        self.train_image      = self.train_image.reshape([-1,784])/255
        
      return self.train_image, self.train_label
  
    def update_test_img(self):
      print("Read image and labels" )
      """path join"""
      label_path = os.path.join(self.path,'t10k-labels-idx1-ubyte')
      image_path = os.path.join(self.path,'t10k-images-idx3-ubyte')
      
      """read lable"""
      with open(label_path, 'rb') as lbpath:
        magic, n    = struct.unpack('>II', lbpath.read(8))
        self.test_label = np.fromfile(lbpath, dtype=np.uint8)
        self.test_label = self.test_label.reshape([-1,1])
    
      """read image"""  
      with open(image_path, 'rb') as imgpath:
        magic, n, row, column = struct.unpack('>IIII', imgpath.read(16))
        self.test_image       = np.fromfile(imgpath, dtype=np.uint8)
        self.test_image       = self.test_image.reshape([-1,784])/255

      return self.test_image, self.test_label
  
    def update_train_batch(self,batch=200):
        self.batch = batch
        img_lb = np.concatenate([self.train_image, self.train_label], axis=1)
        np.random.shuffle(img_lb)
        row_rand = img_lb[0:self.batch,:]
        self.train_image_batch = row_rand[:,0:784]
        self.train_label_batch = np.array(row_rand[:,784], dtype='uint8')
        if(self.one_hot==True):
            label_one_hot = np.zeros([self.batch, 10])
            for i in range(self.batch):
                label_one_hot[i, self.train_label_batch[i]] = 1
            self.train_label_batch = label_one_hot
        return self.train_image_batch, self.train_label_batch

    def update_test_batch(self,batch=200):
        self.batch = batch
        img_lb = np.concatenate([self.test_image, self.test_label], axis=1)
        np.random.shuffle(img_lb)
        row_rand = img_lb[0:self.batch,:]
        self.test_image_batch = row_rand[:,0:784]
        self.test_label_batch = np.array(row_rand[:,784], dtype='uint8')
        if(self.one_hot==True):
            label_one_hot = np.zeros([self.batch, 10])
            for i in range(self.batch):
                label_one_hot[i, self.test_label_batch[i,1]] = 1
            self.test_label_batch = label_one_hot
        return self.test_image_batch, self.test_label_batch
    
    
# x = mnist_img()
# x.update_train_img()
# x.update_test_img()
# x.update_train_batch()
# x.update_test_batch()

# image = x.test_image_batch[5].reshape([28,28])
# label = x.test_label_batch[5]
# print(label)
# for i in range(10):
#     if(label[i]==1):
#         print(i)
# plt.figure("image")
# plt.imshow(image, cmap ='gray')



# img, lb = mnist_img()

# image = img
# image0 = image[0].reshape([28, 28])
# image_sh = image0.copy()
# print(image[0].reshape([28,28]))
# img = np.zeros((300,300))
# # cv2.imshow('img', image_sh)
# # cv2.waitKey(0)

# for i in range(10):
#     if(lb[0,i]==1):
#         print(i)
# plt.figure("image")
# plt.imshow(image_sh, cmap ='gray')






import numpy as np
import struct
import matplotlib.pyplot as plt
import random
from RBFNetwork import rbf as NW



def loadImage(filename):
    file = open(filename,"rb")
    buf = file.read()

    index = 0  #指针标志
    #读取头
    magic,image_num,rows_num,cols_num = struct.unpack_from('>4I',buf,index)
    index += struct.calcsize('>4I')

    image_set = []

    for i in range(image_num):
        fmt = '>' + str(rows_num * cols_num) + 'B'
        im = struct.unpack_from(fmt,buf,index)
        index += struct.calcsize(fmt)
        #im = np.array(im)
        #im = im.reshape(rows_num,cols_num)
        image_set.append(list(im))
    print(image_set[0])
    return image_set

def loadLabel(filename):
    file = open(filename,"rb")
    buf = file.read()

    index = 0

    magic,pic_num = struct.unpack_from('>2I',buf,index)
    index += struct.calcsize('>2I')

    image_label = []

    for i in range(pic_num):
        label = struct.unpack_from('>1B',buf,index)
        index += struct.calcsize('>1B')
        image_label.append(label[0])
    return image_label


def createMap(image_set,image_label):
    image_dct = {}
    for i,label in enumerate(image_label):
       image = image_set[i]
       if label in image_dct.keys():  #已有当前标签
           image_dct[label].append(image)
       else:
           image_dct[label] = [image]

    return image_dct

def imshowImage(image):
    image = np.array(image)
    image.reshape(28,28)
    fig = plt.figure()
    plot_window = fig.add_subplot(111)
    plt.imshow(image, cmap='gray')
    plt.show()


def chooseCenterFunc(center_num,image_dct):
   center_set = []
   times = int(center_num / len(image_dct))

   for i in range(len(image_dct)):
       image_set = image_dct[i]
       length = len(image_set) - 1
       for j in range(times):
           center_set.append(image_set[random.randint(0,length)])

   #剩余的除不尽的顶点随机补充
   center_num -= (times * len(image_dct))
   for i in range(center_num):
       label = random.randint(0,9)
       image_set = image_dct[label]
       index = random.randint(0,len(image_set) - 1)
       center_set.append(image_set[index])
   return center_set

def chooseSubSet(pic_num,image_dct):
    s_image_set= []
    s_image_label = []
    times = int(pic_num / len(image_dct))

    for key in image_dct.keys():
        image_set = image_dct[key]
        for i in range(times):
            s_image_set.append(image_set[i])
            s_image_label.append(image_label[i])
    return s_image_set,s_image_label

if __name__ == "__main__":
    image_set = loadImage('train\\train-images.idx3-ubyte')
    image_label = loadLabel('train\\train-labels.idx1-ubyte')
    image_dct = createMap(image_set,image_label)

    test_set = loadImage('train\\t10k-images.idx3-ubyte')
    test_label = loadLabel('train\\t10k-labels.idx1-ubyte')

    pic_num = 5000
    s_image_set,s_image_label = chooseSubSet(pic_num,image_dct)


    center_num = 100
    rbf = NW.RBF(center_num,s_image_set,s_image_label,10,0.2,chooseCenterFunc(center_num,image_dct))
    rbf.train(1000)
    test = test_set[0]
    n,y = rbf.rbfCount(test)
    print(test_label[0])
    for i,val in enumerate(y):
        print("%d : %f"%(i,val))

    test = test_set[1]
    n,y = rbf.rbfCount(test)
    print(test_label[1])
    for i,val in enumerate(y):
        print("%d : %f"%(i,val))

    test = test_set[2]
    n,y = rbf.rbfCount(test)
    print(test_label[2])
    for i,val in enumerate(y):
        print("%d : %f"%(i,val))

    test = test_set[3]
    n,y = rbf.rbfCount(test)
    print(test_label[3])
    for i,val in enumerate(y):
        print("%d : %f"%(i,val))



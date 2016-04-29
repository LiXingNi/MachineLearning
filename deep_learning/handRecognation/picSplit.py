# _*_ coding:utf-8 _*_
from cv2 import *
import numpy as np
from leNetDeserialize import *

def splitImage(path):
    im = imread(path)
    r_im = 255 - im
    g_im = cvtColor(r_im,COLOR_BGR2GRAY)
    b_im = threshold(g_im,50,255,THRESH_BINARY)[1]
    b_im = b_im.astype('float32') / 255.0

#split org image along cols
    split_cols = []
    space = True
    cols_max = np.argmax(b_im,axis = 0)

    #find split cols
    for col,row in enumerate(cols_max):
        if b_im[row,col] != 0 and space is True:
            split_cols.append(col)
            space = False
        else:
            if b_im[row,col] == 0 and space is False:
                split_cols.append(col)
                space = True

    #begin split
    spcol_ims = []
    for i in range(len(split_cols) / 2):
        spcol_ims.append(b_im[:,split_cols[2 * i] : split_cols[2 * i + 1]])
        

#split splited_col image along rows
        
    #find split row for every image
    sp_ims = []
    for c_im in spcol_ims:
        beg_row = -1
        last_row = -1
        for c_row in range(c_im.shape[0]):
            row_sum = c_im[c_row,:].sum()
            
            if row_sum != 0:
                if beg_row == -1:
                    last_row = beg_row = c_row
                else:
                    last_row = c_row
        sp_ims.append(c_im[beg_row:last_row,:])

    return sp_ims
        

def resizeImage(src,show_image = False):
    sz = np.max(src.shape)
    row = src.shape[0]
    col = src.shape[1]
    
    dst = np.zeros((sz,sz),np.float32)

    if sz == row:
        beg_col = (sz / 2) - (col / 2)
        dst[:,beg_col : beg_col + col] = src
    else:
        beg_row = (sz / 2) - (row / 2)
        dst[beg_row : beg_row + row,:] = src

    dst = resize(dst,(20,20), interpolation=INTER_AREA)

    f_dst = np.zeros((28,28),np.float32)
    f_dst[4:24,4:24] = dst

    if show_image is True:
        namedWindow('d', WINDOW_NORMAL)
        imshow('d',f_dst)
        waitKey()
    return f_dst
    

if __name__ == "__main__":
    sp_ims = splitImage("1.jpg")

    test_ims = []
    for im in sp_ims:
        t_im = resizeImage(im, True)
        test_ims.append(t_im.reshape(1,28 * 28))

    print ("split image ops")

    classifier = leNet()
    for im in test_ims:
        print(classifier.prediction(im))
    #trainLeNet(test_image = test_ims)



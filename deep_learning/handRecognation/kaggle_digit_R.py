# _*_ coding:utf-8 _*_
from cv2 import *
import numpy as np
from leNetDeserialize import *
import csv

def testFunc(file_name = "test.csv",out_file = "result.csv"):
    csv_reader = csv.reader(open(file_name,'r'))
    first_line = csv_reader.next()
    classifier = leNet()
    result = []
    for im in csv_reader:
        im = np.asarray(im,'float32')
        im = im / 255.0
        pred = classifier.prediction(im.reshape(1,28*28))
        result.append(pred[0])

    arranged_result = [(i+1,val) for i,val in enumerate(result)]

    wfile_obj = open(out_file,'wb')
    csv_writer = csv.writer(wfile_obj)
    csv_writer.writerow(('ImageId','Label'))
    csv_writer.writerows(arranged_result)
    wfile_obj.close()

    return result



if __name__ == "__main__":
    result = testFunc()






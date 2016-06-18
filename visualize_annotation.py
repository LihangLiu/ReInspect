"""train.py is used to generate and train the
ReInspect deep network architecture."""

import numpy as np
import json
import os
import cv2
import random
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)
from utils.annolist import AnnotationLib as al

def cut_image(impath_list):
    for impath in impath_list:
        try:
            image = imread(impath)
        except:
            print impath
        if image.shape != (480,640,3):
            print impath, image.shape
            image = image[-480:,:640,:]
            imsave(impath,image)

if __name__ == "__main__":
    # txtfile = './multi_scene_data/annnotation/tianmu/images_640_480/tianmu.txt'
    # impath_list = [ os.path.join(os.path.dirname(os.path.realpath(txtfile)), x.strip()) 
    #                         for x in open(txtfile,'r').readlines()]
    # cut_image(impath_list)
    # exit()
    idlfile = './multi_scene_data/annnotation/second_carteen/train.idl'
    annolist = al.parse(idlfile)
    annos = [x for x in annolist]
    cnt = 0
    max_num = 0
    for anno in annos:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
        image = imread(anno.imageName)
        # image = image[-480:,:640,:]
        cnt += len(anno.rects)
        max_num = max(max_num,len(anno.rects))
        for r in anno.rects:
            cv2.rectangle(image,  (int(r.x1),int(r.y1)), (int(r.x2),int(r.y2)),
                                                        (255,0,0), 2)
        # plt.imshow(image)
        # plt.show()

    print cnt
    print max_num

# python  
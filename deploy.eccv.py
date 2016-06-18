import cv2
import json
import os
import copy
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.misc import imread, imsave
from IPython import display
import caffe
import apollocaffe # Make sure that caffe is on the python path:

from utils.annolist import AnnotationLib as al
from train_boost_ip_split import forward
from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)

def prepocessd_image(raw_img, data_mean):
	img_mean = image_to_h5(raw_img, data_mean, image_scaling=1.0)
	return {"raw": raw_img, "image": img_mean}


def forward_test(net, inputs, net_config, enable_ip_split=True):
	net.phase = 'test'
	bbox_list, conf_list = forward(net, inputs, net_config, deploy=True, enable_ip_split=enable_ip_split)

	img = np.copy(inputs["raw"])
	all_rects = [[[] for x in range(net_config["grid_width"])] for y in range(net_config["grid_height"])]
    	pix_per_w = net_config["img_width"]/net_config["grid_width"]
	pix_per_h = net_config["img_height"]/net_config["grid_height"]
    	for n in range(len(bbox_list)):
        	for k in range(net_config["grid_height"] * net_config["grid_width"]):
            		y = int(k / net_config["grid_width"])
            		x = int(k % net_config["grid_width"])
            		bbox = bbox_list[n][k]
		    	conf = conf_list[n][k,1].flatten()[0]
		    	abs_cx = pix_per_w/2 + pix_per_w*x + int(bbox[0,0,0])
		    	abs_cy = pix_per_h/2 + pix_per_h*y+int(bbox[1,0,0])
		    	w = bbox[2,0,0]
		    	h = bbox[3,0,0]
			if conf < 0.9:
				continue
		    	all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    	acc_rects = stitch_rects(all_rects, net_config)
    
    	for rect in acc_rects:
        	if rect.true_confidence < 0.9:
            		continue
        	cv2.rectangle(img, 
                      (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)), 
                      (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)), 
                      (255,0,0),
                          2)
    	return img

if __name__ == '__main__':
    	# load video 
	config = json.load(open("config_boost_ip_split.json", 'r'))
	net_config = config['net']
	data_mean = load_data_mean(config["data"]["idl_mean"], 
	                           config["net"]["img_width"], 
	                           config["net"]["img_height"], image_scaling=1.0)

	# init apollocaffe
	enable_ip_split = True
	apollocaffe.set_random_seed(config["solver"]["random_seed"])
	apollocaffe.set_device(0)
	net = apollocaffe.ApolloNet()
	net.phase = 'test'
	dummy_input =  {"image": np.zeros((1,3,640,480))}
	forward(net, dummy_input, net_config, deploy=True, enable_ip_split=enable_ip_split)

	# # # # choose init weights
	# second carteen
	net_list_SC = [("./tmp/saved/second_carteen_0_100.h5", '.0.jpg'),
		   ("./tmp/saved/second_carteen_1_600.h5", '.1.jpg'),
		   ("./tmp/saved/second_carteen_2_1100.h5", '.2.jpg'),
		   ("./tmp/saved/second_carteen_3_1800.h5", '.3.jpg')]
	# net_list_SC = [("./data/brainwash_800000.h5",'.-1.jpg')]
	test_dir_SC = './multi_scene_data/pre_data/images_eccv/second_carteen'

	# laoximen
	net_list_LXM = [("./tmp/saved/laoximen_0_100.h5", '.0.jpg'),
		   ("./tmp/saved/laoximen_1_100.h5", '.1.jpg'),
		   ("./tmp/saved/laoximen_2_1500.h5", '.2.jpg'),
		   ("./tmp/saved/laoximen_3_1100.h5", '.3.jpg')]
	# net_list_LXM = [("./data/brainwash_800000.h5",'.-1.jpg')]
	test_dir_LXM = './multi_scene_data/pre_data/images_eccv/laoximen'

	# tianmulu
	net_list_TML = [("./tmp/saved/tianmu_0_200.h5", '.0.jpg'),
		   ("./tmp/saved/tianmu_1_700.h5", '.1.jpg'),
		   ("./tmp/saved/tianmu_2_1500.h5", '.2.jpg'),
		   ("./tmp/saved/tianmu_3_1400.h5", '.3.jpg')]
	# net_list_TML = [("./data/brainwash_800000.h5",'.-1.jpg')]
	test_dir_TML = './multi_scene_data/pre_data/images_eccv/tianmu'


	#### deploy
	net_list = net_list_SC
	test_dir = test_dir_SC

	output_dir = test_dir+'_output'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for x in net_list:
		print x
		weights = x[0]
		img_suffix = x[1]
		net.load(weights)
		for imgname in os.listdir(test_dir):
			impath = os.path.abspath(os.path.join(test_dir, imgname))
			img = imread(impath)
			inputs = prepocessd_image(img, data_mean)
			annotated_img = forward_test(net, inputs, net_config, enable_ip_split=enable_ip_split)
			imsave(os.path.join(output_dir,imgname+img_suffix), annotated_img)




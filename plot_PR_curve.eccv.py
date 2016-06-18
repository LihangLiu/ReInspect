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
from train_boost_ip_split import forward,load_idl, overlap_union
from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)

def get_accuracy(anno, bbox_list, conf_list, threshold):
	count_anno = 0.0
	for r in anno:
		count_anno += 1
	pix_per_w = net_config["img_width"]/net_config["grid_width"]
	pix_per_h = net_config["img_height"]/net_config["grid_height"]

	all_rects = [[[] for x in range(net_config["grid_width"])] for y in range(net_config["grid_height"])]
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
			if conf < threshold:
				continue
			all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

	acc_rects = stitch_rects(all_rects, net_config)
	count_cover = 0.0
	count_error = 0.0
	count_pred = 0.0
	for rect in acc_rects:
		if rect.true_confidence < threshold:
			continue
		else:
			count_pred += 1
			x1 = rect.cx - rect.width/2.
			x2 = rect.cx + rect.width/2.
			y1 = rect.cy - rect.height/2.
			y2 = rect.cy + rect.height/2.
			iscover = False
			for r in anno:
				if overlap_union(x1,y1,x2,y2, r.x1,r.y1,r.x2,r.y2) >= 0.5:    # 0.2 is for bad annotation
					iscover = True
					break
			if iscover:
				count_cover += 1
			else:
				count_error += 1

	return (count_cover, count_error, count_anno, count_pred)

def forward_test(net, config, thresholds_list,enable_ip_split):
	"""Trains the ReInspect model using SGD with momentum
	and prints out the logging information."""

	# # # init arguments # # #
	net_config = config["net"]
	data_config = config["data"]
	solver = config["solver"]

	image_mean = load_data_mean(
		data_config["idl_mean"], net_config["img_width"],
		net_config["img_height"], image_scaling=1.0)

	# # # load image data # # # 
	test_gen = load_idl(data_config["boost_test_idl"],
	                image_mean, net_config, jitter=False, if_random=False)

	net.phase = 'test'
	cc_dict = {}
	ce_dict = {}
	ca_dict = {}
	cp_dict = {}
	for threshold in thresholds_list:
		cc_dict[threshold] = []
		ce_dict[threshold] = []
		ca_dict[threshold] = []
		cp_dict[threshold] = []
	for _ in range(solver["test_iter"]):
		input_en = test_gen.next()
		bbox_list, conf_list = forward(net, input_en, net_config,enable_ip_split=enable_ip_split)
		for threshold in thresholds_list:
			(cc,ce,ca, cp) = get_accuracy(input_en['anno'], bbox_list, conf_list, threshold)
			cc_dict[threshold].append(cc)
			ce_dict[threshold].append(ce)
			ca_dict[threshold].append(ca)
			cp_dict[threshold].append(cp)
	for threshold in thresholds_list:
		precision = np.sum(cc_dict[threshold])/np.sum(cp_dict[threshold])
		recall = np.sum(cc_dict[threshold])/np.sum(ca_dict[threshold])
		f1 = 2*precision*recall/(precision+recall)
		print threshold, "%.03f %.03f %.03f " % (1-precision,recall,f1)

    

    



if __name__ == '__main__':
    	# load video 
	config = json.load(open("config_boost_ip_split.json", 'r'))
	net_config = config['net']

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
	net_list = ["./tmp/saved/second_carteen_0_100.h5",
		   "./tmp/saved/second_carteen_1_600.h5", 
		   "./tmp/saved/second_carteen_2_1100.h5",
		   "./tmp/saved/second_carteen_3_1800.h5"]
	# net_list = ["./data/brainwash_800000.h5"]
	thresholds_list = [0.001,0.003,0.006,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.96,0.99,0.993,0.996,0.999]
	config['data']['boost_test_idl'] = "./multi_scene_data/annnotation/second_carteen/test.idl"
	

	for i,weights in enumerate(net_list):
		# print weights
		net.load(weights)
		print 'pr%d=['%i
		forward_test(net, config, thresholds_list,enable_ip_split=enable_ip_split)
		print '];'




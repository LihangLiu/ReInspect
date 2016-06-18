import cv2
import json
import copy
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.misc import imread
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

def load_video_file(video_file, net_config, data_mean):
	cap = cv2.VideoCapture(video_file)
	if not cap.isOpened():
	    print "error open"
	    return

	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	while True:
	    flag, frame = cap.read()
	    if flag:
	        # The frame is ready and already captured
	        #cv2.imshow('video', frame)
	        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	        raw_img = frame[:net_config['img_height'], :net_config['img_width'], :]
	        # raw_img[:180, : ,:] = 0
	        yield prepocessd_image(raw_img, data_mean)

	    else:
	        # The next frame is not ready, so we try to read it again
	        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
	        print "frame is not ready"
	        # It is better to wait for a while for the next frame to be ready
	        cv2.waitKey(1000)

	    if cv2.waitKey(10) == 27:
	        break
	    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
	        # If the number of captured frames is equal to the total number of frames,
	        # we stop
	        break
	cap.release()


def forward_test(net, inputs, net_config, enable_ip_split):
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

def forward_multi(nets_list, input_ens, net_config):
	imgs_list = []
	for i,input_en in enumerate(input_ens):
		imgs = []
		nets = nets_list[i]
		for j,net in enumerate(nets):
			img = forward_test(net, input_en, net_config, enable_ip_split=src_list[i][1][j][1])
			imgs.append(img)
		imgs_list.append(imgs)

	column = len(imgs_list)
	row = len(imgs_list[0])
	width = net_config["img_width"]
	height = net_config["img_height"]
	final_img = np.zeros((row*height, column*width, 3),dtype=np.uint8)
	for c in range(column):
		for r in range(row):
			img = imgs_list[c][r]
			img[:40,:,:] = 0
			final_img[r*height:(r+1)*height, c*width:(c+1)*width,:] = img
	# final_img = cv2.resize(final_img, (width, height)) 
	return final_img

def updatefig(*args):
    new_frame = forward_multi(nets_list, [input_gen.next() for input_gen in input_gens], config["net"])
    im.set_array(new_frame)
    output_video.write(new_frame)
    return im,

if __name__ == "__main__":
	# load config 
	config = json.load(open("config.json", 'r'))
	data_mean = load_data_mean(config["data"]["idl_mean"], 
	                           config["net"]["img_width"], 
	                           config["net"]["img_height"], image_scaling=1.0)

	# init apollocaffe
	apollocaffe.set_random_seed(config["solver"]["random_seed"])
	apollocaffe.set_device(0)

	# model and video source
	global src_list
	# src_list = [ ("./multi_scene_data/pre_data/video_640_480/second_carteen_02.mov",
	# 		[('./data/brainwash_800000.h5',False),
	# 		 ("./tmp/saved/second_carteen_3_1800.h5", True)]),
	# 	   ("./multi_scene_data/pre_data/video_640_480/laoximen.mov",
	# 		[('./data/brainwash_800000.h5',False),
	# 		 ("./tmp/saved/laoximen_3_1100.h5", True)]),
	# 	  ("./multi_scene_data/pre_data/video_640_480/tianmulu_03.mov",
	# 		[('./data/brainwash_800000.h5',False),
	# 		 ("./tmp/saved/tianmu_3_1400.h5", True)])]
	src_list = [ ("./multi_scene_data/pre_data/video_640_480/second_carteen_02.mov",
			[('./data/brainwash_800000.h5',False),
			 ("./tmp/saved/second_carteen_3_1800.h5", True)])]
	
	# load model and video
	global input_gens
	input_gens = []
	global nets_list
	nets_list = []
	for src in src_list:
		video_file = src[0]
		print video_file
		input_gen = load_video_file(video_file, config["net"], data_mean)
		input_gens.append(input_gen)

		input_en = input_gen.next()
		nets_file = src[1]
		nets = []
		for net_file in nets_file:
			weights = net_file[0]
			if_split = net_file[1]
			net = apollocaffe.ApolloNet()
			net.phase = 'test'
			forward(net, input_en, config["net"], deploy=True, enable_ip_split=if_split)
			net.load(weights)
			nets.append(net)
		nets_list.append(nets)


	# # load video
	# video_file = r'./multi_scene_data/pre_data/second_carteen_03.mov'
	# input_gen = load_video_file(video_file, config["net"], data_mean)

	# # init apollocaffe
	# apollocaffe.set_random_seed(config["solver"]["random_seed"])
	# apollocaffe.set_device(0)
	# net = apollocaffe.ApolloNet()
	# net.phase = 'test'
	# forward(net, input_gen.next(), config["net"], True)
	# # net.load("./data/brainwash_800000.h5")
	# net.load("./tmp/saved/second_carteen_3_1800.h5")

	# init output video
	column = len(nets_list)
	row = len(nets_list[0])
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	output_video = cv2.VideoWriter('./multi_scene_data/pre_data/eccv2.avi',fourcc, 20.0, 
				(column*config["net"]["img_width"],row*config["net"]["img_height"]))

	fig = plt.figure()
	new_frame = forward_multi(nets_list, [input_gen.next() for input_gen in input_gens], config["net"])
	im = plt.imshow(new_frame, cmap=plt.get_cmap('viridis'), animated=True)
	ani = animation.FuncAnimation(fig, updatefig, interval=1, blit=True)
	plt.show()

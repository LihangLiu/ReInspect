import cv2
import json
import copy
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.misc import imread
from IPython import display
import apollocaffe # Make sure that caffe is on the python path:

from utils.annolist import AnnotationLib as al
from train import load_idl, forward
from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)

def prepocessd_image(raw_img, data_mean):
	img_mean = image_to_h5(raw_img, data_mean, image_scaling=1.0)
	return {"raw": raw_img, "image": img_mean}

def load_video_file(video_file, net_config, data_mean):
	cap = cv2.VideoCapture(video_file)
	while not cap.isOpened():
	    cap = cv2.VideoCapture(video_file)
	    cv2.waitKey(1000)
	    print "Wait for the header"

	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	while True:
	    flag, frame = cap.read()
	    if flag:
	        # The frame is ready and already captured
	        #cv2.imshow('video', frame)
	        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	        raw_img = frame[-net_config['img_height']-1:-1, :net_config['img_width'], :]
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


def forward_test(net, inputs, net_config):
	net.phase = 'test'
	bbox_list, conf_list = forward(net, inputs, net_config, deploy=True)

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

def updatefig(*args):
    new_frame = forward_test(net, input_gen.next(), config["net"])
    im.set_array(new_frame)
    output_video.write(new_frame)
    return im,

# load video 
config = json.load(open("config.json", 'r'))
data_mean = load_data_mean(config["data"]["idl_mean"], 
                           config["net"]["img_width"], 
                           config["net"]["img_height"], image_scaling=1.0)

video_file = r'/home/pig/apollocaffe/data/end_to_end_people_detection/Video2.mp4'
input_gen = load_video_file(video_file, config["net"], data_mean)

# init apollocaffe
apollocaffe.set_random_seed(config["solver"]["random_seed"])
apollocaffe.set_device(0)
net = apollocaffe.ApolloNet()
net.phase = 'test'
forward(net, input_gen.next(), config["net"], True)
net.load("/home/pig/ReInspect/tmp/reinspect_10000.h5")

# init output video
fourcc = cv2.cv.CV_FOURCC(*'XVID')
output_video = cv2.VideoWriter('output.avi',fourcc, 20.0, (config["net"]["img_width"],config["net"]["img_height"]))



fig = plt.figure()
im = plt.imshow(forward_test(net, input_gen.next(), config["net"]), cmap=plt.get_cmap('viridis'), animated=True)
ani = animation.FuncAnimation(fig, updatefig, interval=1, blit=True)
plt.show()

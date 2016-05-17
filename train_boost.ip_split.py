"""train.py is used to generate and train the
ReInspect deep network architecture."""

import _init_paths
import cv2
import numpy as np
import json
import math
import time
import os
import random
import itertools
from scipy.misc import imread
import caffe
import apollocaffe
from apollocaffe.models import googlenet
from apollocaffe.layers import (Power, LstmUnit, Convolution, NumpyData,
                                Transpose, Filler, SoftmaxWithLoss, Reshape,
                                Softmax, Concat, Dropout, InnerProduct)

from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)
from utils.annolist import AnnotationLib as al
from utils.annolist.AnnotationLib import Annotation, AnnoRect
from utils.pyloss import MMDLossLayer

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def overlap_union(x1,y1,x2,y2,x3,y3,x4,y4):
    SI = max(0, min(x2,x4)-max(x1,x3)) * max(0, min(y2,y4)-max(y1,y3))
    SU = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - SI + 0.0
    return SI/SU

def get_accuracy(net, inputs, net_config, threshold = 0.9):
    bbox_list, conf_list = forward(net, inputs, net_config)
    anno = inputs['anno']
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
                if overlap_union(x1,y1,x2,y2, r.x1,r.y1,r.x2,r.y2) >= 0.5:
                    iscover = True
                    break
            if iscover:
                count_cover += 1
            else:
                count_error += 1

    return (count_cover, count_error, count_anno, count_pred)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def load_idl(idlfile, data_mean, net_config, jitter=True, if_random=True):
    """Take the idlfile, data mean and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = [x for x in annolist]
    for anno in annos:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
    while True:
        if if_random:
            random.shuffle(annos)
        for anno in annos:
            if jitter:
                jit_image, jit_anno = annotation_jitter(
                    anno, target_width=net_config["img_width"],
                    target_height=net_config["img_height"])
            else:
                jit_image = imread(anno.imageName)
                jit_anno = anno
            image = image_to_h5(jit_image, data_mean, image_scaling=1.0)
            boxes, box_flags = annotation_to_h5(
                jit_anno, net_config["grid_width"], net_config["grid_height"],
                net_config["region_size"], net_config["max_len"])
            yield {"imname": anno.imageName, "raw": jit_image, "image": image,
                   "boxes": boxes, "box_flags": box_flags, 'anno': jit_anno}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def load_imname_list(imfile):
    with open(imfile, 'r') as f:
        lines = f.readlines()
    imname_list = [os.path.join(os.path.dirname(os.path.realpath(imfile)), line.strip()) 
                    for line in lines]
    return imname_list

def generate_input_en(imname, data_mean, net_config):
    raw_image = imread(imname)
    image = image_to_h5(raw_image, data_mean, image_scaling=1.0)
    return {"imname": imname, "raw": raw_image, "image": image}

def convert_deploy_2_train(boot_deploy_list, data_mean, net_config,
                               threshold=0.9, jitter=True, if_random=True):
    annos = []
    cnt = 0
    pix_per_w = net_config["img_width"]/net_config["grid_width"]
    pix_per_h = net_config["img_height"]/net_config["grid_height"]
    for dic in boot_deploy_list:
        anno = Annotation()
        anno.imageName = dic["imname"]
        bbox_list = dic["bbox"]
        conf_list = dic["conf"]

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
        for rect in acc_rects:
            if rect.true_confidence >= threshold:
                r = AnnoRect()
                r.x1 = int(rect.cx - rect.width/2.)
                r.x2 = int(rect.cx + rect.width/2.)
                r.y1 = int(rect.cy - rect.height/2.)
                r.y2 = int(rect.cy + rect.height/2.)
                anno.rects.append(r)
        annos.append(anno)
        cnt += len(anno.rects)
    print 'deployed',len(annos),'images with', cnt,'heads'


    while True:
        if if_random:
            random.shuffle(annos)
        for anno in annos:
            if jitter:
                jit_image, jit_anno = annotation_jitter(
                    anno, target_width=net_config["img_width"],
                    target_height=net_config["img_height"])
            else:
                jit_image = imread(anno.imageName)
                jit_anno = anno
            image = image_to_h5(jit_image, data_mean, image_scaling=1.0)
            boxes, box_flags = annotation_to_h5(
                jit_anno, net_config["grid_width"], net_config["grid_height"],
                net_config["region_size"], net_config["max_len"])
            yield {"imname": anno.imageName, "raw": jit_image, "image": image,
                   "boxes": boxes, "box_flags": box_flags, 'anno': jit_anno}


# # # # # # # # # # # # new layers for ip split # # # # # # # # # # # # # # 

def DotMultiply(name, bottoms=[], tops=[]):
    tops_ = []
    if len(tops)==0:
        tops_.append(name)
    else:
        tops_ = tops
    s = """
      type: "DotMultiply"
      name: "%s"
      bottom: "%s"
      top: "%s"
      dot_multiply_param {
        weight_filler {
          type: "uniform" 
          value: 0.1
        }
      }""" % (name, bottoms[0], tops_[0])
    return s 

def Sum(name, bottoms=[], tops=[]):
    tops_ = []
    if len(tops)==0:
        tops_.append(name)
    else:
        tops_ = tops
    s = """
      type: "Sum"
      name: "%s"
      bottom: "%s"
      top: "%s"
      sum_param {
        bias_filler {
          type: "constant" 
          value: 0
        }
      }""" % (name, bottoms[0], tops_[0])
    return s

def MMDLoss(name, bottoms=[], tops=[], loss_weight=1.0):
    tops_ = []
    if len(tops)==0:
        tops_.append(name)
    else:
        tops_ = tops
    s = str("""
          type: 'MMDLoss'
          name: '%s'
          top: '%s'
          bottom: '%s'
          bottom: '%s'
          loss_weight: %s
          """ % (name, tops_[0], bottoms[0], bottoms[1], loss_weight))
    return s


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def generate_decapitated_googlenet(net, net_config):
    """Generates the googlenet layers until the inception_5b/output.
    The output feature map is then used to feed into the lstm layers."""

    google_layers = googlenet.googlenet_layers()
    google_layers[0].p.bottom[0] = "image"
    for layer in google_layers:
        if "loss" in layer.p.name:
            continue
        if layer.p.type in ["Convolution", "InnerProduct"]:
            for p in layer.p.param:
                p.lr_mult *= net_config["googlenet_lr_mult"]
        net.f(layer)
        if layer.p.name == "inception_5b/output":
            break

def generate_intermediate_layers(net):
    """Takes the output from the decapitated googlenet and transforms the output
    from a NxCxWxH to (NxWxH)xCx1x1 that is used as input for the lstm layers.
    N = batch size, C = channels, W = grid width, H = grid height."""

    net.f(Convolution("post_fc7_conv", bottoms=["inception_5b/output"],
                      param_lr_mults=[1., 2.], param_decay_mults=[0., 0.],
                      num_output=1024, kernel_dim=(1, 1),
                      weight_filler=Filler("gaussian", 0.005),
                      bias_filler=Filler("constant", 0.)))
    net.f(Power("lstm_fc7_conv", scale=0.01, bottoms=["post_fc7_conv"]))
    net.f(Transpose("lstm_input", bottoms=["lstm_fc7_conv"]))

def generate_ground_truth_layers(net, box_flags, boxes):
    """Generates the NumpyData layers that output the box_flags and boxes
    when not in deploy mode. box_flags = list of bitstring (e.g. [1,1,1,0,0])
    encoding the number of bounding boxes in each cell, in unary,
    boxes = a numpy array of the center_x, center_y, width and height
    for each bounding box in each cell."""

    old_shape = list(box_flags.shape)
    new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
    net.f(NumpyData("box_flags", data=np.reshape(box_flags, new_shape)))        # (300,1,5,1)

    old_shape = list(boxes.shape)
    new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
    net.f(NumpyData("boxes", data=np.reshape(boxes, new_shape)))                # (300,4,5,1)

def generate_lstm_seeds(net, num_cells):
    """Generates the lstm seeds that are used as
    input to the first lstm layer."""

    net.f(NumpyData("lstm_hidden_seed",
                    np.zeros((net.blobs["lstm_input"].shape[0], num_cells))))
    net.f(NumpyData("lstm_mem_seed",
                    np.zeros((net.blobs["lstm_input"].shape[0], num_cells))))

def get_lstm_params(step):
    """Depending on the step returns the corresponding
    hidden and memory parameters used by the lstm."""

    if step == 0:
        return ("lstm_hidden_seed", "lstm_mem_seed")
    else:
        return ("lstm_hidden%d" % (step - 1), "lstm_mem%d" % (step - 1))

def generate_lstm(net, step, lstm_params, lstm_out, dropout_ratio):
    """Takes the parameters to create the lstm, concatenates the lstm input
    with the previous hidden state, runs the lstm for the current timestep
    and then applies dropout to the output hidden state."""

    hidden_bottom = lstm_out[0]
    mem_bottom = lstm_out[1]
    num_cells = lstm_params[0]
    filler = lstm_params[1]
    net.f(Concat("concat%d" % step, bottoms=["lstm_input", hidden_bottom]))
    try:
        lstm_unit = LstmUnit("lstm%d" % step, num_cells,
                       weight_filler=filler, tie_output_forget=True,
                       param_names=["input_value", "input_gate",
                                    "forget_gate", "output_gate"],
                       bottoms=["concat%d" % step, mem_bottom],
                       tops=["lstm_hidden%d" % step, "lstm_mem%d" % step])
    except:
        # Old version of Apollocaffe sets tie_output_forget=True by default
        lstm_unit = LstmUnit("lstm%d" % step, num_cells,
                       weight_filler=filler,
                       param_names=["input_value", "input_gate",
                                    "forget_gate", "output_gate"],
                       bottoms=["concat%d" % step, mem_bottom],
                       tops=["lstm_hidden%d" % step, "lstm_mem%d" % step])
    net.f(lstm_unit)
    net.f(Dropout("dropout%d" % step, dropout_ratio,
                  bottoms=["lstm_hidden%d" % step]))

def generate_ip_split_layers(net, bottom_name, bottom_c_i, top_name, top_c_o):
    """ 
        bottom (n, c_i) 
        -> bottom.ip_multiply_i.top (n,c_i) 
        -> bottom.ip_sum_i.top (n, 1,1,1) 
        -> top (n, c_o,1,1)
    """
    ip_sum_concat = []
    for c in range(top_c_o):
        ip_mul_name = "%s.ip_multiply_%d.%s" % (bottom_name, c, top_name)
        # print ip_mul_name
        net.f(DotMultiply(name=ip_mul_name, bottoms=[bottom_name]))
        ip_sum_name = "%s.ip_sum_%d.%s" % (bottom_name, c, top_name)
        ip_sum_concat.append(ip_sum_name)
        net.f(Sum(name=ip_sum_name, bottoms=[ip_mul_name]))
    net.f(Concat(top_name, bottoms=ip_sum_concat, concat_dim=1))

def copy_param_of_ip_split(src_net, tar_net, bottom_name, bottom_c_i, top_name, top_c_o):
    """
        src_top.p0 (top_c_o, bottom_c_i)
        -> \sum{top_c_o}{bottom.ip_multiply_i.top.p0 (bottom_c_i, 1, 1, 1)}
        src_top.p1 (top_c_o)
        -> \sum{top_c_o} bottom.ip_sum_i.top.p0 (1)
    """
    src_bottom_p0 = src_net.params[top_name+'.p0'].data
    src_bottom_p1 = src_net.params[top_name+'.p1'].data
    for c_o in range(top_c_o):
        ip_mul_name = "%s.ip_multiply_%d.%s" % (bottom_name, c_o, top_name)
        ip_sum_name = "%s.ip_sum_%d.%s" % (bottom_name, c_o, top_name)
        tar_net.params[ip_mul_name+'.p0'].data[:] = src_bottom_p0[c_o, :]
        tar_net.params[ip_sum_name+'.p0'].data[0] = src_bottom_p1[c_o]


def generate_inner_products(net, step, filler):
    """Inner products are fully connected layers. They generate
    the final regressions for the confidence (ip_soft_conf),
    and the bounding boxes (ip_bbox)"""

    net.f(InnerProduct("ip_conf%d" % step, 2, bottoms=["dropout%d" % step],
                       output_4d=True,
                       weight_filler=filler))
    net.f(InnerProduct("ip_bbox_unscaled%d" % step, 4,
                       bottoms=["dropout%d" % step], output_4d=True, 
                       weight_filler=filler))
    net.f(Power("ip_bbox%d" % step, scale=100,
                bottoms=["ip_bbox_unscaled%d" % step]))
    net.f(Softmax("ip_soft_conf%d" % step, bottoms=["ip_conf%d"%step]))

def generate_inner_products_using_ip_split(net, step, filler):
    bottom_name = "dropout%d" % step
    generate_ip_split_layers(net, bottom_name, 250, 
                                  "ip_conf%d"%step, 2)
    generate_ip_split_layers(net, bottom_name, 250, 
                                  "ip_bbox_unscaled%d"%step, 4)
    net.f(Power("ip_bbox%d" % step, scale=100,
                bottoms=["ip_bbox_unscaled%d" % step]))
    net.f(Softmax("ip_soft_conf%d" % step, bottoms=["ip_conf%d"%step]))

def generate_losses(net, net_config):
    """Generates the two losses used for ReInspect. The hungarian loss and
    the final box_loss, that represents the final softmax confidence loss"""

    net.f("""
          name: "hungarian"
          type: "HungarianLoss"
          bottom: "bbox_concat"
          bottom: "boxes"
          bottom: "box_flags"
          top: "hungarian"
          top: "box_confidences"
          top: "box_assignments"
          loss_weight: %s
          hungarian_loss_param {
            match_ratio: 0.5
            permute_matches: true
          }""" % net_config["hungarian_loss_weight"])
    net.f(SoftmaxWithLoss("box_loss",
                          bottoms=["score_concat", "box_confidences"],
                          ignore_label=net_config["ignore_label"]))

def forward(net, input_data, net_config, deploy=False, enable_ip_split=True):
    """Defines and creates the ReInspect network given the net, input data
    and configurations."""

    net.clear_forward()
    if deploy:
        image = np.array(input_data["image"])
    else:
        image = np.array(input_data["image"])
        box_flags = np.array(input_data["box_flags"])               # (1,300,1,5,1)
        boxes = np.array(input_data["boxes"])                       # (1,300,4,5,1)


    net.f(NumpyData("image", data=image))
    generate_decapitated_googlenet(net, net_config)
    generate_intermediate_layers(net)
    if not deploy:
        generate_ground_truth_layers(net, box_flags, boxes)
    generate_lstm_seeds(net, net_config["lstm_num_cells"])

    filler = Filler("uniform", net_config["init_range"])
    concat_bottoms = {"score": [], "bbox": []}
    lstm_params = (net_config["lstm_num_cells"], filler)
    for step in range(net_config["max_len"]):
        lstm_out = get_lstm_params(step)
        generate_lstm(net, step, lstm_params,
                      lstm_out, net_config["dropout_ratio"])
        if enable_ip_split:
            generate_inner_products_using_ip_split(net, step, filler)
        else:
            generate_inner_products(net, step, filler)

        concat_bottoms["score"].append("ip_conf%d" % step)
        concat_bottoms["bbox"].append("ip_bbox%d" % step)

    net.f(Concat("score_concat", bottoms=concat_bottoms["score"], concat_dim=2))
    net.f(Concat("bbox_concat", bottoms=concat_bottoms["bbox"], concat_dim=2))


    if not deploy:
        generate_losses(net, net_config)

    bbox = [np.array(net.blobs["ip_bbox%d" % j].data)           # [(300,4,1,1)] * 5
            for j in range(net_config["max_len"])]
    conf = [np.array(net.blobs["ip_soft_conf%d" % j].data)      # [(300,2,1,1)] * 5
            for j in range(net_config["max_len"])]
    return (bbox, conf)
    

def add_MMD_loss_layer(target_net, src_net, MMD_config):
    # # # loss of ip split
    for layers, loss_weight in zip(MMD_config['layers'], MMD_config['loss_weights']):
        if loss_weight == 0:
            continue
        for bottom0 in layers:
            bottom1 = 'src_' + bottom0
            target_net.f(NumpyData(bottom1, data=src_net.blobs[bottom0].data))
            top = bottom0 +'_loss'
            target_net.f(MMDLoss(name=top, bottoms=[bottom0, bottom1], 
                                        loss_weight=loss_weight))


        
def net_convert_ip_2_ip_split(src_net, tar_net, input_en, net_config):
    forward(tar_net, input_en, net_config)
    tar_net.copy_params_from(src_net)
    for step in range(net_config["max_len"]):
        copy_param_of_ip_split(src_net, tar_net, "dropout%d" % step, 
                                250, "ip_conf%d"%step, 2)
        copy_param_of_ip_split(src_net, tar_net, "dropout%d" % step, 
                                250, "ip_bbox_unscaled%d"%step, 4)
    return tar_net



def train(config):
    """Trains the ReInspect model using SGD with momentum
    and prints out the logging information."""

    # # # init arguments # # #
    net_config = config["net"]
    data_config = config["data"]
    solver = config["solver"]
    logging = config["logging"]
    MMD_config = config["MMD"]

    image_mean = load_data_mean(
        data_config["idl_mean"], net_config["img_width"],
        net_config["img_height"], image_scaling=1.0)

    # # # load image data # # # 
    re_train_gen = load_idl(data_config["reinspect_train_idl"],
                                   image_mean, net_config)
    re_test_gen = load_idl(data_config["reinspect_test_idl"],
                        image_mean, net_config, jitter=False, if_random=False)
    boost_test_gen = load_idl(data_config["boost_test_idl"],
                        image_mean, net_config, jitter=False, if_random=False)
    boost_imname_list = load_imname_list(data_config['boost_idl'])


    # # # init apollocaffe # # # 
    # source net
    src_net_ = apollocaffe.ApolloNet()
    net_config["ignore_label"] = -1
    forward(src_net_, re_test_gen.next(), net_config, enable_ip_split=False)
    if solver["weights"]:
        src_net_.load(solver["weights"])
    else:
        src_net_.load(googlenet.weights_file())

    # transform inner product layer of src_net 
    src_net = apollocaffe.ApolloNet()
    net_convert_ip_2_ip_split(src_net_, src_net, re_test_gen.next(), net_config)

    # boost net with MMD loss layer
    boost_net = apollocaffe.ApolloNet()
    net_config["ignore_label"] = 0
    forward(boost_net, boost_test_gen.next(), net_config)
    add_MMD_loss_layer(boost_net, src_net, MMD_config)
    boost_net.copy_params_from(src_net)

    # reinspect net with MMD loss layer
    re_net = apollocaffe.ApolloNet()
    net_config["ignore_label"] = 1
    forward(re_net, re_test_gen.next(), net_config)
    add_MMD_loss_layer(re_net, src_net, MMD_config)

    # # # init log # # # 
    loss_hist = {"train": [], "test": []}
    loggers = [
        apollocaffe.loggers.TrainLogger(logging["display_interval"],
                                        logging["log_file"]),
        apollocaffe.loggers.TestLogger(solver["test_interval"],
                                       logging["log_file"]),
        apollocaffe.loggers.SnapshotLogger(logging["snapshot_interval"],
                                           logging["snapshot_prefix"]),
        ]

    # # #  boost # # # 
    for i in range(solver["start_iter"], solver["max_iter"]):
        # # test and evaluation
        if i % solver["test_interval"] == 0:
            boost_net.phase = 'test'
            test_loss = []
            test_loss2 = []
            cc_list = []
            ce_list = []
            ca_list = []
            cp_list = []
            for _ in range(solver["test_iter"]):
                input_en = boost_test_gen.next()
                (cc,ce,ca, cp) = get_accuracy(boost_net, input_en, net_config)
                add_MMD_loss_layer(boost_net, src_net, MMD_config)
                test_loss.append(boost_net.loss)
                cc_list.append(cc)
                ce_list.append(ce)
                ca_list.append(ca)
                cp_list.append(cp)
            loss_hist["test"].append(np.mean(test_loss))
            precision = np.sum(cc_list)/np.sum(cp_list)
            recall = np.sum(cc_list)/np.sum(ca_list)
            print 'hungarian loss:', np.mean(test_loss)
            print input_en['imname']
            for layers, loss_weight in zip(MMD_config['layers'], MMD_config['loss_weights']):
                if loss_weight == 0:
                    continue
                print boost_net.blobs[layers[0]+'_loss'].diff[0],'*',
                for layer in layers:
                    print boost_net.blobs[layer+'_loss'].data[0],
                print ''
            print 'iterate:  %6.d error, recall, F1: (%.3f %.3f) -> %.3f' % (i, 1-precision, recall, 2*precision*recall/(precision+recall))
            

        # # deploy for subsequent training, select boost_iter images from boost_iter_max images
        if i % solver["boost_interval"] == 0:
            boost_deploy_list = []
            random.shuffle(boost_imname_list)                                    
            for imname in boost_imname_list[:solver["boost_iter_max"]]:                      # not all images are needed for boost training
                input_en = generate_input_en(imname, image_mean, net_config)
                (bbox, conf) = forward(boost_net, input_en, net_config, deploy=True)
                add_MMD_loss_layer(boost_net, src_net, MMD_config)
                mmd_losses = []
                for layers, loss_weight in zip(MMD_config['layers'], MMD_config['loss_weights']):
                    if loss_weight == 0:
                      continue
                    mmd_losses += [loss_weight*boost_net.blobs[x+'_loss'].data[0]
                                         for x in layers]
                boost_deploy_list.append({'imname':imname, 'bbox':bbox, 'conf':conf, 
                                        'MMDLoss':np.mean(mmd_losses)})
            boost_deploy_list = sorted(boost_deploy_list, 
                                    key=lambda x:x['MMDLoss'],reverse=solver['reverse'])[:solver['boost_iter']]
            thres = 0.9
            boot_train_gen = convert_deploy_2_train(boost_deploy_list, image_mean, net_config, 
                                                      threshold=thres, if_random=solver['random'])


        # # train # # 
        learning_rate = (solver["base_lr"] *
                         (solver["gamma"])**(i // solver["stepsize"]))

        # feed new data to source net to aid "add_MMD_loss_layer"
        forward(src_net, re_test_gen.next(), net_config) 

        # train on reinspect dataset
        re_net.phase = "train"
        re_net.copy_params_from(boost_net) 
        for _ in range(10):
            forward(re_net, re_train_gen.next(), net_config)
            add_MMD_loss_layer(re_net, src_net, MMD_config)
            if not math.isnan(re_net.loss):  
                re_net.backward()
            re_net.update(lr=learning_rate, momentum=solver["momentum"],
                       clip_gradients=solver["clip_gradients"])

        boost_net.copy_params_from(re_net)

        # train on boost dataset
        boost_net.phase = 'train'
        forward(boost_net, boot_train_gen.next(), net_config)
        add_MMD_loss_layer(boost_net, src_net, MMD_config)
        loss_hist["train"].append(boost_net.loss)
        if not math.isnan(boost_net.loss):      # loss may be "nan", caused by ignore label. 
            boost_net.backward()
        boost_net.update(lr=learning_rate, momentum=solver["momentum"],
                   clip_gradients=solver["clip_gradients"])
        for logger in loggers:
            logger.log(i, {'train_loss': loss_hist["train"],
                           'test_loss': loss_hist["test"],
                           'apollo_net': boost_net, 'start_iter': 0})


def main():
    """Sets up all the configurations for apollocaffe, and ReInspect
    and runs the trainer."""
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    if args.weights is not None:
        config["solver"]["weights"] = args.weights
    config["solver"]["start_iter"] = args.start_iter
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    print json.dumps(config['net'], indent=4, sort_keys=True)
    print json.dumps(config['solver'], indent=4, sort_keys=True)
    print json.dumps(config['MMD'], indent=4, sort_keys=True)

    train(config)

if __name__ == "__main__":
    main()


# python train_boost.ip_split.py --gpu=0 --config=config_boost.ip_split.json  --weights=./data/brainwash_800000.h5

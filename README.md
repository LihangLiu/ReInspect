<img src=http://russellsstewart.com/s/ReInspect_output.jpg></img>

# ReInspect Domain Adapation
Unsupervised domain adaptation architercture for pedestrian detection.
See <a href="https://github.com/LihangLiu93/Reinspect.report/blob/master/eccv2016submission.pdf" target="_blank">the paper</a> for details or the <a href="https://github.com/LihangLiu93/Reinspect.report/blob/master/Unsupervised%20Domain%20Adaptation%20for%20Pedestrian%20Detection.pdf" target="_blank">ppt</a> for a demonstration.

# Installation

## Prerequisite - install ApolloCaffe
Some new layers used for this project are implemented in my forked <a href="https://github.com/LihangLiu/apollocaffe">ApolloCaffe</a>. Please firstly pull and compile the ApolloCaffe. If you have problems installing ApolloCaffe, you can refer to http://apollocaffe.com.

	$ git clone https://github.com/LihangLiu93/apollocaffe.git

## Install ReInspect

With ApolloCaffe installed, you can run ReInspect with:

    $ git clone https://github.com/LihangLiu93/ReInspect.git
    $ cd reinspect

## Data

The data consists of:

1) The source domain data from <a href="https://github.com/Russell91/ReInspect">Russell's project</a>, which this project is built on. The data can be found <a href="http://datasets.d2.mpi-inf.mpg.de/brainwash/brainwash.tar">here</a>.

2) The target domain data collected for domain adaptation, please put the data from the following links into the corresponding directories.

	dir:./multi_scene_data/annnotation/second_carteen/     link: https://pan.baidu.com/s/1c16O1Hm       password: d9kc 
	
	dir:./multi_scene_data/pre_data/images_640_480/second_carteens[01-03]/    link: https://pan.baidu.com/s/1dFqhGWh      password: rs9b

## Run

	$ python train_boost_ip_split.py --gpu=0 --config=config_boost_ip_split.json  --weights=./data/brainwash_800000.h5

# Q&A

If you have further questions regarding the project, please email backchord at gmail dot com.



<img src=http://russellsstewart.com/s/ReInspect_output.jpg></img>

# ReInspect Domain Adapation
Unsupervised domain adaptation architercture for pedestrian detection.
See <a href="https://github.com/LihangLiu93/Reinspect.report/blob/master/eccv2016submission.pdf" target="_blank">the paper</a> for details or the <a href="https://github.com/LihangLiu93/Reinspect.report/blob/master/Unsupervised%20Domain%20Adaptation%20for%20Pedestrian%20Detection.pdf" target="_blank">ppt</a> for a demonstration.

## Installation
ReInspect depends on <a href="http://github.com/bvlc/caffe" target="_blank">Caffe</a> and requires
the <a href="http://apollocaffe.com">ApolloCaffe</a> pull request. With ApolloCaffe installed, you can run ReInspect with:

    $ git clone https://github.com/LihangLiu93/ReInspect.git
    $ cd reinspect
    $ python train_boost_ip_split.py --gpu=0 --config=config_boost_ip_split.json  --weights=./data/brainwash_800000.h5

Data should be placed in /path/to/reinspect/data/ and can be found <a href="http://datasets.d2.mpi-inf.mpg.de/brainwash/brainwash.tar">here</a>.
Data for domain adapatation, please email backchord at gmail dot com.



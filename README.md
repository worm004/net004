# Net004

This is a personal C++ CNN framework toy.

This toy is created to play with new CNN structures/ideas easier than Caffe for me ***on my macbook currently = =***.

Caffe relies on so many 3rd-party libraries which are not very suitable to quick test.

To prove the correctness of the code, two things have been done:

1. Several caffe models are converted into the net004 models.
2. Several forward and backward steps are conducted on each models, output resluts are compared between net004 and caffe.

Currently, following models are supported:

Base Net: cifar10, alexnet, gnetv4, sqnet\_res, resnet50, inception-res-v2, vgg16, gnetv1, sqnet1.0, dense121, resnet101, gnetv3, sqnet1.1, resnet152

Detection Net: yolov1, faster\_rcnn

Segmentation Net: fcn\_seg\_8s

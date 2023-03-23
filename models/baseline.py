import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor #, FastRCNNPredictor_DROPOUT
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import nn, Tensor
import torch
from torchvision.ops import RoIAlign

      
def baseline(num_classes = 10): #9 + background: Put only in the Box predictor head, the backbone is trained with COCO which has 91 classes!!
    # 9 are the affordances classes!!! We want to segment now by the affordances, not by the objects->affordances
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 

    model = torchvision.models.detection.mask_rcnn.maskrcnn_resneXt101_fpn(pretrained = True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one -> ESTO CON EL DROPOUT CAMBIA
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def object_detection_loss(out_maskrcnn):
    total_loss = 0
    for key, value in out_maskrcnn.items():
        total_loss += out_maskrcnn[key]
    return total_loss

def data_to_GPU(data_batch, device):
    #Send the data to the GPU!
    images_cuda = [_image.to(device) for _image in data_batch[0]] 
    targets_cuda = []
    for _target in data_batch[1]:
        for key, value in _target.items():
            _target[key] = _target[key].to(device)
        targets_cuda.append(_target)
    return images_cuda, targets_cuda
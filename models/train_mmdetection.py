import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from imantics import Polygons, Mask


def convert_to_coco_dataset(old_label, img, img_file):
    img = {'file_name': img_file, 'height': img.shape[0], 'width': img.shape[1], 'id': 0}
    
    for i in range(old_label):
        polygon_i = 2
        annotations_i = {'segmentation': polygon_i,  
                        'area': (bbox[2] - bbox[0])*(bbox[3] - bbox[1]),
                        'iscrowd': 0, 
                        'image_id': 0,
                        'bbox': old_label['bboxes'][i], 
                        'category_id': old_label['label'][i],
                        'id': 0}
    return 






# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'


device='cuda:0' # Set the device to be used for evaluation
config = mmcv.Config.fromfile(config) # Load the config
config.model.pretrained = None # Set pretrained to be None since we do not need pretrained model here
model = build_detector(config.model) # Initialize the detector
checkpoint = load_checkpoint(model, checkpoint, map_location=device) # Load checkpoint
model.CLASSES = checkpoint['meta']['CLASSES'] # Set the classes of models for inference


model.cfg = config # We need to set the model's cfg for inference


model.to(device) # Convert the model to GPU

model.eval() # Convert the model into evaluation mode